import os
import json
import re
import fitz
import numpy as np
import pandas as pd
from itertools import groupby
from PIL import Image
from paddleocr import PaddleOCR
import torch
import subprocess
import extract_msg
import pdfkit
import pytesseract
from skimage import io as skio
from scipy.ndimage import interpolation as inter
import cv2
from transformers import AutoProcessor
from transformers import AutoModelForTokenClassification
from transformers import LayoutLMv3ImageProcessor
import logging
from utils import AWSutils
import jpype
import asposecells
jpype.startJVM()
from asposecells.api import Workbook, SaveFormat, PdfSaveOptions

logger = logging.getLogger('ad_logger')

UPLOAD_FOLDER = '/flask_app/files/'
# ---- THIS DEPENDS ON THE MODEL ----
PROCESSOR_PATH = 'microsoft/layoutlmv3-base'
MODEL_PATH = 'DataIntelligenceTeam/LOC1.0'
lang = 'eng'
# -----------------------------------

# define id2label: list of entities the model was trained for
# ---- THIS DEPENDS ON THE MODEL ----
id2label = {
    0: 'others',
    1: 'lc_number',
    2: 'date_of_issue',
    3: 'applicant',
    4: 'beneficiary',
    5: 'port_of_loading',
    6: 'port_of_discharge',
    7: 'latest_date_of_shipment',
    8: 'description'
}

header_keys = ['lc_number', 'date_of_issue', 'applicant', 'beneficiary', 'port_of_loading','port_of_discharge','latest_date_of_shipment','description']
details_keys = []
# -----------------------------------

processor = AutoProcessor.from_pretrained(MODEL_PATH, apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

ppocr = PaddleOCR(lang='en', use_gpu=False)

logger.info('Model loaded')


def generate_pdf(doc_path, path):
    subprocess.call(['soffice',
                 #'--headless',
                 '--convert-to',
                 'pdf',
                 '--outdir',
                 path,
                 doc_path])
    return doc_path.replace(".docx", "").replace(".doc", "") + ".pdf"


def content_extraction(msg_file):
    try:
        msg = extract_msg.openMsg(msg_file)
        str_mail_msg = msg.body
        return str_mail_msg
    except(UnicodeEncodeError,AttributeError,TypeError) as e:
        pass


def DataImporter(msg_file):
    str_msg = content_extraction(msg_file)
    # encoding the message to UTF-8
    msg = str_msg.encode('utf-8', errors='ignore').decode('utf-8')
    return msg


def string_to_pdf(text, pdf_file):
    # Replace newlines with <br> tag
    text = text.replace('\n', '<br>')

    # Replace tab spaces with non-breaking spaces
    text = text.replace('\t', '&nbsp;' * 1)
    # Wrap the text in HTML tags
    html = f"<html><body>{text}</body></html>"
    pdfkit.from_string(html, pdf_file)


def process_docx(filePath,ocr):
    logger.info('process_docx start')

    pdf_filePath = generate_pdf(filePath, os.path.dirname(filePath))
    result = process_PDF(pdf_filePath,ocr)

    logger.info('process_docx end')
    return result


def process_msg(filePath, ocr):
    print('filePath',filePath)
    logger.info('process_msg start')
    pdf_filePath = os.path.join(os.path.dirname(filePath), "output.pdf")
    content_extraction(filePath)
    text = DataImporter(filePath)
    string_to_pdf(text, pdf_filePath)
    result = process_PDF(pdf_filePath, ocr)
    logger.info('process_msg end')
    return result

def process_excel(filePath, ocr):
    logger.info('process_excel end')
    # Load Excel file
    workbook = Workbook(filePath)
    pdf_filePath = os.path.join(os.path.dirname(filePath), "output.pdf")
    # Convert Excel to PDF
    workbook.save(pdf_filePath, SaveFormat.PDF)
    result = process_PDF(pdf_filePath, ocr)
    logger.info('process_excel end')
    return result


def elab(filePath,ocr):
    logger.info('elab start')
    # response = mockupElab(filePath)
    if ".pdf" in filePath.lower():
        elab_data = process_PDF(filePath,ocr)
    elif any(ext in filePath.lower() for ext in [".png", ".jpeg", ".jpg"]):
        elab_data = process_image(filePath)
    elif any(ext in filePath.lower() for ext in [".docx", ".doc"]):
        elab_data = process_docx(filePath,ocr)
    elif any(ext in filePath.lower() for ext in [".xlsx", ".xls",".XLS"]):
        elab_data = process_excel(filePath,ocr)
    elif any(ext in filePath.lower() for ext in [".msg"]):
        elab_data = process_msg(filePath,ocr)
    response = dict()
    count_pag = 0
    for page in elab_data:
        if len(page) != 0:
            response[count_pag] = structuredResponse(elab_data[page], count_pag)
        else:
            pass
        count_pag += 1
    logger.info('elab end')
    return unify_response(response), response


def mockupElab(filePath):
    response = dict()
    response[0] = structuredResponse("test", 0)
    return response


def compute_detection_index(expected_keys, found_keys, found_details, expected_details_keys):
    # TODO: UPDATE
    # for the key-value pairs, we check which labels we found wrt the set of expected labels
    details_total_score = 0  # for the details, we check if the values are not null
    if len(found_details) != 0:
        for row in found_details:
            row_count = 0
            for element in row:
                if isinstance(element, float) and element != np.nan:
                    row_count += 1
                elif isinstance(element, list) and len(element) != 0:
                    row_count += 1
                elif isinstance(element, str) and element != "":
                    row_count += 1
            row_score = row_count/len(expected_details_keys)
            details_total_score += row_score
        details_score = details_total_score/len(found_details)
    else:
        details_score = 0
    return ((len(found_keys)/len(expected_keys)) + details_score)/2


def prune_text(text):
    chars = "\\`*_{}[]()>#+-.!$"
    for c in chars:
        if c in text:
            text = text.replace(c, "\\" + c)
    return text


def structuredResponse(content, n_pag):
    # ---- THIS DEPENDS ON THE MODEL ----
    header_keys = ['lc_number', 'date_of_issue', 'applicant', 'beneficiary', 'port_of_loading',
                   'port_of_discharge','latest_date_of_shipment','description']
    det_keys = []
    # -----------------------------------

    result = dict()
    data = []

    if str(content) == "test":
        main_keys = header_keys
        details_keys = det_keys
        main_values = ['ILC22A001181',
                       '02/06/22',
                       'DD 25/11 MARINA DI CARRARA PORT MARITIME ITALIEN PORT ALGER ALGERIE',
                       'SOGETHERM 71 BD ABA CHOUAIB DOUKKALI 20000 CASABLANCA MAROC',
                       'DAB PUMPS S,P,A SEDE LEGALE 35035 MESTRINO (PD)- VIA M,POLO, 14-ITALY',
                       'EUROPEAN PORT',
                       'CASABLANCA PORT MOROCCO',
                       '30/06/22'
                       ]
        details_values = []
    else:
        values = []
        det_values = []
        details = dict()
        count = 0
        if len(content) != 0:
            content_main = content[0]
            content_details = content[1]
            if len(content_main) != 0:
                main_keys = content_main["labels"]
                main_values = content_main["values"]
            else:
                main_keys = []
                main_values = []
            if len(content_details) != 0:
                details_keys = content_details.columns.values.tolist()
                details_values = content_details.values.tolist()
            else:
                details_keys = det_keys
                details_values = []
        else:
            main_keys = []
            main_values = []
            details_keys = det_keys
            details_values = []

    # detection_index = compute_detection_index(keys, main_keys, details_values, det_keys)
    # TODO: UPDATE
    detection_index = 0.8
    result['detection_index'] = "{:.2f}".format(detection_index)

    header = dict()
    header['key'] = 'Header'
    header['type'] = 'Inputs'
    h_values = []
    for k,v in zip(main_keys, main_values):
        val = dict()
        val['key'] = k
        val['value'] = str(v).strip()
        val['state'] = 'INCOMPLETE'
        # val['coordinates'] = []
        h_values.append(val)
    header['value'] = h_values
    # header['coordinates'] = [[0,0], [0,0], [0,0], [0,0], [0,0]]
    header['page'] = n_pag + 1
    if len(h_values) != 0:
        data.append(header)

    details = dict()
    details['key'] = 'Details'
    details['type'] = 'Table'
    d_values = dict()
    d_values['header'] = details_keys
    d_data = []
    for row in details_values:
        stripped = [str(s).strip() for s in row]
        clean = [prune_text(str(s)) for s in stripped]
        d_data.append(clean)
    d_values['data'] = d_data
    details['value'] = d_values
    # details['coordinates'] = [[0,0], [0,0], [0,0], [0,0], [0,0]]
    details['page'] = n_pag + 1
    if len(d_data) != 0:
        data.append(details)

    result['data_to_review'] = data

    # with open("files/test.json", "w") as outfile:
    #     json.dump(result, outfile)
    return result


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def intersect(w, z):
    # this method will detect if there is any intersect between two boxes or not
    x1 = max(w[0], z[0]) #190  | 881  |  10
    y1 = max(w[1], z[1]) #90   | 49   | 273
    x2 = min(w[2], z[2]) #406  | 406  | 1310
    y2 = min(w[3], z[3]) #149  | 703  | 149
    if (x1 > x2 or y1 > y2):
        return 0
    else:
        # because sometimes in annotating, it is possible to overlap rows or columns by mistake
        # for very small pixels, we check a threshold to delete them
        area = (x2-x1) * (y2-y1)
        if (area > 0):  #500 is minumum accepted area
            return [int(x1), int(y1), int(x2), int(y2)]
        else:
            return 0

# calculates the verticle distance between boxes
def dist_height(y1,y2):
    return abs(y1- y2)


def mergeBoxes(df):
    xmin, ymin, xmax, ymax = [], [], [], []
    for i in range(df.shape[0]):
        box = df['bbox_column'].iloc[i]
        xmin.append(box[0])
        ymin.append(box[1])
        xmax.append(box[2])
        ymax.append(box[3])
    return [min(xmin), min(ymin), max(xmax), max(ymax)]


def mergeCloseBoxes(pr, bb, wr, threshold):
    idx = 0
    final_bbox =[]
    final_preds =[]
    final_words=[]

    for box, pred, word in zip(bb, pr, wr):
        if (pred=='others'):
            continue
        else:
            final_bbox.append(box)
            final_preds.append(pred)
            final_words.append(word)
            for b, p, w in zip(bb, pr, wr):
                if (p == 'others'):
                    continue
                elif (box==b): # we shouldn't check each item with itself
                   continue
                else:
                    XMIN, YMIN, XMAX, YMAX = box
                    xmin, ymin, xmax, ymax = b
                    intsc = intersect([XMIN, YMIN, XMAX+threshold, YMAX], [xmin-threshold, ymin, xmax, ymax])
                    if (intsc != 0 and pred==p):
                    #if(abs(XMAX - xmin) < treshold and abs(YMIN - ymin) < 10):
                        if(box in final_bbox):
                            final_bbox[idx]= [XMIN, min(YMIN, ymin), xmax, max(YMAX, ymax)]
                            final_words[idx] = word + ' ' + w
                            continue
        idx = idx +1
    return final_bbox, final_preds, final_words


def isInside(w, z):
    # return True if w is inside z, if z is inside w return false
    if(w[0] >= z[0] and w[1] >= z[1] and w[2] <= z[2] and w[3] <= z[3]):
        return True
    return False


def removeSimilarItems(final_bbox, final_preds, final_words):
    _bb =[]
    _pp=[]
    _ww=[]
    for i in range(len(final_bbox)):
        _bb.append(final_bbox[i])
        _pp.append(final_preds[i])
        _ww.append(final_words[i])
        for j in range(len(final_bbox)):
            if (final_bbox[i] == final_bbox[j]):
                continue
            elif (isInside(final_bbox[i], final_bbox[j]) and final_preds[i]==final_preds[j] ):
                # box i is inside box j, so we have to remove it
                _bb = _bb[:-1]
                _pp = _pp[:-1]
                _ww = _ww[:-1]
                continue
    return _bb, _pp, _ww


def createDataframe(labels, words):
    print('labels', labels)
    detail_dict = {}
    main_dict = {}
    for i in range(len(labels)):
        if labels[i] in details_keys:
            if labels[i] not in detail_dict:
                detail_dict[labels[i]] = [words[i]]
            else:
                detail_dict[labels[i]].append(words[i])
        elif labels[i] != "others":
            if labels[i] not in main_dict:
                main_dict[labels[i]] = [words[i]]
            else:
                main_dict[labels[i]].append(words[i])

    #df_main = pd.DataFrame.from_dict(main_dict, orient='index', columns=['labels', 'values'])
    df_main = pd.DataFrame.from_dict(main_dict, orient='index', columns=['values'])
    df_main['labels'] = df_main.index
    df = pd.DataFrame.from_dict(detail_dict, orient='index')
    df_details = df.transpose()
    return df_main, df_details


def process_form(preds, words, bboxes):
    logger.info('process_form start')
    # the following combines all labels (preds) with the associated words;
    # it can contain multiple instances of the same label (because of values
    # of more than 1 word)
    cmb_list = []
    for ix, (prediction, box) in enumerate(zip(preds, bboxes)):
        cmb_list.append([prediction, words[ix]])

    # the following groups the words with the same label
    grouper = lambda l: [[k] + sum((v[1::] for v in vs), []) for k, vs in groupby(l, lambda x: x[0])]
    list_final = grouper(cmb_list)
    lst_final = []
    for x in list_final:
        json_dict = dict()
        json_dict[x[0]] = (' ').join(x[1:])
        lst_final.append(json_dict)

    # TODO: check if replace with createDataframe
    columns = id2label.values()
    data = dict()
    for c in columns:
        data[c] = []
    for i in lst_final:  # list of dicts
        for k in i:  # dict
            v = i[k]  # value
            if k in columns and k != "others":  # if the key is one of the relevant predictions:
                data[k].append(v)

    # join the list of strings for each column and convert to a dataframe
    key_value_pairs = []
    for col in columns:
        if col != "others":
            val = ' '.join(data[col])
            key_value_pairs.append({'labels': col, 'values': val})
    df_main = pd.DataFrame(key_value_pairs)
    df_details = pd.DataFrame()

    logger.info('process_form end')
    return [df_main, df_details]


def process_PDF(filePath, ocr):
    logger.info('process_PDF start')
    # we unpack the PDF into multiple images
    doc = fitz.open(filePath)

    result = {}
    # for i in range(0, doc.page_count):
    # CAREFUL - in this case we consider ONLY the first page, assuming it contains the only relevant document (HR).
    # The better scenario would be to classify pages in order to process only HR (and not missing any of them if there were multiple ones).
    for i in range(0, len(doc)):
        page = doc.load_page(i)     # number of page
        zoom = 2                    # zoom factor
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix = mat, dpi = 300)
        if filePath[-4:] == ".pdf":
            imgOutput = filePath.replace(".pdf", "_{}.png".format(i))
        elif filePath[-4:] == ".PDF":
            imgOutput = filePath.replace(".PDF", "_{}.png".format(i))
        pix.save(imgOutput)
        # rotation = checkRotation(imgOutput)
        # print(rotation)
        # if rotation == -1:
        #     # img is blank: delete it?
        #     os.remove(imgOutput)
        #     result[(imgOutput.replace(UPLOAD_FOLDER, ""))] = []
        #     continue
        # elif rotation != 0:
        #     im = Image.open(imgOutput)
        #     rotated = im.rotate(-(int(rotation)), expand=True)
            # angle, skewed_image = correct_skew(rotated)
            # print(angle)
            # out = remove_borders(skewed_image)
            # cv2.imwrite(imgOutput, out)
            # out.save(imgOutput)
        # each image goes through the model
        pageResult = process_page(imgOutput, ocr)
        # result is saved in a dict-like shape to be returned
        result[(imgOutput.replace(UPLOAD_FOLDER, ""))] = pageResult

    # clean up function to delete local files - both the original pdf
    # (that was previously uploaded to S3) and the newly created images
    if filePath[-4:] == ".pdf":
        pattern = (filePath.replace(".pdf","")) + "*"
    elif filePath[-4:] == ".PDF":
        pattern = (filePath.replace(".PDF","")) + "*"
    cleanup(pattern)

    logger.info('process_PDF end')
    return result


def process_image(filePath):
    result = {(filePath.replace(UPLOAD_FOLDER, "")): process_page(filePath)}

    # clean up function to delete local files - both the original pdf
    # (that was previously uploaded to S3) and the newly created images
    pattern = filePath
    cleanup(pattern)

    return result


def process_page(filePath, ocr):
    logger.info('process_page start')
    # load image (at this stage all pdf pages have been transformed to images)
    image = Image.open(filePath).convert("RGB")
    bboxes, preds, l_words, image = infer(image, ocr)
    predicates = []
    for id in preds:
        predicates.append(id2label.get(id))
    dfs = process_form(predicates, l_words, bboxes)

    logger.info('process_page end')
    return dfs


def create_bounding_box(bbox_data, width_scale, height_scale):
    xs = []
    ys = []
    for x, y in bbox_data:
        xs.append(x)
        ys.append(y)

    left = int(max(0, min(xs) * width_scale))
    top = int(max(0, min(ys) * height_scale))
    right = int(min(1000, max(xs) * width_scale))
    bottom = int(min(1000, max(ys) * height_scale))

    return [left, top, right, bottom]


def aws_processor(image, width, height):
    # extract text from image
    response = AWSutils.detect_document_text(image)

    # process response to get text and bounding boxes
    words = []
    bboxes = []
    for item in response['Blocks']:
        if item['BlockType'] == 'WORD':
            words.append(item['Text'])
            bbox = item['Geometry']['BoundingBox']
            # rescale bbox coordinates to be within 0-1000 range
            x1 = int(bbox['Left'] * 1000)
            y1 = int(bbox['Top'] * 1000)
            x2 = int((bbox['Left'] + bbox['Width']) * 1000)
            y2 = int((bbox['Top'] + bbox['Height']) * 1000)
            bboxes.append((x1, y1, x2, y2))

    return words, bboxes


def paddle_processor(image,width,height):
    width, height = image.size
    width_scale = 1000 / width
    height_scale = 1000 / height

    # Perform OCR on the image
    results = ppocr.ocr(np.array(image))

    # Extract the words and bounding boxes from the OCR results
    words = []
    boxes = []
    for line in results:
        for bbox in line:
            words.append(bbox[1][0])
            boxes.append(create_bounding_box(bbox[0], width_scale, height_scale))
    return words, boxes


# function infer might change with new findings on model
def infer(image, ocr):
    width, height = image.size
    lang = 'eng'
    custom_config = r'--oem 3 --psm 6'


    if ocr == 'aws':
        words, boxes = aws_processor(image, width, height)
        logger.info('infer - aws_processor OK')
    elif ocr == 'paddle' or ocr =='':
        words, boxes = paddle_processor(image, width, height)
        logger.info('infer - paddle_processor OK')
    else:
        feature_extractor = LayoutLMv3ImageProcessor(apply_ocr = True)
        # feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr = True, lang = lang, config = custom_config)
        logger.info('infer - feature extractor OK')
        encoding_feature_extractor = feature_extractor(image, return_tensors="pt", truncation = True)
        logger.info('infer - encoding feature extractor OK')
        words, boxes = encoding_feature_extractor.words, encoding_feature_extractor.boxes

    # encode
    # encoding = processor(image, truncation = True, return_offsets_mapping = True, return_tensors = "pt",
    #                      padding="max_length", stride = 128, max_length = 512, return_overflowing_tokens = True)
    encoding = processor(image, words, boxes = boxes, truncation = True, return_offsets_mapping = True, return_tensors="pt",
                         padding = "max_length", stride = 128, max_length = 512, return_overflowing_tokens = True)
    logger.info('infer - encoding OK')
    offset_mapping = encoding.pop('offset_mapping')
    overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')

    # change the shape of pixel values
    x = []
    for i in range(0, len(encoding['pixel_values'])):
        x.append(encoding['pixel_values'][i])
    x = torch.stack(x)
    encoding['pixel_values'] = x
    logger.info('infer - change shape of pixel values OK')

    # forward pass
    outputs = model(**encoding)
    logger.info('infer - forward pass OK')

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    logger.info('infer - predictions OK')
    token_boxes = encoding.bbox.squeeze().tolist()
    logger.info('infer - token boxes OK')

    # only keep non-subword predictions
    preds = []
    l_words = []
    bboxes = []
    token_section_num = []

    if (len(token_boxes) == 512):
        predictions = [predictions]
        token_boxes = [token_boxes]


    for i in range(0, len(token_boxes)):
        for j in range(0, len(token_boxes[i])):
            unnormal_box = unnormalize_box(token_boxes[i][j], width, height)
            if (np.asarray(token_boxes[i][j]).shape != (4,)):
                continue
            elif (token_boxes[i][j] == [0, 0, 0, 0] or token_boxes[i][j] == 0):
                #print('zero found!')
                continue
            # if bbox is available in the list, just we need to update text
            elif (unnormal_box not in bboxes):
                preds.append(predictions[i][j])
                l_words.append(processor.tokenizer.decode(encoding["input_ids"][i][j]))
                bboxes.append(unnormal_box)
                token_section_num.append(i)
            else:
                # we have to update the word
                _index = bboxes.index(unnormal_box)
                if (token_section_num[_index] == i):
                    # check if they're in a same section or not (documents with more than 512 tokens will divide to seperate
                    # parts, so it's possible to have a word in both of the pages and we have to control that repetetive words
                    # HERE: because they're in a same section, so we can merge them safely
                    l_words[_index] = l_words[_index] + processor.tokenizer.decode(encoding["input_ids"][i][j])
                else:
                    continue

    logger.info('infer end')
    return bboxes, preds, l_words, image


def cleanup(pattern):
    for f in os.listdir(UPLOAD_FOLDER):
        if re.search(pattern, os.path.join(UPLOAD_FOLDER, f)):
            os.remove(os.path.join(UPLOAD_FOLDER, f))


def checkRotation(filePath):
    im = skio.imread(filePath)
    try:
        newdata = pytesseract.image_to_osd(im, nice=1)
        rotation = re.search('(?<=Rotate: )\d+', newdata).group(0)
    except:
        # Exception might happen with blank pages (tesseract not detecting anything)
        # so to mark it we set rotation = -1
        rotation = -1
    return rotation


# correct the skewness of images
def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    # Convert the PIL Image object to a numpy array
    image = np.asarray(image.convert('L'), dtype=np.uint8)

    # Apply thresholding
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)
    return best_angle, corrected


def remove_borders(img):
    result = img.copy()

    try:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) # convert to grayscale
    except:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        gray = result[:, :, 0]
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)
    return result


def unify_response(response):
    unified_response = dict()
    detection_index = 0
    data_to_review = []
    count_important = 0
    for pag_nr in response:
        pag_data = response[pag_nr]
        if len(pag_data["data_to_review"]) != 0:
            count_important += 1
            detection_index += float(pag_data["detection_index"])
            data_to_review.append(pag_data["data_to_review"])
    unified_response["detection_index"] = detection_index/count_important
    unified_response["data_to_review"] = data_to_review
    return unified_response