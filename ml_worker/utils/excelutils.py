import os
import re
from datetime import datetime as dt
from dateutil.parser import parse
from thefuzz import fuzz

UPLOAD_FOLDER = '/flask_app/files/xlsx/'
import pandas as pd


def format_doc(doc_type, doc_name, extracted_data, pathfile):
    if doc_type == 'vgm':
        return format_vgm(doc_name, extracted_data)
    elif doc_type == 'loc':
        return loc(doc_name, extracted_data)
    else:
        return


def prune_text(text):
    chars = "\\`*_\{\}[]\(\)\|/<>#-\'\"+!$,\."
    for c in chars:
        if c in text:
            text = text.replace(c, "")
    return text


def cleanup_text(text):
    result = re.sub(r'[^a-zA-Z0-9]+', '', text)
    print('result',result)
    return result


def remove_leading_trailing_special_characters(text):
    result = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', text)
    return result




def format_vgm(doc_name, extracted_data):
    """
    EXAMPLE extracted_data
    {
       "0":{
          "detection_index":"0.80",
          "data_to_review":[
             {
                "key":"Header",
                "type":"Inputs",
                "value":[
                   {
                      "key":"booking number",
                      "value":"CNAN12961",
                      "state":"INCOMPLETE"
                   },
                   {
                      "key":"container id",
                      "value":"EMAU3021997",
                      "state":"INCOMPLETE"
                   },
                   {
                      "key":"signer name",
                      "value":"DANIELA  FASANELLA",
                      "state":"INCOMPLETE"
                   },
                   {
                      "key":"shipper",
                      "value":"COMPANY SPA",
                      "state":"INCOMPLETE"
                   },
                   {
                      "key":"vgm",
                      "value":"Kg  27260",
                      "state":"INCOMPLETE"
                   }
                ],
                "page":1
             }
          ]
       },
       "1":{
          "detection_index":"0.80",
          "data_to_review":[
             {
                "key":"Header",
                "type":"Inputs",
                "value":[
                   {
                      "key":"container id",
                      "value":"EMAU3021997",
                      "state":"INCOMPLETE"
                   },
                   {
                      "key":"vgm",
                      "value":"27.260",
                      "state":"INCOMPLETE"
                   }
                ],
                "page":2
             }
          ]
       }
    }
    """

    doc_name_contents = re.split("_", doc_name, 2)
    if len(doc_name_contents) == 3:
        attach_filename = doc_name_contents[2]
    else:
        attach_filename = doc_name
    reference_number = attach_filename.replace(".PDF", "").replace(".pdf", "")

    containerId, bookingNumber, authorizedPerson, vgm, signerName, shipper, containerType = [], [], [], [], [], [], []
    for page_nr in extracted_data:
        print(page_nr)
        data_to_review = extracted_data[page_nr]['data_to_review']
        for element in data_to_review:
            if element['key'] == 'Header':
                page = element['page']
                print('page: {}'.format(page))
                for element_item in element['value']:
                    # print(element_item['key'])
                    if element_item['key'] == 'container id' and element_item['value'] != "":
                        value = cleanup_text(element_item['value'])
                        containerId.append(value)
                    if element_item['key'] == 'booking number' and element_item['value'] != "":
                        value = cleanup_text(element_item['value'])
                        bookingNumber.append(value)
                    if element_item['key'] == 'authorized person' and element_item['value'] != "":
                        authorizedPerson.append(element_item['value'])
                    if element_item['key'] == 'vgm' and element_item['value'] != "":
                        vgm.append(element_item['value'])
                    if element_item['key'] == 'signer name' and element_item['value'] != "":
                        signerName.append(element_item['value'])
                    if element_item['key'] == 'shipper' and element_item['value'] != "":
                        shipper.append(element_item['value'])
                    if element_item['key'] == 'container type' and element_item['value'] != "":
                        containerType.append(element_item['value'])

    xls_filepath = os.path.join(UPLOAD_FOLDER, reference_number + ".xlsx")
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # CURRENT EXCEL HEADER
    # num. Booking | Tipo Cnt (Codice Iso) | Num. Container | Peso VGM | Nome persona autorizzata al Vgm | Caricatore/Shipper | Metodo 1 (Conservo scontrino) | Metodo 1 (allego scontrino) | Metodo 2(certificazione AEO o ISO9001/28000)
    df = pd.DataFrame({
                       'num. Booking': pd.Series(bookingNumber),
                       'Tipo Cnt': pd.Series(containerType),
                       'Num. Container': pd.Series(containerId),
                       'Peso VGM': pd.Series(vgm),
                       'Nome persona autorizzata al Vgm': pd.Series(authorizedPerson),
                       'Nome persona che ha firmato il Vgm': pd.Series(signerName),
                       'Shipper': pd.Series(shipper)
                       })

    writer = pd.ExcelWriter(xls_filepath, engine='xlsxwriter')
    df.to_excel(writer, sheet_name="Sheet1", index=False)  # send df to writer
    worksheet = writer.sheets["Sheet1"]  # pull worksheet object
    for idx, col in enumerate(df):  # loop through all columns
        series = df[col]
        max_len = max((
            series.astype(str).map(len).max(),  # len of largest item
            len(str(series.name))  # len of column name/header
            )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width
    writer.close()

    # df.to_excel(xls_filepath, index=False)

    print('excel file has been created!')
    print(df)

    return xls_filepath, reference_number + ".xlsx"


def loc(doc_name, extracted_data):
    """
    EXAMPLE extracted_data
    {
  "data_to_review": [
    [
      {
        "key": "Header",
        "page": 1,
        "type": "Inputs",
        "value": [
          {
            "key": "lc_number",
            "state": "INCOMPLETE",
            "value": "0541ICD0000322099"
          },
          {
            "key": "date_of_issue",
            "state": "INCOMPLETE",
            "value": "220510"
          },
          {
            "key": "applicant",
            "state": "INCOMPLETE",
            "value": "BENBELLAT AHMED CITE LOMBARKIA 47 ROUTE DE TAZOULT 05000 BATNA ALGERIE"
          },
          {
            "key": "beneficiary",
            "state": "INCOMPLETE",
            "value": "BENETTI MACCHINE VIA PROVINCIALE NAZZANO 20 54033 CARRARA ITALY TEL 390585844347 FAX 390585842667"
          },
          {
            "key": "port_of_loading",
            "state": "INCOMPLETE",
            "value": "PORT ITALIEN"
          },
          {
            "key": "port_of_discharge",
            "state": "INCOMPLETE",
            "value": "PORT DE SKIKDA"
          },
          {
            "key": "description",
            "state": "INCOMPLETE",
            "value": "CFR PORT DE SKIKDA INCOTERMS 2020 01 HAVEUSE A CHAINE ET UN LOT DE PIECES DE RECHANGE MONTANT DE MARCHANDISE : EUR 130.000,00 MONTANT DU FRET : EUR 2.000,00 TOTAL : EUR 132.000,00 SUIVANT FACTURE PROFORMA NR 373 2021 Rev8 DU 04 05 2022"
          }
        ]
      }
    ]
  ],
  "detection_index": 0.8
}
"""

    doc_name_contents = re.split("_", doc_name, 2)
    if len(doc_name_contents) == 3:
        attach_filename = doc_name_contents[2]
    else:
        attach_filename = doc_name
    reference_number = attach_filename.replace(".PDF", "").replace(".pdf", "")

    lcNumber, dateOfIssue, Applicant, Beneficiary, portOfLoading = [], [], [], [], []
    portOfDischarge, latestDateOfShipment, Description = [], [], []
    for page_nr in extracted_data:
        print(page_nr)
        data_to_review = extracted_data[page_nr]['data_to_review']
        for element in data_to_review:
            if element['key'] == 'Header':
                page = element['page']
                print('page: {}'.format(page))
                for element_item in element['value']:
                    # print(element_item['key'])
                    if element_item['key'] == 'lc_number' and element_item['value'] != "":
                        # print('lc_number Before',lcNumber)
                        value = remove_leading_trailing_special_characters(element_item['value'])
                        lcNumber.append(value)
                    if element_item['key'] == 'date_of_issue' and element_item['value'] != "":
                        value = remove_leading_trailing_special_characters(element_item['value'])
                        dateOfIssue.append(value)
                    if element_item['key'] == 'applicant' and element_item['value'] != "":
                        value = remove_leading_trailing_special_characters(element_item['value'])
                        Applicant.append(value)
                    if element_item['key'] == 'beneficiary' and element_item['value'] != "":
                        value = remove_leading_trailing_special_characters(element_item['value'])
                        Beneficiary.append(value)
                    if element_item['key'] == 'port_of_loading' and element_item['value'] != "":
                        value = remove_leading_trailing_special_characters(element_item['value'])
                        portOfLoading.append(value)
                    if element_item['key'] == 'port_of_discharge' and element_item['value'] != "":
                        value = remove_leading_trailing_special_characters(element_item['value'])
                        portOfDischarge.append(value)
                    if element_item['key'] == 'latest_date_of_shipment' and element_item['value'] != "":
                        value = remove_leading_trailing_special_characters(element_item['value'])
                        latestDateOfShipment.append(value)
                    if element_item['key'] == 'description' and element_item['value'] != "":
                        value = remove_leading_trailing_special_characters(element_item['value'])
                        Description.append(value)

    xls_filepath = os.path.join(UPLOAD_FOLDER, reference_number + ".xlsx")
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # CURRENT EXCEL HEADER
    # MERCE | TIPOLOGIA CONTAINER | SIGLA | COLLI | PESO LORDO | PESO NETTO | VOLUME | SIGILLI | TIPO | IMBALLO | TARA
    df = pd.DataFrame({
                       'lc_number': pd.Series(lcNumber),
                       'date_of_issue': pd.Series(dateOfIssue),
                       'applicant': pd.Series(Applicant),
                       'beneficiary': pd.Series(Beneficiary),
                       'port_of_loading': pd.Series(portOfLoading),
                       'port_of_discharge': pd.Series(portOfDischarge),
                       'latest_date_of_shipment': pd.Series(latestDateOfShipment),
                       'description': pd.Series(Description)
                       })

    writer = pd.ExcelWriter(xls_filepath, engine='xlsxwriter')
    df.to_excel(writer, sheet_name="Sheet1", index=False)  # send df to writer
    worksheet = writer.sheets["Sheet1"]  # pull worksheet object
    for idx, col in enumerate(df):  # loop through all columns
        series = df[col]
        max_len = max((
            series.astype(str).map(len).max(),  # len of largest item
            len(str(series.name))  # len of column name/header
            )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width
    writer.close()

    # df.to_excel(xls_filepath, index=False)

    print('excel file has been created!')
    print(df)

    return xls_filepath, reference_number + ".xlsx"