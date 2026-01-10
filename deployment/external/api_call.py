"""
Module focuses on calling the relevant API.

"""

import requests
import json
import pandas as pd

def call_predict_gas_api(link):

    result = requests.get(link)
    status_code = result.status_code
    return result.json(), result.status_code



def get_request_with_authentication():

    api_key = "1ocgRB9yyf0QBhis3R3T3f8Rp2J0raO2OLxgSnI5"  #API Key Used for authentication.
    headers = {'x-api-key': api_key,
               'start_date': "2021-01-01",
               'end_date': "2027-12-31",
               'state': "Virginia",
               'component_type': "residential"
               }

    request_string = "https://7hebop3w6c.execute-api.us-east-2.amazonaws.com/default/predictgas"
    return request_string, headers



def test_api_gateway(request_string, headers, result=None):

    if result is None:
        try:
            result = requests.get(request_string, headers=headers)
        except Exception as error:
            raise RuntimeError(error)

    status = "Success"
    if not result.ok:
        message_dict = json.loads(result.text)
        message = message_dict['message']
        if message == 'Missing Authentication Token':
            status = """Failure in the API Gateway. There is a missing Authentication Token. 
                               To get a Authentication token, please contact nr282@cornell.edu, if you are
                               entitled to the token. Need to make sure that the request has the relevant
                               headers"""
        if message == "Internal server error":
            status = """Internal Service Error was found. The issue is with the server not the client.
                               The client has the correct authentication key, but server is not working properly.
                               Please contact nr282@cornell.edu for maintenance."""
        if message == "Forbidden":
            status = """API Key exists, but is incorrect."""
    else:
        status = "Success"

    try:
        df = pd.DataFrame.from_dict(json.loads(result.text))
    except:
        df = None

    return df, status



