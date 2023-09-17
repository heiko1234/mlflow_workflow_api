


import pandas as pd


from experimental.api_call_clients import APIBackendClient







data_statistics_dict = {
    "blobcontainer": "chemical-data",
    "subcontainer": "chemical-data",
    "file_name": "ChemicalManufacturingProcess.parquet"
    }


endpoint = "data_statistics"

headers = None
headers = {"accept": "application/json", "Content-Type": "application/json"}


dataclient=APIBackendClient()


response = dataclient.Backendclient.execute_post(
    headers=headers,
    endpoint=endpoint,
    json=data_statistics_dict
    )


response.status_code     # 200

df = response.json()

df


# import json dataframe into pandas dataframe
pddf = pd.read_json(df, orient="split")
pddf



# #############################


endpoint = "testdata"


response = dataclient.Backendclient.execute_get(
    headers=headers,
    endpoint=endpoint,
    )


response.status_code   # 200

df = response.json()

df


