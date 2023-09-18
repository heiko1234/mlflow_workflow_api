








import pandas as pd


from experimental.api_call_clients import APIBackendClient


dataclient=APIBackendClient()







headers = None
endpoint = "data_statistics"


blobstorage_environment = "devstoreaccount1"


data_statistics_dict = {
    "blobcontainer": "chemical-data",
    "subcontainer": "chemical-data",
    "file_name": "ChemicalManufacturingProcess.parquet",
    "account": blobstorage_environment
}



response = dataclient.Backendclient.execute_post(
    headers=headers,
    endpoint=endpoint,
    json=data_statistics_dict
    )

response.status_code     # 200

if response.status_code == 200:
    output = response.json()

output


output_df = pd.read_json(output, orient='split')
print(f"update_descriptive data head: {output_df.head()}")

digits = 2
output_df = output_df.round(digits)














