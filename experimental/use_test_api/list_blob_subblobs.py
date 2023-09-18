











import pandas as pd


from experimental.api_call_clients import APIBackendClient


dataclient=APIBackendClient()






headers = None
headers = {"accept": "application/json", "Content-Type": "application/json"}




endpoint = "list_available_accounts"


response = dataclient.Backendclient.execute_get(
    headers=headers,
    endpoint=endpoint,
    )


response.status_code   # 200

df = response.json()

df





headers = None
endpoint = "list_available_blobs"


blobstorage_environment = "devstoreaccount1"


data_statistics_dict = {
    "blobcontainer": None,
    "subcontainer": None,
    "file_name": None,
    "account": blobstorage_environment
}



response = dataclient.Backendclient.execute_post(
    headers=headers,
    endpoint=endpoint,
    json=data_statistics_dict
    )


if response.status_code == 200:
    output = response.json()

output
