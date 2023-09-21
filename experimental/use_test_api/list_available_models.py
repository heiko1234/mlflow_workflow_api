





import pandas as pd


from experimental.api_call_clients import APIBackendClient


dataclient=APIBackendClient()



headers = None
endpoint = "list_available_models"


blobstorage_environment = "devstoreaccount1"




response = dataclient.Backendclient.execute_get(
    headers=headers,
    endpoint=endpoint,
    )


response.status_code     # 200


if response.status_code == 200:
    output = response.json()

output

isinstance(output, list)

output_df = pd.read_json(output, orient='split')

output_df




