


# (.venv) cd backend_service/backend_service/
# (.venv) uvicorn main:app --reload






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


# #######################


endpoint = "list_available_blobs"


response = dataclient.Backendclient.execute_get(
    headers=headers,
    endpoint=endpoint,
    )


response.status_code   # 200

df = response.json()

df









