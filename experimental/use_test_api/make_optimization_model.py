




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
endpoint = "model_make_optimizing"


blobstorage_environment = "devstoreaccount1"



# class make_optimization_with_data(BaseModel):
#     account: str | None = Field(example="devstoreaccount1")
#     use_model_name: str | None = Field(example="my_model_name")
#     limits_dict: Dict[str, Dict[str, Union[str, int, float]]] | None = Field(example={"BioMaterial1": {"min": 10, "max": 20}})
#     staging: str | None = Field(example="None, Staging or Production")







data_statistics_dict = {
    "account": blobstorage_environment,
    "use_model_name": "project_name",
    "limits_dict": {
        "BiologicalMaterial02": {
            "min": 50,
            "max": 66
        },
        "ManufacturingProcess06": {
            "min": 201,
            "max": 230
        }
    },
    "staging": "Staging",
    "target": 44
}


data_statistics_dict




response = dataclient.Backendclient.execute_post(
    headers=headers,
    endpoint=endpoint,
    json=data_statistics_dict
    )


if response.status_code == 200:
    output = response.json()



output








