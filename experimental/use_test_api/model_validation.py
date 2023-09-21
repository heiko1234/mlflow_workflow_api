

import pandas as pd


from experimental.api_call_clients import APIBackendClient


dataclient=APIBackendClient()




headers = None


endpoint = "model_validation"
# endpoint = "list_available_blobs"


blobstorage_environment = "devstoreaccount1"




# class make_prediction(BaseModel):
#     blobcontainer: str | None = Field(example="chemical-data")
#     subcontainer: str | None = Field(example="chemical-data")
#     file_name: str | None = Field(example="ChemicalManufacturingProcess.parquet")
#     account: str | None = Field(example="devstoreaccount1")
#     use_model_name: str | None = Field(example="my_model_name")



data_statistics_dict = {
    "blobcontainer": "chemical-data",
    "subcontainer": "chemical-data",
    "file_name": "ChemicalManufacturingProcess.parquet",
    "account": blobstorage_environment,
    "use_model_name": "project_name",
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

output_df