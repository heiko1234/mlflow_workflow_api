




import pandas as pd


from experimental.api_call_clients import APIBackendClient


dataclient=APIBackendClient()








headers = None
endpoint = "list_available_models"

response = dataclient.Backendclient.execute_get(
    headers=headers,
    endpoint=endpoint,
    )

response.status_code     # 200

if response.status_code == 200:
    output = response.json()

output

["prouject_name"]

# ########################



output = None
headers = None
endpoint = "get_model_artifact"

data_statistics_dict = {
    "account": "devstoreaccount1",
    "use_model_name": "project_name",
    "artifact": "feature_limits_unscaled.json"
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
# >>> output
# {'BiologicalMaterial02': {'min': 51.28, 'max': 64.75}, 
# 'BiologicalMaterial06': {'min': 44.3, 'max': 59.38}, 
# 'ManufacturingProcess13': {'min': 32.1, 'max': 38.6}}





# #####################



headers = None
endpoint = "model_prediction_send_data"


blobstorage_environment = "devstoreaccount1"



# use_model_name: str | None = Field(example="my_model_name")
# data_dict: Dict[str, Union[str, int, float]] | None = Field(example={"BioMaterial1": 10, "BioMaterial2": 20, "ProcessValue1": 30})



#                   feature   value
# 0    BiologicalMaterial02  58.015
# 1    BiologicalMaterial06  51.840
# 2  ManufacturingProcess13  35.350



df = pd.DataFrame(
    {
        "feature": ["BiologicalMaterial02", "BiologicalMaterial06", "ManufacturingProcess13"],
        "value": [58.015, 51.840, 35.350],
    }
)

df

df.T

#                             0                     1                       2
# feature  BiologicalMaterial02  BiologicalMaterial06  ManufacturingProcess13
# value                  58.015                 51.84                   35.35


df
# change columns to rows and rows to columns
df.T
# change column names to the name of first row
df.T.rename(columns=df.T.iloc[0])
# remove first row
df.T.rename(columns=df.T.iloc[0]).iloc[1:]


# DAs hier

df_for_modelling = df.T.rename(columns=df.T.iloc[0]).iloc[1:].reset_index(drop=True)
df_for_modelling

# >>> df_for_modelling
#   BiologicalMaterial02 BiologicalMaterial06 ManufacturingProcess13
# 0               58.015                51.84                  35.35


data_dict = df_for_modelling.to_dict(orient="records")

# df
# data_dict
# [{'feature': 'BiologicalMaterial02', 'value': 58.015}, {'feature': 'BiologicalMaterial06', 'value': 51.84}, {'feature': 'ManufacturingProcess13', 'value': 35.35}]

# df_for_modelling
# >>> data_dict
# [{'BiologicalMaterial02': 58.015, 'BiologicalMaterial06': 51.84, 'ManufacturingProcess13': 35.35}]




data_dict
# >>> data_dict
# [{'BiologicalMaterial02': 58.015, 'BiologicalMaterial06': 51.84, 'ManufacturingProcess13': 35.35}]



df_reimport = None
df_reimport = pd.DataFrame.from_dict(data_dict, orient="columns").reset_index()
df_reimport




data_statistics_dict = {
    "account": blobstorage_environment,
    "use_model_name": "project_name",
    "data_dict": data_dict[0]
}

data_statistics_dict



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
output_df.iloc[0,0]



# ##################



output = None
headers = None
endpoint = "get_model_artifact"

data_statistics_dict = {
    "account": "devstoreaccount1",
    "use_model_name": "project_name",
    "artifact": "target_limits.json"
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

# ... 
# >>> output
# {'Yield': {'min': 33.2392790595385, 'max': 48.35072094046151}}




# #########################

df = pd.read_parquet("./data/ChemicalManufacturingProcess.parquet")

data_dict = df.to_dict(orient="records")
data_dict



headers = None
endpoint = "model_prediction_send_data"


blobstorage_environment = "devstoreaccount1"






data_statistics_dict = {
    "account": blobstorage_environment,
    "use_model_name": "project_name",
    "data_dict": data_dict[0]
}

data_statistics_dict





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
output_df.iloc[0,0]




