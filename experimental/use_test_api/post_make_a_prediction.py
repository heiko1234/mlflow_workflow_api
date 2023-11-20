




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

["project_name"]






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
        "feature": ["BiologicalMaterial02", "ManufacturingProcess12", "ManufacturingProcess42"],
        "value": [58.015, 4000 , 11],
    }
)

df

df.T

#                             0                     1                       2
# feature  BiologicalMaterial02  BiologicalMaterial06  ManufacturingProcess13
# value                  58.015                 51.84                   35.35




df = pd.DataFrame(
    {
        "feature": ["BiologicalMaterial02", "BiologicalMaterial06", "ManufacturingProcess13"],
        "value": [58.015, 51.840, 35.350],
    }
)




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





# read complete dataset and makek a prediction

df = pd.read_parquet("/home/heiko/Schreibtisch/Repos/data/ChemicalManufacturingProcess.parquet")

df_for_modelling = df




data_dict = df_for_modelling.to_dict(orient="records")


data_dict

# ...
# 'BiologicalMaterial02': 51.44, 'BiologicalMaterial03': 63.61, 'BiologicalMaterial04': 10.49, 'BiologicalMaterial05': 18.04, 'BiologicalMaterial06': 44.73, 'BiologicalMaterial07': 100.0, 'BiologicalMaterial08': 17.18, 'BiologicalMaterial09': 12.95, 'BiologicalMaterial10': 2.46, 'BiologicalMaterial11': 143.84, 'BiologicalMaterial12': 19.85, 'ManufacturingProcess01': 11.8, 'ManufacturingProcess02': 21.8, 'ManufacturingProcess03': 1.55, 'ManufacturingProcess04': 936.0, 'ManufacturingProcess05': 987.9, 'ManufacturingProcess06': 205.5, 'ManufacturingProcess07': 177.0, 'ManufacturingProcess08': 177.0, 'ManufacturingProcess09': 44.7, 'ManufacturingProcess10': 9.3, 'ManufacturingProcess11': 9.1, 'ManufacturingProcess12': 0.0, 'ManufacturingProcess13': 35.2, 'ManufacturingProcess14': 4816.0, 'ManufacturingProcess15': 5983, 'ManufacturingProcess16': 4579, 'ManufacturingProcess17': 35.2, 'ManufacturingProcess18': 4834, 'ManufacturingProcess19': 5997, 'ManufacturingProcess20': 4583, 'ManufacturingProcess21': 0.0, 'ManufacturingProcess22': 8.0, 'ManufacturingProcess23': 2.0, 'ManufacturingProcess24': 2.0, 'ManufacturingProcess25': 4841.0, 'ManufacturingProcess26': 6010.0, 'ManufacturingProcess27': 4583.0, 'ManufacturingProcess28': 10.2, 'ManufacturingProcess29': 19.5, 'ManufacturingProcess30': 9.1, 'ManufacturingProcess31': 71.4, 'ManufacturingProcess32': 155, 'ManufacturingProcess33': 62.0, 'ManufacturingProcess34': 2.5, 'ManufacturingProcess35': 481.0, 'ManufacturingProcess36': 0.019, 'ManufacturingProcess37': 1.0, 'ManufacturingProcess38': 2, 'ManufacturingProcess39': 6.9, 'ManufacturingProcess40': 0.0, 'ManufacturingProcess41': 0.0, 
# 'ManufacturingProcess42': 11.7, 'ManufacturingProcess43': 0.8, 'ManufacturingProcess44': 1.9, 'ManufacturingProcess45': 2.0}]




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



isinstance(data_dict, dict)
isinstance(data_dict, list)

data_dict[0]
data_dict

data_dict[0:2]



# TODO: not working on a whole dataset, but with a dict of it


# entweder
data_statistics_dict = {
    "account": blobstorage_environment,
    "use_model_name": "project_name",
    "data_dict": data_dict[0:10]
}

# oder
data_statistics_dict = {
    "account": blobstorage_environment,
    "use_model_name": "project_name",
    "data_dict": data_dict[1]
}

data_statistics_dict



df = pd.DataFrame.from_dict([data_statistics_dict], orient="columns").reset_index()
df



headers = None
endpoint = "model_prediction_send_data"
blobstorage_environment = "devstoreaccount1"



response = dataclient.Backendclient.execute_post(
    headers=headers,
    endpoint=endpoint,
    json=data_statistics_dict
    )

# Report: simple line of data, works
# TODO: multiple data, not yet working
response.status_code     # 200



output = None
if response.status_code == 200:
    output = response.json()

output


output_df = pd.read_json(output, orient='split')
output_df.iloc[0,0]

output_df


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




