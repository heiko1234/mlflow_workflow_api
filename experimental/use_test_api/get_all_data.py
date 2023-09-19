






import pandas as pd


from experimental.api_call_clients import APIBackendClient


dataclient=APIBackendClient()







headers = None
endpoint = "all_data"


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

output_df



output_df.loc[:, ["Yield", "BiologicalMaterial01", "BiologicalMaterial02", "ManufacturingProcess42"]]



# >>> output_df.loc[:, ["Yield", "BiologicalMaterial01", "BiologicalMaterial02", "ManufacturingProcess42"]]
#     Yield  BiologicalMaterial01  BiologicalMaterial02  ManufacturingProcess42
# 0   43.12                  7.48                 64.47                    11.7
# 1   43.06                  6.94                 63.60                    11.4
# 2   41.49                  6.94                 63.60                    11.4
# 3   42.45                  6.94                 63.60                    11.3
# 4   42.04                  7.17                 61.23                    11.0



# ####################






headers = None
endpoint = "data_load_and_clean"


blobstorage_environment = "devstoreaccount1"



# blobcontainer: str | None = Field(example="chemical-data")
# subcontainer: str | None = Field(example="chemical-data")
# file_name: str | None = Field(example="ChemicalManufacturingProcess.parquet")
# account: str | None = Field(example="devstoreaccount1")
# features: List[str] | None = Field(example=["BioMaterial1", "BioMaterial2", "ProcessValue1"])
# spc_cleaning_dict: Dict[str, Union[str, int, float]] | None = Field(example={"BioMaterial1": "no cleaning", "BioMaterial2": "remove data", "ProcessValue1": 0.5})
# limits_dict: Dict[str, Dict[str, Union[str, int, float]]] | None = Field(example={"BioMaterial1": {"min": 10, "max": 20}})
# transformation_dict: Dict[str, str] | None = Field(example={"Yield": "no transformation", "BioMaterial1": "log", "BioMaterial2": "sqrt", "ProcessValue1": "1/x"})



data_statistics_dict = {
    "blobcontainer": "chemical-data",
    "subcontainer": "chemical-data",
    "file_name": "ChemicalManufacturingProcess.parquet",
    "account": blobstorage_environment,
    "features": ["Yield", "BiologicalMaterial01", "BiologicalMaterial02", "ManufacturingProcess42"],
    "spc_cleaning_dict": {"BioMaterial1": {"rule1": "no cleaning"}, "BiologicalMaterial02": {"rule1": "remove data"}, "ManufacturingProcess42": {"rule5": "remove data"}},
    "limits_dict": {"Yield": {"min": 36, "max": 44}},
    "transformation_dict": {"Yield": "no transformation", "BiologicalMaterial01": "log", "BiologicalMaterial02": "sqrt", "ManufacturingProcess42": "1/x"}
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


# >>> output_df
#     Yield  BiologicalMaterial01  BiologicalMaterial02  ManufacturingProcess42
# 0   43.12              2.012233              8.029321                0.085470
# 1   43.06              1.937302              7.974961                0.087719
# 2   41.49              1.937302              7.974961                0.087719
# 3   42.45              1.937302              7.974961                0.088496
# 4   42.04              1.969906              7.824960                0.090909


# ###########################


# original data

# >>> output_df.loc[:, ["Yield", "BiologicalMaterial01", "BiologicalMaterial02", "ManufacturingProcess42"]]
#     Yield  BiologicalMaterial01  BiologicalMaterial02  ManufacturingProcess42
# 0   43.12                  7.48                 64.47                    11.7
# 1   43.06                  6.94                 63.60                    11.4
# 2   41.49                  6.94                 63.60                    11.4
# 3   42.45                  6.94                 63.60                    11.3
# 4   42.04                  7.17                 61.23                    11.0



# "Yield": "no transformation"
# "BiologicalMaterial01": "log"
# "BiologicalMaterial02": "sqrt"
# "ManufacturingProcess42": "1/x"

# >>> output_df
#     Yield  BiologicalMaterial01  BiologicalMaterial02  ManufacturingProcess42
# 0   43.12              2.012233              8.029321                0.085470
# 1   43.06              1.937302              7.974961                0.087719
# 2   41.49              1.937302              7.974961                0.087719
# 3   42.45              1.937302              7.974961                0.088496
# 4   42.04              1.969906              7.824960                0.090909




