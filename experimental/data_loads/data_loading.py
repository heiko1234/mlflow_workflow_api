





# from upath import Upath


import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as pds

from adlfs import AzureBlobFileSystem


from azure.identity import DefaultAzureCredential


import polars as pl

import os
import deltalake



actual_deploymant = "mlflowstorage"
actual_deploymant="devstoreaccount1"



storage_options = {
    "AZURE_STORAGE_ACCOUNT_NAME": actual_deploymant,
    "AZURE_CLIENT_ID": os.getenv("AZURE_CLIENT_ID"),
    "AZURE_CLIENT_SECRET": os.getenv("AZURE_CLIENT_SECRET"),
    "AZURE_TENANT_ID": os.getenv("AZURE_TENANT_ID"),
}



storage_options = {
    "AZURE_STORAGE_ACCOUNT_NAME": actual_deploymant,
    "AZURE_CLIENT_ID": os.getenv("AZURE_STORAGE_ACCOUNT"),
    "AZURE_CLIENT_SECRET": os.getenv("AZURE_ACCOUNT_KEY"),
    "AZURE_TENANT_ID": None
}




# delta_table = deltalake.DeltaTable(
#     Upath(f"abfs://{actual_deploymant}.dfs.core.windows.net/"),
#     filesystem="adlfs",
#     storage_options=storage_options
# ))


root_path = f"az://data/data/.../.../objectdata/"

root_path="az://chemical-data/chemical-data/ChemicalManufacturingProcess.parquet"



delta_table = deltalake.DeltaTable(
    root_path,
    storage_options=storage_options
)

delta_table.files()
ds = delta_table.to_pyarrow_dataset()

ds.to_table(filter=ds.field("date") == "2023-06-06").to_pandas()




# ####################


credentials = DefaultAzureCredential(exclude_environment_credential=True)

actual_deploymant = "devstoreaccount1"

root_path="az://chemical-data/chemical-data/ChemicalManufacturingProcess.parquet"

master_table = pl.read_parquet(
    root_path,
    storage_options={
        "account_name": actual_deploymant,
        "anon": False,
        "credential": credentials
    }
)
master_table

# ##############

root_path=f"az://data/data/source/.../.../objectdata/"

credentials = DefaultAzureCredential(exclude_environment_credential=True)

path = Upath(
    root_path,
    account_name=actual_deploymant,
    anon=False,
    credentials=credentials
    )

[p for p in path.iterdir()]



# ###########################

def storage_options_form_upath(path: Upath):
    
    storage_options_map={
        "account_name": "AZURE_STORAGE_ACCOUNT_NAME",
        "client_id": "AZURE_CLIENT_ID",
        "client_secret": "AZURE_CLIENT_SECRET",
        "tenant_id": "AZURE_TENANT_ID",
    }

    output = {
        storage_options_map.get(key, key): val 
        for key, val in path._kwargs.items()
    }
    return output


storage_options_form_upath(path)




# ###########################


account_str = "anyaccount"



account_str="devstoreaccount1"
credentials = DefaultAzureCredential(exclude_environment_credential=True)

abfs = AzureBlobFileSystem(
    account_name=account_str,
    credential=credentials,
    anon=False
)

file_system = fs.PyFileSystem(fs.FSSpecHandler(abfs))

container_name = "chemical-data"
subcontainer = "chemical-data"
file = "ChemicalManufacturingProcess.parquet"


file_path = f"{container_name}/{subcontainer}/{file}"


dataset_path = pds.dataset(file_path, filesystem=file_system, format="parquet", partitioning="hive")

get_data = dataset_path.to_table().to_pandas()

get_data.head()









