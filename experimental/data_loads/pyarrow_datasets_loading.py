



import os

from dotenv import load_dotenv

load_dotenv()



import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pds
import pyarrow.compute as pc
import pyarrow.fs as fs

from adlfs import AzureBlobFileSystem





from azure.identity.aio import AzureCliCredential as AioAzureCliCredential

credential = AioAzureCliCredential()



credential = os.getenv("AZURE_STORAGE_KEY")
credential

account = os.getenv("AZURE_STORAGE_ACCOUNT")
account



abfs = AzureBlobFileSystem(
    account_name=account,
    credential=credential,
    anon = False
    )


# ###########

container_name = "chemical-data"
subcontainer = "chemical-data"
file = "ChemicalManufacturingProcess.parquet"

file_system = fs.PyFileSystem(fs.FSSpecHandler(abfs))

file_path = f"{container_name}/{subcontainer}/{file}"



ds = pds.dataset(
    str(file_path),
    filesystem=file_system,
    format="parquet",
    partitioning="hive"
    )

ds.files
dstable = ds.to_table()

dstable_pandas = dstable.to_pandas()

dstable_pandas.head()


# >>> dstable_pandas.head()
#    Yield  BiologicalMaterial01  BiologicalMaterial02  BiologicalMaterial03  BiologicalMaterial04  ...  ManufacturingProcess41  ManufacturingProcess42  ManufacturingProcess43  ManufacturingProcess44  ManufacturingProcess45
# 0  43.12                  7.48                 64.47                 72.41                 13.82  ...                     0.0                    11.7                     0.7                     2.0                     2.2
# 1  43.06                  6.94                 63.60                 72.06                 15.70  ...                     0.0                    11.4                     0.8                     2.0                     2.2
# 2  41.49                  6.94                 63.60                 72.06                 15.70  ...                     0.0                    11.4                     0.9                     1.9                     2.1
# 3  42.45                  6.94                 63.60                 72.06                 15.70  ...                     0.0                    11.3                     0.8                     1.9                     2.4
# 4  42.04                  7.17                 61.23                 70.01                 13.36  ...                     0.0                    11.0                     1.0                     1.9                     1.8







