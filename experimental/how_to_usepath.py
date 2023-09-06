











from upath import UPath

from azure.identity import DefaultAzureCredential


credential = DefaultAzureCredential(exclude_environment_credential=True)


path = UPath("az://chemical-data/chemical-data/", account_name="devstoreaccount1", anon=True, credential=credential)


[p for p in path.iterdir()]







# #######################




import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc


# load data with dataset 

dataset = ds.dataset('data/ChemicalManufcturingProcess.parquet', format='parquet', partitioning='hive')



pc.field("Yield")

pc.match_substring(pc.field("Yield"), "7.8")

dataset.scanner()

generator = dataset.scanner().to_batches()

generator
# <generator object at 0x7f18fe833cc0>

dataset.scanner().to_batches()
dataset.scanner().to_table().to_pandas()

# >>> dataset.scanner().to_table().to_pandas()
#     Yield  BiologicalMaterial01  BiologicalMaterial02  ...  ManufacturingProcess43  ManufacturingProcess44  ManufacturingProcess45
# 0   43.12                  7.48                 64.47  ...                     0.7                     2.0                     2.2
# 1   43.06                  6.94                 63.60  ...                     0.8                     2.0                     2.2


# dataset.scanner(pc.match_substring(pc.field("Yield"), "7.8")).to_table().to_pandas()


dataset.filter(pc.match_substring(pc.field("Yield"), "7.8")).to_table().to_pandas()



df = dataset.to_table().to_pandas()
df










# ######################


import polars as pds


credential = DefaultAzureCredential(exclude_environment_credential=True)




df = pds.read_parquet('az://data/ChemicalManufcturingProcess.parquet')

df







