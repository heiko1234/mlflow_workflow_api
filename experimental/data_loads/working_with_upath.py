




import os
from dotenv import load_dotenv

load_dotenv()

from upath import UPath




path = UPath("az://chemical-data", account_name="devstoreaccount1", anon=False)

[p for p in path.iterdir()]
[str(p).split("/")[-1] for p in path.iterdir() if ".parquet" in str(p)]




# 


from azure.identity import DefaultAzureCredential


credential = DefaultAzureCredential(exclude_environment_credential=True)


credential = os.getenv("AZURE_STORAGE_KEY")


path = UPath("az://", account_name="devstoreaccount1", anon=True, credential=credential)


[str(p).split("/")[-1] for p in path.iterdir()]





path = UPath("az://chemical-data", account_name="devstoreaccount1", anon=True, credential=credential)


[str(p).split("/")[-1] for p in path.iterdir()]









