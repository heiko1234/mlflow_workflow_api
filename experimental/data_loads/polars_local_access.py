

import os
from dotenv import load_dotenv

import polars as pl

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient, BlobBlock



from dotenv import load_dotenv

load_dotenv()


account = "devstoreaccount1"

# credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True, exclude_environment_credential=True)

credential = os.getenv("AZURE_STORAGE_KEY")
credential


# http://127.0.0.1:10000/devstoreaccount1/chemical-data


# master_data = pl.read_parquet(
#     f"http://127.0.0.1:10000/{account}/chemical-data/chemical-data/ChemicalManufacturingProcess.parquet",
#     storage_options={"account_name": account, "credential": credential}
#     )



master_data = pl.read_parquet(
    f"az://chemical-data/chemical-data/ChemicalManufacturingProcess.parquet",
    storage_options={"account_name": account, "credential": credential}
    )


master_data


# >>> master_data
# shape: (86, 58)
# ┌───────┬──────────────────────┬──────────────────────┬──────────────────────┬───┬────────────────────────┬────────────────────────┬────────────────────────┬────────────────────────┐
# │ Yield ┆ BiologicalMaterial01 ┆ BiologicalMaterial02 ┆ BiologicalMaterial03 ┆ … ┆ ManufacturingProcess42 ┆ ManufacturingProcess43 ┆ ManufacturingProcess44 ┆ ManufacturingProcess45 │
# │ ---   ┆ ---                  ┆ ---                  ┆ ---                  ┆   ┆ ---                    ┆ ---                    ┆ ---                    ┆ ---                    │
# │ f64   ┆ f64                  ┆ f64                  ┆ f64                  ┆   ┆ f64                    ┆ f64                    ┆ f64                    ┆ f64                    │
# ╞═══════╪══════════════════════╪══════════════════════╪══════════════════════╪═══╪════════════════════════╪════════════════════════╪════════════════════════╪════════════════════════╡
# │ 43.12 ┆ 7.48                 ┆ 64.47                ┆ 72.41                ┆ … ┆ 11.7                   ┆ 0.7                    ┆ 2.0                    ┆ 2.2                    │
# │ 43.06 ┆ 6.94                 ┆ 63.6                 ┆ 72.06                ┆ … ┆ 11.4                   ┆ 0.8                    ┆ 2.0                    ┆ 2.2                    │
# │ 41.49 ┆ 6.94                 ┆ 63.6                 ┆ 72.06                ┆ … ┆ 11.4                   ┆ 0.9                    ┆ 1.9                    ┆ 2.1                    │
# │ 42.45 ┆ 6.94                 ┆ 63.6                 ┆ 72.06                ┆ … ┆ 11.3                   ┆ 0.8                    ┆ 1.9                    ┆ 2.4                    │
# │ …     ┆ …                    ┆ …                    ┆ …                    ┆ … ┆ …                      ┆ …                      ┆ …                      ┆ …                      │
# │ 37.86 ┆ 6.01                 ┆ 51.83                ┆ 63.8                 ┆ … ┆ 11.6                   ┆ 1.1                    ┆ 1.8                    ┆ 2.3                    │
# │ 38.05 ┆ 5.89                 ┆ 51.28                ┆ 64.04                ┆ … ┆ 11.4                   ┆ 1.0                    ┆ 1.8                    ┆ 2.4                    │
# │ 37.87 ┆ 5.9                  ┆ 51.44                ┆ 63.61                ┆ … ┆ 11.8                   ┆ 0.6                    ┆ 1.9                    ┆ 2.1                    │
# │ 38.6  ┆ 5.9                  ┆ 51.44                ┆ 63.61                ┆ … ┆ 11.7                   ┆ 0.8                    ┆ 1.9                    ┆ 2.0                    │
# └───────┴──────────────────────┴──────────────────────┴──────────────────────┴───┴────────────────────────┴────────────────────────┴────────────────────────┴────────────────────────┘






