



# from azure.storage.blob.aio import BlobContainerClient
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobClient
from azure.storage.blob import ContainerClient


# import variables from .env file
import os
from dotenv import load_dotenv


import io
import pandas as pd



account_name = "local_azurite"
container_name="coinbasedata"




try:
    load_dotenv()
    url = f"https://{account_name}.blob.core.windows.net"

    # build connection_string from account_name and credential
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    # connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={credential};EndpointSuffix=core.windows.net"
    connection_string = ";".join(
                [
                    "DefaultEndpointsProtocol=http",
                    f"AccountName={account_name}",
                    f"AccountKey={account_key}",
                    f"DefaultEndpointsProtocol=http",
                    f"BlobEndpoint=http://127.0.0.1:10000/{account_name}",
                    f"QueueEndpoint=http://127.0.0.1:10001/{account_name}",
                ]
            )
except Exception as e:
    print(e)
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")



account_name
account_key
connection_string



# see General Dagster Pipeline (Repo)

blob_client = BlobServiceClient.from_connection_string(connection_string)




container_name = "coinbasedata"
blob_container_client = ContainerClient.from_connection_string(
    connection_string,
    container_name
    )


subcontainer = "coinbasedata"
files_with = ".parquet"

output = []
for blob in blob_container_client.list_blobs():
    print(blob.name)
    if subcontainer in blob.name and files_with in blob.name:
        output.append(blob.name.split("/")[1])
output


subcontainer = "coinbasedata"
file = "coinbase_data.parquet"


blob_str = subcontainer + "/" + file
bytes = (
    blob_container_client
    .get_blob_client(blob=blob_str)
    .download_blob()
    .readall()
)
pq_file = io.BytesIO(bytes)
df = pd.read_parquet(pq_file)

df



# get a file with blobclient
blob_container_client = BlobClient.from_connection_string(conn_str=connection_string, container_name=container_name, blob_name="coinbasedata")

subcontainer = "coinbasedata"
file = "coinbase_data.parquet"

blob_str = subcontainer + "/" + file
bytes=blob_container_client.get_blob_client(blob=blob_str).download_blob().readall()
pq_file = io.BytesIO(bytes)
df = pd.read_parquet(pq_file)
df



container = "coinbasedata"

blob_container_client=ContainerClient.from_connection_string(connection_string, container_name=container)

blobs=blob_container_client.list_blobs()
list_blobs = [blob for blob in blobs]
list_blobs


# to get subcontainers
list_blobs = [blob.name.split("/")[0] for blob in blobs]
list_blobs



# to list files in subcontainers
subcontainer = "coinbasedata"
files_with = ".parquet"

output = []
for blob in blob_container_client.list_blobs():
    if subcontainer in blob.name and files_with in blob.name:
        output.append(blob.name.split("/")[1])

output





# list all blobs in a container
blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_string)
container_client = blob_service_client.get_container_client(container_name)
blobs = container_client.list_blobs()
list_blobs = [blob for blob in blobs]
list_blobs


# download a blob


# blob_str = blob + "/" + file
# bytes = (
#     BlobStorageConnector(container_name=container_name)
#     .get_container_client()
#     .get_blob_client(blob=blob_str)
#     .download_blob()
#     .readall()
# )
# pq_file = io.BytesIO(bytes)
# df = pd.read_parquet(pq_file)



# bytes=blob_container_client.get_blob_client(blob="coinbasedata/coinbase_data.parquet").download_blob().readall()
# pq_file = io.BytesIO(bytes)
# df = pd.read_parquet(pq_file)
# df





# # list all blob containers and blobs
# blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_string)
# containers = blob_service_client.list_containers()
# list_containers = [container for container in containers]
# list_containers












