


import os
import io
import yaml

from pathlib import Path
import pandas as pd

import pyarrow.fs as fs 
import pyarrow.parquet as pq

from adlfs import AzureBlobFileSystem

from azure.core.exceptions import (
    ClientAuthenticationError,
    ResourceNotFoundError,
    ServiceRequestError
    
)

from azure.identity import AzureCliCredential
from azure.identity import AzureCliCredential as AioAzureCliCredential
from azure.storage.blob import BlobServiceClient, ContainerClient

from azure.identity import DefaultAzureCredential
from azure.identity import EnvironmentCredential


from dotenv import load_dotenv


class BlobStorageConnector:
    def __init__(self, storage_account_name="any_global_stroage", container_name=None, local_run=False):
        
        load_dotenv()
        
        self.url = f"https://{storage_account_name}.blob.core.windows.net"
        self.storage_account_name = storage_account_name
        self.local_run = local_run
        self.container_name = container_name
        
        try:
            self.blobclient = self.get_client_by_string(container_name, local_run)
            next(self.blobclient.list_blobs())
        except (ServiceRequestError):
            pass
        except (ResourceNotFoundError, ValueError, ClientAuthenticationError, KeyError, AttributeError) as e:
            print(e)
            print("No connection established")
            self.blobclient = self.get_client_by_cli(storage_account_name, container_name)
            
    def get_client_by_cli(self, storage_account_name, container_name):

        credential = self.get_credentials()
        blob_service_client = BlobServiceClient(
            account_url=self.url,
            credential=credential
        )
        blob_container_client = blob_service_client.get_container_client(container_name)
        print("client by cli")
        return blob_container_client
    
    
    def construct_connectionstring(self, local_run):
        
        if local_run:
            account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
            account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
            
            if account_name != "devstoreaccount1":
                connection_string = ";".join(
                            [
                                "DefaultEndpointsProtocol=http",
                                f"AccountName={account_name}",
                                f"AccountKey={account_key}",
                                f"EndpointSuffix=core.windows.net",
                            ]
                )
            else:
                connection_string = ";".join(
                    [
                        "DefaultEndpointsProtocol=https",
                        f"AccountName={account_name}",
                        f"AccountKey={account_key}",
                        f"BlobEndpoint=http://127.0.0.1:10000/{account_name}",
                        f"QueueEndpoint=http://127.0.0.1:10001/{account_name}",
                    ]
                )
        
        else:
            connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
            
        return connection_string
    
    def get_client_by_string(self, container_name, local_run):
        print("try to get client by string")
        
        if local_run:
            try:
                connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            except Exception as e:
                print(e)
                connection_string = self.construct_connectionstring(local_run)
        
        else:
            connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
            
        blob_container_client = ContainerClient.from_connection_string(
            connection_string,
            container_name
            )
        print("client by string")
        return blob_container_client
    
    def get_credentials(self):
        try:
            credential = AzureCliCredential()
        except Exception as e:
            print(e)
            credential = DefaultAzureCredential(exclude_environment_credential=True)
        return credential
        
    def read_anyfile(self, subcontainer, file):
        try:
            blob_str = subcontainer + "/" + file
            bytes = self.blobclient.get_blob_client(blob_str).download_blob().readall()
            pq_file = io.BytesIO(bytes)
            return pq_file
        
        except Exception as e:
            print(e)
            return None
        
    def list_all_files(self, subcontainer, files_with):
        output = []
        for blob in self.blobclient.list_blobs():
            if subcontainer in blob.name and files_with in blob.name:
                output.append(blob.name.split("/")[1])
        return output
    
    def read_parquet(self, subcontainer, file):
        try:
            pq_file = self.read_anyfile(subcontainer, file)
            df = pd.read_parquet(pq_file, engine="pyarrow")
            return df
        
        except Exception as e:
            print(e)
            return None
        
    def read_partioned_parquet(self, subcontainer, file):
        try:
            credentials = self.get_credentials()
            
            adfs = AzureBlobFileSystem(
                account_name=self.storage_account_name,
                credential=credentials,
                anon=False,
            )
            file_system = fs.FileSystem(adfs)
            
            file_path = f"{subcontainer}/{file}"
            
            df = pq.read_table(
                source=self.container_name+file_path,
                columns = None,
                filesystem=file_system,
                filters=None).to_pandas()

            return df
        
        except Exception as e:
            print(e)
            return None
        
    def read_yaml(self, subcontainer, filename):
        yaml_file = self.read_anyfile(subcontainer, filename)
        if yaml_file:
            output = yaml.safe_load(yaml_file)
        else:
            output = None
        
        return output
    
    def upload_anydata_to_blob(self, df, container_name, subcontainer, filename):

        relative_filename = f"./data/{subcontainer}/{filename}"
        file_path = Path(relative_filename)
        
        dir_to_create = file_path.parent
        os.makedirs(dir_to_create, exist_ok=True)
        
        blob_str = f"{subcontainer}/{filename}"
        client = self.blobclient.get_blob_client(blob_str)
        
        with file_path.open("rb") as data:
            client.upload_blob(data, overwrite=True)
            
        print("upload successful")
        
        Path(relative_filename).unlink()


    def upload_data_to_blob(self, df, container, subcontainer, filename, filetype="parquet"):
        
        relative_filename = f"./data/{subcontainer}/{filename}"
        file_path = Path(relative_filename)
        
        dir_to_create = file_path.parent
        os.makedirs(dir_to_create, exist_ok=True)
        
        if filetype == "parquet":
            df.to_parquet(file_path, engine="pyarrow")
        elif filetype == "csv":
            df.to_csv(file_path, index=False)
        elif filetype == "json":
            df.to_json(file_path, orient="records")
        elif filetype == "yaml":
            with file_path.open("w") as f:
                yaml.dump(df, f)
        else:
            print("filetype not supported")
            return None
        
        blob_str = f"{subcontainer}/{filename}"
        client = self.blobclient.get_blob_client(blob_str)
        
        with file_path.open("rb") as data:
            client.upload_blob(data, overwrite=True)
            
        print("upload successful")
        
        Path(relative_filename).unlink()




