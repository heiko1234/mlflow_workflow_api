



import os
import yaml
import numpy as np
import pandas as pd


from dotenv import load_dotenv
from datetime import datetime

from typing import Dict, List, Union, Set, Optional

from fastapi import FastAPI, Depends, Request

from pydantic import BaseModel, Field, validator


# from backend_service.utilities.mlflow_training_class import mlflow_training
# from backend_service.utilities.mlflow_predict_class import mlflow_model
# from backend_service.utilities.data_preprocess import data_preprocessing



import polars as pl

from upath import UPath



from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient, BlobBlock



from dotenv import load_dotenv

load_dotenv()







def read_configuration(configuration_file_path):
    
    with open(configuration_file_path) as file:
        config = yaml.full_load(file)
        
    return config





load_dotenv()

local_run = os.getenv("LOCAL_RUN", False)



class Data_load(BaseModel):
    blobcontainer: str | None = Field(example="chemical-data")
    subcontainer: str | None = Field(example="chemical-data")
    file_name: str | None = Field(example="ChemicalManufacturingProcess.parquet")
    account: str | None = Field(example="devstoreaccount1")


app = FastAPI()


@app.get("/")
def read_root():
    return {"Connected to Backend Service"}


@app.get("/health")
def health():
    return True


@app.get("/testdata")
def testdata():
    return {"testdata": "testdata"}





@app.get("/list_available_accounts")
def get_available_blobs():

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:
        account = "devstoreaccount1"

    else:
        account = os.environ["AZURE_STORAGE_ACCOUNT"]

    return account



@app.post("/list_available_blobs")
def get_available_blobs(query_input: Data_load):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:
        account = "devstoreaccount1"
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:
        # account = os.environ["AZURE_STORAGE_ACCOUNT"]
        account=query_input.account
        credential = DefaultAzureCredential(exclude_environment_credential=True)


    path = UPath("az://", account_name=account, anon=True, credential=credential)


    output = [str(p).split("/")[-1] for p in path.iterdir()]

    return output


@app.post("/list_available_subblobs")
def get_available_subblobs(query_input: Data_load):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:
        account = "devstoreaccount1"
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:
        # account = os.environ["AZURE_STORAGE_ACCOUNT"]
        account=query_input.account
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query_input.blobcontainer


    path = UPath(f"az://{blobcontainer}", account_name=account, anon=True, credential=credential)


    output = [str(p).split("/")[-1] for p in path.iterdir()]

    return output



@app.post("/list_available_files")
def get_available_files(query_input: Data_load):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:
        account = "devstoreaccount1"
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:
        # account = os.environ["AZURE_STORAGE_ACCOUNT"]
        account=query_input.account
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query_input.blobcontainer
    subcontainer=query_input.subcontainer


    path = UPath(f"az://{blobcontainer}/{subcontainer}", account_name=account, anon=True, credential=credential)


    output = [str(p).split("/")[-1] for p in path.iterdir()]

    return output





@app.post("/data_statistics")
def post_data_statistics(query_input: Data_load):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:

        # account = "devstoreaccount1"
        account=query_input.account
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:

        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query_input.blobcontainer
    subcontainer=query_input.subcontainer
    file=query_input.file_name

    if (blobcontainer is not None) and (subcontainer is not None) and (file is not None):

        print("blobcontainer, subcontainer, file for data statistics")
        master_data = pl.read_parquet(
            f"az://{blobcontainer}/{subcontainer}/{file}",
            storage_options={"account_name": account, "credential": credential}
            )
        df = master_data.to_pandas()

        dft=df.describe().reset_index(drop = True).T
        dft = dft.reset_index(drop=False)
        dft.columns= ["description", "counts", "mean", "std", "min", "25%", "50%", "75%", "max"]
        dft["nan"]=df.isna().sum().values


        digits = 2
        output_df=dft.round(digits).to_json(orient='split')

        print("data statistics done")

    else:
        output_df = None

    return output_df


















# @app.post("/make_model")
# def make_model():
    
#     # TODO: Get data from database from 
#     # data = load_data()
    
    

#     # data_transformation = {"Yield": "no transformation", "BioMaterial1": "log", "BioMaterial2": "sqrt", "ProcessValue1": "1/x"}


#     # data_preprocessed= data_preprocessing(df=data, transformation_dict=data_transformation)


#     # mlflow_training_obj = mlflow_training(model_name="Project_name")


#     # mlflow_training_obj.make_model(
#     #     data=data,
#     #     target="Yield",
#     #     features=["BiologicalMaterial02", "BiologicalMaterial06", "ManufacturingProcess06"],
#     #     test_size = 0.2,
#     #     scaler_expand_by="std",
#     #     model_name="Project_name",
#     #     model_parameter=None,
#     #     model_typ="linear_regression"
#     #     )
    
#     return None













