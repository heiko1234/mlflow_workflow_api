



import os
import yaml
import numpy as np
import pandas as pd

from os import getenv
from dotenv import load_dotenv
from datetime import datetime

from typing import Dict, List, Union, Set, Optional

from fastapi import FastAPI, Depends, Request

from pydantic import BaseModel, Field, validator


# from backend_service.utilities.mlflow_training_class import mlflow_training
# from backend_service.utilities.mlflow_predict_class import mlflow_model
# from backend_service.utilities.data_preprocess import data_preprocessing



import polars as pl

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient, BlobBlock



from dotenv import load_dotenv

load_dotenv()







def read_configuration(configuration_file_path):
    
    with open(configuration_file_path) as file:
        config = yaml.full_load(file)
        
    return config





load_dotenv()

local_run = getenv("LOCAL_RUN", False)



class Data_load(BaseModel):
    blobcontainer: str = Field(example="chemical-data")
    subcontainer: str = Field(example="chemical-data")
    file_name: str = Field(example="ChemicalManufacturingProcess.parquet")
    


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



@app.post("/data_statistics")
def post_data_statistics(query_input: Data_load):
    
    local_run = getenv("LOCAL_RUN", False)
    
    if local_run:
    
        account = "devstoreaccount1"
        credential = os.getenv("AZURE_STORAGE_KEY")
        credential
        
    else:

        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)
        
    blobcontainer=query_input.blobcontainer
    subcontainer=query_input.subcontainer
    file=query_input.file_name

    master_data = pl.read_parquet(
        f"az://{blobcontainer}/{subcontainer}/{file}",
        storage_options={"account_name": account, "credential": credential}
        )
    df = master_data.to_pandas()

    dft=df.describe().reset_index(drop = True).T
    dft = dft.reset_index(drop=False)
    dft.columns= ["description", "counts", "mean", "std", "min", "25%", "50%", "75%", "max"]
    dft["nan"]=df.isna().sum().values

    output_df=dft.round(2).to_json(orient='split')
    
    return output_df



@app.post("/make_model")
def make_model():
    
    # TODO: Get data from database from 
    # data = load_data()
    
    

    # data_transformation = {"Yield": "no transformation", "BioMaterial1": "log", "BioMaterial2": "sqrt", "ProcessValue1": "1/x"}


    # data_preprocessed= data_preprocessing(df=data, transformation_dict=data_transformation)


    # mlflow_training_obj = mlflow_training(model_name="Project_name")


    # mlflow_training_obj.make_model(
    #     data=data,
    #     target="Yield",
    #     features=["BiologicalMaterial02", "BiologicalMaterial06", "ManufacturingProcess06"],
    #     test_size = 0.2,
    #     scaler_expand_by="std",
    #     model_name="Project_name",
    #     model_parameter=None,
    #     model_typ="linear_regression"
    #     )
    
    return None













