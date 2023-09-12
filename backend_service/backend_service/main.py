



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


from backend_service.utilities.mlflow_training_class import mlflow_training
from backend_service.utilities.mlflow_class import mlflow_model
from backend_service.utilities.data_preprocess import data_preprocessing





def read_configuration(configuration_file_path):
    
    with open(configuration_file_path) as file:
        config = yaml.full_load(file)
        
    return config





load_dotenv()

local_run = getenv("LOCAL_RUN", False)


# def get_config(local_run: bool):
    
    
#     load_dotenv()
#     local_run = getenv("LOCAL_RUN", False)

#     if local_run:
#         try:
#             config = read_configuration("./configuration/local_run.yaml")
#         except Exception as e:
#             print(e)
#             config = read_configuration("./backend_service/configuration/production_run.yaml")
#     else:
#         try:
#             config = read_configuration("configuration/production_run.yaml")
#         except Exception as e:
#             print(e)
#             config = read_configuration("./backend_service/configuration/production_run.yaml")
        
#     return config




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













