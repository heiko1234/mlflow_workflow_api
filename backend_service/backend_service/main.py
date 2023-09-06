



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







def read_configuration(configuration_file_path):
    
    with open(configuration_file_path) as file:
        config = yaml.full_load(file)
        
    return config





load_dotenv()

local_run = getenv("LOCAL_RUN", False)


def get_config(local_run: bool):
    
    
    load_dotenv()
    local_run = getenv("LOCAL_RUN", False)

    if local_run:
        try:
            config = read_configuration("./configuration/local_run.yaml")
        except Exception as e:
            print(e)
            config = read_configuration("./backend_service/configuration/production_run.yaml")
    else:
        try:
            config = read_configuration("configuration/production_run.yaml")
        except Exception as e:
            print(e)
            config = read_configuration("./backend_service/configuration/production_run.yaml")
        
    return config




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



















