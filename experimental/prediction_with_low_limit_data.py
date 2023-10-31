

# TODO

# Saw, when model is made with SPC limits on data, based on data max or min, the prediction output is None. 

# Needs to be evaluated, this is done here








import os
import json
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path


from pathlib import PurePosixPath
from mlflow import MlflowClient


from sklearn.preprocessing import MinMaxScaler


from dotenv import load_dotenv


load_dotenv()





# list all mlflow models


client = MlflowClient()

output = []
for rm in client.search_registered_models():
    output.append(rm.name)


output

# project_name



# call a model

load_dotenv()

model_name = "project_name"
staging = "Staging"



azure_model_dir = "models:/"

if staging == "Staging":
    artifact_path = str(PurePosixPath(azure_model_dir).joinpath(model_name, "Staging"))
elif staging == "Production":
    artifact_path = str(PurePosixPath(azure_model_dir).joinpath(model_name, "Production"))
else:
    print("staging must be either 'Staging' or 'Production'")
    raise ValueError

model = mlflow.pyfunc.load_model(artifact_path)






# load data

df = pd.read_parquet("./data/ChemicalManufacturingProcess.parquet")
df.head()



# load predicting class of model

from backend_service.utilities.data_preprocess import data_preprocessing
# from backend_service.utilities.mlflow_training_class import mlflow_training
from backend_service.utilities.mlflow_predict_class import mlflow_model, list_all_registered_models

from backend_service.utilities.plots import validation_plot



my_fucking_model = mlflow_model(model_name="project_name", staging="Staging")


my_fucking_model.get_model_version()  # 22

my_fucking_model.make_predictions(df)



# ['BiologicalMaterial08', 'ManufacturingProcess11', 'ManufacturingProcess17', 'ManufacturingProcess32', 'ManufacturingProcess45']

df.loc[:,  ['BiologicalMaterial08', 'ManufacturingProcess11', 'ManufacturingProcess17', 'ManufacturingProcess32', 'ManufacturingProcess45']]


df["ManufacturingProcess32"] = df["ManufacturingProcess32"].astype("int32")

df["ManufacturingProcess32"] = df["ManufacturingProcess32"].astype("float32")



df.loc[:,  ['BiologicalMaterial08', 'ManufacturingProcess11', 'ManufacturingProcess17', 'ManufacturingProcess32', 'ManufacturingProcess45']]


df["ManufacturingProcess32"]

df.describe()


my_fucking_model.get_model_artifact("feature_dtypes.json")


data =df


mlflow_dtypes = {
    "float": "float32",
    "integer": "int32",
    "boolean": "bool",
    "double": "double",
    "string": "object",
    "binary": "binary",
}

dtype_dict = my_fucking_model.get_model_artifact(artifact="feature_dtypes.json")

for element in list(dtype_dict.keys()):
    try:
        data[element] = data[element].astype(mlflow_dtypes[dtype_dict[element]])
    except BaseException:
        pass

data

data.loc[:, ['BiologicalMaterial08', 'ManufacturingProcess11', 'ManufacturingProcess17', 'ManufacturingProcess32', 'ManufacturingProcess45']]




my_fucking_model.make_predictions(data)


my_fucking_model.make_predictions(df)


