

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
staging = "None"


azure_model_dir = "models:/"

if staging == "Staging":
    artifact_path = str(PurePosixPath(azure_model_dir).joinpath(model_name, "Staging"))
elif staging == "Production":
    artifact_path = str(PurePosixPath(azure_model_dir).joinpath(model_name, "Production"))
elif staging == "None":
    artifact_path = str(PurePosixPath(azure_model_dir).joinpath(model_name, "None"))
else:
    print("staging must be either 'Staging' or 'Production', 'None'")
    raise ValueError

model = mlflow.pyfunc.load_model(artifact_path)



model.metadata.model_uuid
# >>> model.metadata.model_uuid
# '394fad5160aa4d21bd72bdae7454e567'
model.metadata.run_id
# >>> model.metadata.run_id
# '1975ac510ad04b3fbdd1578d46c746eb'

model.metadata.get_model_info()


# get version of model in staging
# client = MlflowClient()
client.get_latest_versions(name=model_name, stages=["Staging"])[0].version   # 9


client.get_latest_versions(name=model_name, stages=["None"])[0].version   # 9



client.get_latest_versions(name=model_name, stages=["Production"])[0].version   # 9



# register a fresh trained model, and stage it to 'staging'
#
# # https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model
# # https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model





# The second way is to use the mlflow.register_model() method, after all your experiment runs complete and when you have decided which model is most suitable to add to the registry. For this method, you will need the run_id as part of the runs:URI argument.
# result = mlflow.register_model(
#     "runs:/d16076a3ec534311817565e6527539c0/sklearn-model", "sk-learn-random-forest-reg"
# )




# # https://mlflow.org/docs/latest/model-registry.html
# client = MlflowClient()
# client.transition_model_version_stage(
#     name="sk-learn-random-forest-reg-model", version=3, stage="Production"
# )



# TODO: https://mlflow.org/docs/latest/model-registry.html#adding-an-mlflow-model-to-the-model-registry






