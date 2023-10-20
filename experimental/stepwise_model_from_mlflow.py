



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






import pandas as pd

df = pd.DataFrame()


df["BiologicalMaterial02"] = [55, 54, 48, 58, 60.5]
df["BiologicalMaterial06"] = [40.9, 55.5, 50.5, 60.5, 60.5]
df["ManufacturingProcess06"] = [200, 215, 210, 209, 202]

df["Yield"] = [44.0, 43.0, 46.0, 40.1, 42.2]
df



df_scaled = df.copy()
df_scaled[["BiologicalMaterial02", "BiologicalMaterial06", "ManufacturingProcess06"]] = MinMaxScaler().fit_transform(df[["BiologicalMaterial02", "BiologicalMaterial06", "ManufacturingProcess06"]])

df_scaled

# convert df_scaled from float64 to float32
df_scaled = df_scaled.astype(np.float32)

df_scaled




df_predictions = model.predict(df_scaled)


df_predictions
# array([0.25756781, 0.55341868, 0.53458307, 0.66981172, 0.34144465])


print("MinMaxScaler Target")
target = "Yield"
target_minmaxscaler = MinMaxScaler()
target_minmax_list = list(df.loc[:, target])
target_minmax_list = np.array(target_minmax_list)
target_minmax_list
# array([44. , 43. , 46. , 40.1, 42.2])
target_minmax_list = target_minmax_list.reshape(-1, 1)
target_minmax_list
# array([[44. ],
#        [43. ],
#        [46. ],
#        [40.1],
#        [42.2]])

target_minmaxscaler.fit(target_minmax_list)


df_predictions.shape
len(df_predictions.shape)



if len(df_predictions.shape) == 1:
    df_predictions = df_predictions.reshape(-1, 1)

elif len(df_predictions.shape) == 2:
    if df_predictions.shape[1] is None:
        df_predictions = df_predictions.reshape(-1, 1)


    elif df_predictions.shape[1] == 0:
        df_predictions = df_predictions.reshape(-1, 1)



df_predictions
# array([[0.25756781],
#        [0.55341868],
#        [0.53458307],
#        [0.66981172],
#        [0.34144465]])


output = target_minmaxscaler.inverse_transform(df_predictions)
output





