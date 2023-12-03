




import pandas as pd


# /home/heiko/Schreibtisch/Repos/dash_apps/mlflow_workflow_app/experimental/mflow_workflow/how_to_use_mlflowclass.py

from backend_service.backend_service.utilities.mlflow_predict_class import mlflow_model





my_mlflow_model = mlflow_model(model_name="project_name", staging="Staging")



my_mlflow_model.list_registered_models()

my_mlflow_model.get_model_version()

my_mlflow_model.get_model()

my_mlflow_model.get_features()



my_mlflow_model.get_model_artifact(artifact="feature_dtypes.json")

my_mlflow_model.get_model_artifact(artifact="feature_limits.json")

my_mlflow_model.get_model_artifact(artifact="target_limits.json")

# #################



df = pd.DataFrame()


df["BiologicalMaterial02"] = [55, 60]
df["ManufacturingProcess06"] = [210, 220]
df["Yield"] = [55, 58]


df


my_mlflow_model.validate_data_columns(df)



my_mlflow_model.transform_rawdata(df)
my_mlflow_model.scale_and_transform_rawdata(df)

my_mlflow_model.make_predictions(df)

df["prediction"] = my_mlflow_model.make_predictions(df)

df



df_single = df.iloc[0, :]

df_single
df_single = pd.DataFrame(df_single).T

df_single

my_mlflow_model.make_predictions(df_single)[0]



# #####################


dff = pd.DataFrame()

dff["BiologicalMaterial02"] = [55, 100]
dff["ManufacturingProcess06"] = [210, 220]
dff["Yield"] = [55, 58]


dff


my_mlflow_model.validate_data_columns(dff)



my_mlflow_model.make_predictions(dff)










