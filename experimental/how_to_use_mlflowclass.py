



import pandas as pd



df = pd.DataFrame()

df["Yield"] = [44.0, 43.0, 46.0, 40.1, 42.2]
df["BioMaterial1"]=[5.5, 4.5, 3.5, 1.0, 6.0]
df["BioMaterial2"]=[9.5, 9, 5, 10, 12]
df["ProcessValue1"] = [20, 15, 10, 9, 2]


target = "Yield"

features = ["BioMaterial1", "BioMaterial2", "ProcessValue1"]



# /home/heiko/Schreibtisch/Repos/dash_apps/mlflow_workflow_app/experimental/mflow_workflow/how_to_use_mlflowclass.py

from experimental.mlflow_class import mlflow_model







df[features]


my_mlflow_model = mlflow_model(model_name="project_name", staging="Staging")



my_mlflow_model.list_registered_models()

my_mlflow_model.get_model_version()

my_mlflow_model.get_model()

my_mlflow_model.get_features()

my_mlflow_model.get_model_artifact(artifact="feature_dtypes.json")

my_mlflow_model.get_model_artifact(artifact="feature_limits.json")

my_mlflow_model.get_model_artifact(artifact="target_limits.json")


my_mlflow_model.get_feature_minmaxscaler()

my_mlflow_model.get_target_minmaxscaler()


my_mlflow_model.make_predictions(df)

my_mlflow_model.validate_data_columns(df)




df["prediction"] = my_mlflow_model.make_predictions(df)

df





df_single = df.iloc[0, :]
df_single = pd.DataFrame(df_single).T
df_single

my_mlflow_model.make_predictions(df_single)[0]





