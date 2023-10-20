



import pandas as pd


# from experimental.mlflow_training_class import mlflow_training





from backend_service.utilities.mlflow_training_class import mlflow_training




data = pd.read_parquet("./data/ChemicalManufacturingProcess.parquet")
data

list(data.columns)

# >>> list(data.columns)
# ['Yield', 'BiologicalMaterial01', 'BiologicalMaterial02', 'BiologicalMaterial03', 
#  'BiologicalMaterial04', 'BiologicalMaterial05', 'BiologicalMaterial06', 
#  'BiologicalMaterial07', 'BiologicalMaterial08', 'BiologicalMaterial09', 
#  'BiologicalMaterial10', 'BiologicalMaterial11', 'BiologicalMaterial12', 
#  'ManufacturingProcess01', 'ManufacturingProcess02', 'ManufacturingProcess03', 
#  'ManufacturingProcess04', 'ManufacturingProcess05', 'ManufacturingProcess06', 
#  'ManufacturingProcess07', 'ManufacturingProcess08', 'ManufacturingProcess09', 
#  'ManufacturingProcess10', 'ManufacturingProcess11', 'ManufacturingProcess12', 
#  'ManufacturingProcess13', 'ManufacturingProcess14', 'ManufacturingProcess15', 
#  'ManufacturingProcess16', 'ManufacturingProcess17', 'ManufacturingProcess18', 
#  'ManufacturingProcess19', 'ManufacturingProcess20', 'ManufacturingProcess21', 
#  'ManufacturingProcess22', 'ManufacturingProcess23', 'ManufacturingProcess24', 
#  'ManufacturingProcess25', 'ManufacturingProcess26', 'ManufacturingProcess27', 
#  'ManufacturingProcess28', 'ManufacturingProcess29', 'ManufacturingProcess30', 
#  'ManufacturingProcess31', 'ManufacturingProcess32', 'ManufacturingProcess33', 
#  'ManufacturingProcess34', 'ManufacturingProcess35', 'ManufacturingProcess36', 
#  'ManufacturingProcess37', 'ManufacturingProcess38', 'ManufacturingProcess39', 
#  'ManufacturingProcess40', 'ManufacturingProcess41', 'ManufacturingProcess42', 
#  'ManufacturingProcess43', 'ManufacturingProcess44', 'ManufacturingProcess45']






mlflow_training_obj = mlflow_training(model_name="Project_name")




desc_df = mlflow_training_obj.descriptiontable_with_correlation(data, target="Yield")
desc_df





mlflow_training_obj.make_model(
    data=data,
    target="Yield",
    features=["BiologicalMaterial02", "BiologicalMaterial06", "ManufacturingProcess06"],
    test_size = 0.2,
    scaler_expand_by="std",
    model_name="Project_name",
    model_parameter=None,
    model_typ="linear_regression"
    )










import pandas as pd



df = pd.DataFrame()

df["Yield"] = [44.0, 43.0, 46.0, 40.1, 42.2]





df["BiologicalMaterial02"] = [55, 54, 48, 58, 60.5]
df["BiologicalMaterial06"] = [40.9, 55.5, 50.5, 60.5, 60.5]
df["ManufacturingProcess06"] = [200, 215, 210, 209, 202]


df



target = "Yield"




features = ["BiologicalMaterial02", "BiologicalMaterial06", "ManufacturingProcess06"]


# /home/heiko/Schreibtisch/Repos/dash_apps/mlflow_workflow_app/experimental/mflow_workflow/how_to_use_mlflowclass.py

# from experimental.mlflow_class import mlflow_model



from backend_service.utilities.mlflow_predict_class import mlflow_model





df[features]


my_mlflow_model = mlflow_model(model_name="project_name", staging="Staging")


df["prediction"] = my_mlflow_model.make_predictions(df)

df
# >>> df
#    Yield  BiologicalMaterial02  BiologicalMaterial06  ManufacturingProcess06  prediction
# 0   44.0                  55.0                  40.9                     200   38.897485
# 1   43.0                  54.0                  55.5                     215   40.600306
# 2   46.0                  48.0                  50.5                     210   37.881810
# 3   40.1                  58.0                  60.5                     209   39.944640
# 4   42.2                  60.5                  60.5                     202   39.087551


# only a damm singel row


df_single = df.iloc[0, :]
df_single = pd.DataFrame(df_single).T
df_single

my_mlflow_model.make_predictions(df_single)[0]


# >>> my_mlflow_model.make_predictions(df_single)[0]
# Feature BiologicalMaterial06 has a lower limit of 44.3 but the data has a lower limit of 40.9
# features: ['BiologicalMaterial02', 'BiologicalMaterial06', 'ManufacturingProcess06']
# 'NoneType' object is not subscriptable
# Column Yield not transformed! Yield is not in transformation_dict
# 'NoneType' object is not subscriptable
# Column BiologicalMaterial02 not transformed! BiologicalMaterial02 is not in transformation_dict
# 'NoneType' object is not subscriptable
# Column BiologicalMaterial06 not transformed! BiologicalMaterial06 is not in transformation_dict
# 'NoneType' object is not subscriptable
# Column ManufacturingProcess06 not transformed! ManufacturingProcess06 is not in transformation_dict
# 'NoneType' object is not subscriptable
# Column prediction not transformed! prediction is not in transformation_dict
# /home/heiko/Schreibtisch/Repos/dash_apps/mlflow_workflow_api/.venv/lib/python3.10/site-packages/sklearn/base.py:457: UserWarning: X has feature names, but LinearRegression was fitted without feature names
#   warnings.warn(
# 38.89748525461662

