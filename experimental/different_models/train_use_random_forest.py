



# from sklearn.ensemble import RandomForestRegressor


# sklearn_model = RandomForestRegressor()

# sklearn_model.get_params()
# >>> sklearn_model.get_params()
# {'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 
#  'max_depth': None, 'max_features': 1.0, 'max_leaf_nodes': None, 
#  'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 
#  'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 
#  'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}






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



# Random Forest


mlflow_training_obj.make_model(
    data=data,
    target="Yield",
    features=["BiologicalMaterial02", "BiologicalMaterial06", "ManufacturingProcess06"],
    test_size = 0.2,
    scaler_expand_by="std",
    model_name="Project_name",
    model_parameter=None,
    model_typ="random_forest"
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

from experimental.mlflow_class import mlflow_model

# oder 
from backend_service.utilities.mlflow_predict_class import mlflow_model





df[features]


my_mlflow_model = mlflow_model(model_name="project_name", staging="Staging")


df["prediction"] = my_mlflow_model.make_predictions(df)

df


predictions = my_mlflow_model.make_predictions(df)
predictions



df_single = df.iloc[0, :]
df_single
# >>> df_single
# Yield                      44.0000
# BiologicalMaterial02       55.0000
# BiologicalMaterial06       40.9000
# ManufacturingProcess06    200.0000
# prediction                 36.4835
# Name: 0, dtype: float64

df_single = pd.DataFrame(df_single).T
df_single
# >>> df_single#
#    Yield  BiologicalMaterial02  BiologicalMaterial06  ManufacturingProcess06  prediction
# 0   44.0                  55.0                  40.9                   200.0     36.4835

my_mlflow_model.make_predictions(df_single)[0]




