



import pandas as pd



df = pd.DataFrame()





df["Yield"] = [44.0, 43.0, 46.0, 40.1, 42.2]


df["BioMaterial1"]=[5.5, 4.5, 3.5, 1.0, 6.0]
df["BioMaterial2"]=[9.5, 9, 5, 10, 12]
df["ProcessValue1"] = [20, 15, 10, 9, 2]




df["BiologicalMaterial02"] = [55, 54, 52, 58, 60.5]
df["BiologicalMaterial06"] = [45, 55.5, 50.5, 59, 58]
df["ManufacturingProcess06"] = [204, 215, 210, 209, 204]


df



# other test data

data = pd.DataFrame()
data["BiologicalMaterial02"] = [55, 54, 52.5, 58, 59]
data["BiologicalMaterial06"] = [45, 55.5, 50.5, 57.5, 56]
data["ManufacturingProcess06"] = [204, 213.0, 210, 209, 204]

data["Yield"] = data["BiologicalMaterial02"]*0.4+data["BiologicalMaterial06"]*0.2+data["ManufacturingProcess06"]*0.04


data


# 55*0.4+45*0.2+204*0.04   # 39.16
# 54*0.4+55.5*0.2+215*0.04  # 41.3
# 52*0.4+50.5*0.2+210*0.04  # 39.3
# 58*0.4+59*0.2+209*0.04    # 43.36







target = "Yield"
features = ["BiologicalMaterial02", "BiologicalMaterial06", "ManufacturingProcess06"]




features = ["BioMaterial1", "BioMaterial2", "ProcessValue1"]



# /home/heiko/Schreibtisch/Repos/dash_apps/mlflow_workflow_app/experimental/mflow_workflow/how_to_use_mlflowclass.py




from backend_service.backend_service.utilities.mlflow_predict_class import mlflow_model







df[features]

data[features]



data
df


# #######################

# Load the model
my_mlflow_model = mlflow_model(model_name="project_name", staging="Staging")

my_mlflow_model = mlflow_model(model_name="project_name", staging="Production")



# ###########################




# test some functions


my_mlflow_model.list_registered_models()

my_mlflow_model.get_model_version()

my_mlflow_model.get_model()

my_mlflow_model.get_features()

my_mlflow_model.get_model_artifact(artifact="feature_limits_unscaled.json")

my_mlflow_model.get_model_artifact(artifact="transformation_dict.json")

my_mlflow_model.get_model_artifact(artifact="feature_dtypes.json")

my_mlflow_model.get_model_artifact(artifact="feature_limits.json")

my_mlflow_model.get_model_artifact(artifact="target_limits.json")


my_mlflow_model.get_feature_minmaxscaler()

my_mlflow_model.get_target_minmaxscaler()



# the starting df
my_mlflow_model.make_predictions(df)






# the other data set

my_mlflow_model.get_model_artifact(artifact="feature_dtypes.json")
my_mlflow_model.get_model_artifact(artifact="transformation_dict.json")

my_mlflow_model.transform_rawdata(data)
my_mlflow_model.scale_and_transform_rawdata(data)
#    BiologicalMaterial02  BiologicalMaterial06  ManufacturingProcess06
# 0              0.416252              0.185245                0.179219
# 1              0.348504              0.660296                0.754849
# 2              0.246882              0.434081                0.562972
# 3              0.619496              0.750781                0.499013
# 4              0.687244              0.682917                0.179219

my_mlflow_model.make_predictions(data)

data["prediction"] = my_mlflow_model.make_predictions(data)
data


# test ende with data






my_mlflow_model.validate_data_columns(df)


df


my_mlflow_model.validate_limits_features(df)


my_mlflow_model.transform_rawdata(df)

my_mlflow_model.calculate_transform_rawdata(df)



df["prediction"] = my_mlflow_model.make_predictions(df)

df





df_single = df.iloc[0, :]
df_single = pd.DataFrame(df_single).T
df_single

my_mlflow_model.make_predictions(df_single)[0]





