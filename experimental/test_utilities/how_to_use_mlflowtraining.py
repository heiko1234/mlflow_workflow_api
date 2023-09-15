



import numpy as np


import pandas as pd


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





data = pd.DataFrame()
data["BiologicalMaterial02"] = [55, 54, 52, 58, 60.5]
data["BiologicalMaterial06"] = [45, 55.5, 50.5, 59, 58]
data["ManufacturingProcess06"] = [204, 215, 210, 209, 204]

# 55*0.4+45*0.2+204*0.04   # 39.16
# 54*0.4+55.5*0.2+215*0.04  # 41.3
# 52*0.4+50.5*0.2+210*0.04  # 39.3
# 58*0.4+59*0.2+209*0.04    # 43.36



# create a test dataset with 20 entries
data = pd.DataFrame()
# add a row for BiologicalMaterial02 with 20 entries between 52 and 61
data["BiologicalMaterial02"] = np.random.randint(52, 61, 20)
data["BiologicalMaterial06"] = np.random.randint(45, 60, 20)
data["ManufacturingProcess06"]  = np.random.randint(204, 215, 20)

data["ManufacturingProcess06"] = data["ManufacturingProcess06"] + np.random.normal(0, 0.2, 20)
data["BiologicalMaterial06"] = data["BiologicalMaterial06"] + np.random.normal(0, 0.1, 20)
data["BiologicalMaterial02"] = data["BiologicalMaterial02"] + np.random.normal(0, 0.3, 20)

# ad a row for "Yield" based on the formula: BiologicalMaterial02*0.4+BiologicalMaterial06*0.2+ManufacturingProcess06*0.04
data["Yield"] = data["BiologicalMaterial02"]*0.4+data["BiologicalMaterial06"]*0.2+data["ManufacturingProcess06"]*0.04

# modify column Yield with a bit of noise 
data["Yield"] = data["Yield"] + np.random.normal(0, 0.2, 20)

data




mlflow_training_obj = mlflow_training(model_name="Project_name")


desc_df = mlflow_training_obj.descriptiontable_with_correlation(data, target="Yield")
desc_df

# transformation_dict = {"Yield": "no transformation", "BioMaterial1": "log", "BioMaterial2": "sqrt", "ProcessValue1": "1/x"}



# transformation_dict = {"Yield": "no transformation", "BiologicalMaterial02": "log", "BiologicalMaterial06": "sqrt", "ManufacturingProcess06": "1/x"}


transformation_dict = {
    "Yield": "no transformation",
    "BiologicalMaterial02": "no transformation", 
    "BiologicalMaterial06": "no transformation",
    "ManufacturingProcess06": "no transformation"
    }


transformation_dict = {
    "Yield": "no transformation",
    "BiologicalMaterial02": "log", 
    "BiologicalMaterial06": "no transformation",
    "ManufacturingProcess06": "1/x"
    }




mlflow_training_obj.make_model(
    data=data,
    target="Yield",
    features=["BiologicalMaterial02", "BiologicalMaterial06", "ManufacturingProcess06"],
    test_size = 0.2,
    scaler_expand_by="std",
    model_name="Project_name",
    transformation_dict=transformation_dict,
    model_parameter=None,
    model_typ="linear_regression"
    )







# #############################
# 
###############################

# have a look at the mlflow ui


mlflow_training_obj = mlflow_training(model_name="Project_name")


mlflow_training_obj


target = "Yield"
features = ["BiologicalMaterial02", "BiologicalMaterial06", "ManufacturingProcess06"]



target_and_features = [target] + features

df_tf = data[target_and_features].copy()

df_tf
# >>> df_tf
#     Yield  BiologicalMaterial02  BiologicalMaterial06  ManufacturingProcess06
# 0   43.12                 64.47                 54.45                   210.0
# 1   43.06                 63.60                 54.72                   211.7
# 2   41.49                 63.60                 54.72                   208.7
# 3   42.45                 63.60                 54.72                   209.8
# 4   42.04                 61.23                 52.83                   209.4
# ..    ...                   ...                   ...                     ...
# 81  37.30                 51.75                 44.30                   206.6
# 82  37.86                 51.83                 44.57                   204.6
# 83  38.05                 51.28                 44.74                   206.4
# 84  37.87                 51.44                 44.73                   206.6
# 85  38.60                 51.44                 44.73                   205.5



transformation_dict = {"Yield": "no transformation", "BioMaterial1": "log", "BiologicalMaterial02": "sqrt", "ManufacturingProcess06": "1/x"}



df_tf_t = mlflow_training_obj.transform_rawdata(data=df_tf, transformation_dict=transformation_dict)
df_tf_t


dft = mlflow_training_obj.descriptiontable(df_tf_t)
dft


scaler_expand_by = "std"  # "std" or None

new_minmaxscalingdf = mlflow_training_obj.make_minmaxscalingtable_by_descriptiontable(
    descriptiontable=dft, 
    expand_by=scaler_expand_by)

new_minmaxscalingdf



# Notwendigkeit von data_minmax_dict mit und ohne transformation der daten, z.B. direkt Eingabe Limit check
# und transformiert für den scaler


data_minmax_dict = mlflow_training_obj.create_data_minmax_dict(new_minmaxscalingdf)
data_minmax_dict


feature_minmax_dict = mlflow_training_obj.create_data_minmax_dict(new_minmaxscalingdf.loc[:, features])
feature_minmax_dict
target_minmax_dict = mlflow_training_obj.create_data_minmax_dict(new_minmaxscalingdf.loc[:, target])
target_minmax_dict


feature_dtypes_dict = mlflow_training_obj.create_data_dtype_dict(new_minmaxscalingdf.loc[:,features])


