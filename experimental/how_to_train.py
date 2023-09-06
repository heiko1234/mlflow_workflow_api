







import pandas as pd







df = pd.DataFrame()

df["Yield"] = [44.0, 43.0, 46.0, 40.1, 42.2]
df["BioMaterial1"]=[5.5, 4.5, 3.5, 1.0, 6.0]
df["BioMaterial2"]=[9.5, 9, 5, 10, 12]
df["ProcessValue1"] = [20, 15, 10, 9, 2]


target = "Yield"

features = ["BioMaterial1", "BioMaterial2", "ProcessValue1"]


target_and_fetures = [target] + features







df = df[target_and_fetures]


dft=df.describe().reset_index(drop = True).T
dft = dft.reset_index(drop=False)
dft.columns= ["description", "counts", "mean", "std", "min", "25%", "50%", "75%", "max"]
dft["nan"]=df.isna().sum().values



df 
# >>> df 
#    Yield  BioMaterial1  BioMaterial2  ProcessValue1
# 0   44.0           5.5           9.5             20
# 1   43.0           4.5           9.0             15
# 2   46.0           3.5           5.0             10
# 3   40.1           1.0          10.0              9
# 4   42.2           6.0          12.0              2

dft

# >>> dft
#      description  counts   mean       std   min   25%   50%   75%   max  nan
# 0          Yield     5.0  43.06  2.181284  40.1  42.2  43.0  44.0  46.0    0
# 1   BioMaterial1     5.0   4.10  1.981161   1.0   3.5   4.5   5.5   6.0    0
# 2   BioMaterial2     5.0   9.10  2.559297   5.0   9.0   9.5  10.0  12.0    0
# 3  ProcessValue1     5.0  11.20  6.760178   2.0   9.0  10.0  15.0  20.0    0







def make_minmaxscalingtable_by_descriptiontable(descriptiontable, expand_by=None):
    
    output_df = pd.DataFrame()
    
    if expand_by == None:
        
        for row_index in range(descriptiontable.shape[0]):
            output_df[descriptiontable.loc[row_index, "description"]] = [descriptiontable.loc[row_index, "max"], descriptiontable.loc[row_index, "min"]]

    elif expand_by is "std":
            
            for row_index in range(descriptiontable.shape[0]):
                output_df[descriptiontable.loc[row_index, "description"]] = [descriptiontable.loc[row_index, "max"]+ descriptiontable.loc[row_index, "std"], descriptiontable.loc[row_index, "min"]- descriptiontable.loc[row_index, "std"]]


    return output_df



new_minmaxscalingdf = make_minmaxscalingtable_by_descriptiontable(descriptiontable=dft, expand_by="std")
new_minmaxscalingdf

# >>> new_minmaxscalingdf
#    Yield  BioMaterial1  BioMaterial2  ProcessValue1
# 0   47.5           7.0          10.5          25.00
# 1   39.0           0.0           4.5          -3.34


new_minmaxscalingdf = make_minmaxscalingtable_by_descriptiontable(descriptiontable=dft, expand_by=None)
new_minmaxscalingdf

# >>> new_minmaxscalingdf
#    Yield  BioMaterial1  BioMaterial2  ProcessValue1
# 0   46.0           6.0          12.0           20.0
# 1   40.1           1.0           5.0            2.0





def create_data_minmax_dict(data):
    
    if isinstance(data, pd.DataFrame):

        feature_data_minmax = data.describe().loc[["min", "max"], :]

        output = feature_data_minmax.to_dict()
    
    else:
        output = {data.name: {"min": data.min(), "max": data.max()}}
    
    return output





data_minmax_dict = create_data_minmax_dict(new_minmaxscalingdf)
data_minmax_dict

# {'Yield': {'min': 40.1, 'max': 46.0}, 'BioMaterial1': {'min': 1.0, 'max': 6.0}, 'BioMaterial2': {'min': 5.0, 'max': 12.0}, 'ProcessValue1': {'min': 2.0, 'max': 20.0}}


df[features]

feature_minmax_dict = create_data_minmax_dict(df[features])
feature_minmax_dict



df[target]

target_minmax_dict = create_data_minmax_dict(df[target])
target_minmax_dict


# make a dataframe from a series
# https://stackoverflow.com/questions/17839973/construct-pandas-dataframe-from-items-in-nested-dictionary






import numpy as np


from sklearn.preprocessing import MinMaxScaler

feature_minmaxscaler = MinMaxScaler()




feature_minmaxscaler.fit(new_minmaxscalingdf.loc[:, features])


fitted_df_features = feature_minmaxscaler.transform(df.loc[:,features])

fitted_df_features

# >>> fitted_df_features
# array([[0.9       , 0.64285714, 1.        ],
#        [0.7       , 0.57142857, 0.72222222],
#        [0.5       , 0.        , 0.44444444],
#        [0.        , 0.71428571, 0.38888889],
#        [1.        , 1.        , 0.        ]])



target_minmax_list = list(new_minmaxscalingdf.loc[:, target])
target_minmax_list
# [48.18128402552258, 37.91871597447742]

# convert with numpy in array
target_minmax_list = np.array(target_minmax_list)
target_minmax_list
# array([48.18128403, 37.91871597])

target_minmax_list = target_minmax_list.reshape(-1, 1)
target_minmax_list
# array([[48.18128403],
#        [37.91871597]])



target_minmaxscaler = MinMaxScaler()
target_minmaxscaler.fit(target_minmax_list)


fitted_target = target_minmaxscaler.transform(df.loc[:,[target]])

# MinMaxScaler()
# >>> fitted_target = target_minmaxscaler.transform(df.loc[:,[target]])
# /home/heiko/Schreibtisch/Repos/dash_apps/mlflow_workflow_app/.venv/lib/python3.10/site-packages/sklearn/base.py:457: UserWarning: X has feature names, but MinMaxScaler was fitted without feature names
#   warnings.warn(


fitted_target

# array([[0.59256942],
#        [0.49512793],
#        [0.78745242],
#        [0.21254758],
#        [0.41717473]])



from sklearn.model_selection import train_test_split


target_data = fitted_target
feature_data = fitted_df_features


test_size = 0.2  # 20% test data, 80% train data
random_state = 2023  # random seed

(
    features_train,
    features_test,
    target_train,
    target_test,
) = train_test_split(
    feature_data,
    target_data,
    test_size=test_size,
    random_state=random_state,
    )


# Train
features_train
target_train

# Test
features_test
target_test





pandas_dtypes = {
    "float64": "float",
    "int64": "integer",
    "bool": "boolean",
    "double": "double",
    "object": "string",
    "binary": "binary",
}



def create_feature_dtype_dict(data, pandas_dtypes):
    output = {}

    for element in data.columns:
        output[element] = pandas_dtypes[str(data.dtypes[element])]

    return output




feature_dtypes_dict = create_feature_dtype_dict(
    data = df[features],
    pandas_dtypes = pandas_dtypes)
feature_dtypes_dict
# {'BioMaterial1': 'float', 'BioMaterial2': 'float', 'ProcessValue1': 'integer'}



from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec


input_schema = Schema(
    [
        ColSpec(
            pandas_dtypes[str(df[features].dtypes[element])], element
        )
        for element in df[features].columns
    ]
)
output_schema = Schema(
    [ColSpec(pandas_dtypes[str(df[target].dtypes)])]
)
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

signature

# >>> signature
# inputs: 
#   ['BioMaterial1': float, 'BioMaterial2': float, 'ProcessValue1': integer]
# outputs: 
#   [float]



input_schema
output_schema

signature









import mlflow
import mlflow.sklearn

# mlflow.tracking.get_tracking_uri()
# 'file:///home/heiko/Schreibtisch/Repos/dash_apps/mlflow_workflow_app/mlruns'


# mlflow.set_registry_uri("sqlite:///mlflow.db")


MLFlow_Experiment = "Project_name"


# import variables from .env file
import os
from dotenv import load_dotenv


load_dotenv()





MLFlow_Experiment = "Project_name"

mlflow.set_experiment(MLFlow_Experiment)

mlflow.is_tracking_uri_set()




from sklearn.linear_model import LinearRegression

sk_model = LinearRegression()








features_train
target_train






with mlflow.start_run():

    sk_model.fit(features_train, target_train)

    train_score = round(sk_model.score(features_train, target_train), 4)
    train_score


    test_score = round(sk_model.score(features_test, target_test), 4)
    test_score
    
    mlflow.autolog()


    mlflow.log_params(sk_model.get_params())
    mlflow.log_metric("train_score", train_score)
    mlflow.log_metric("test_score", test_score)

    mlflow.sklearn.log_model(
        sk_model, "model", signature=signature
    )
    
    mlflow.log_dict(
        data_minmax_dict, "model/data_limits.json"
    )
    mlflow.log_dict(
        feature_minmax_dict, "model/feature_limits.json"
    )
    
    mlflow.log_dict(
        target_minmax_dict, "model/target_limits.json"
    )
    
    
    
    mlflow.log_dict(
        feature_dtypes_dict, "model/feature_dtypes.json"
    )


mlflow.end_run()





