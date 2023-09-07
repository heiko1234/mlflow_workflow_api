



import pandas as pd

import os
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path


from pathlib import PurePosixPath



import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec



from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostRegressor




from dotenv import load_dotenv




class mlflow_training():
    
    def __init__(self, model_name=None):
        
        load_dotenv()
        self.model_name = model_name


    def descriptiontable(self, data):
        
        dft=data.describe().reset_index(drop = True).T
        dft = dft.reset_index(drop=False)
        dft.columns= ["description", "counts", "mean", "std", "min", "25%", "50%", "75%", "max"]
        dft["nan"]=data.isna().sum().values
        
        return dft
    
    
    def descriptiontable_with_correlation(self, data, target):
        
        correlations = []
        for i in data.columns:
            if i != target:
                try:
                    corr = np.corrcoef(data[target], data[i])[0, 1]
                    correlations.append(corr)
                except:
                    correlations.append(0)
            else:
                correlations.append(1)
                
        correlations = [round(x, 4) for x in correlations]
        
        dft = self.descriptiontable(data)
        dft["correlation"] = correlations
        
        return dft


    def make_minmaxscalingtable_by_descriptiontable(self, descriptiontable, expand_by=None):
        
        output_df = pd.DataFrame()
        
        if expand_by == None:
            
            for row_index in range(descriptiontable.shape[0]):
                output_df[descriptiontable.loc[row_index, "description"]] = [descriptiontable.loc[row_index, "max"], descriptiontable.loc[row_index, "min"]]

        elif expand_by == "std":
                
                for row_index in range(descriptiontable.shape[0]):
                    output_df[descriptiontable.loc[row_index, "description"]] = [descriptiontable.loc[row_index, "max"]+ descriptiontable.loc[row_index, "std"], descriptiontable.loc[row_index, "min"]- descriptiontable.loc[row_index, "std"]]

        return output_df


    def create_data_minmax_dict(self, data):
    
        if isinstance(data, pd.DataFrame):

            feature_data_minmax = data.describe().loc[["min", "max"], :]

            output = feature_data_minmax.to_dict()
        
        else:
            output = {data.name: {"min": data.min(), "max": data.max()}}
        
        return output


    def create_data_dtype_dict(self, data):
        
        pandas_dtypes = {
            "float64": "float",
            "int64": "integer",
            "bool": "boolean",
            "double": "double",
            "object": "string",
            "binary": "binary",
        }

        output = {}

        for element in data.columns:
            output[element] = pandas_dtypes[str(data.dtypes[element])]

        return output


    def create_signature(self, data, target, features):
        
        pandas_dtypes = {
            "float64": "float",
            "int64": "integer",
            "bool": "boolean",
            "double": "double",
            "object": "string",
            "binary": "binary",
            }
        
        input_schema = Schema(
            [
                ColSpec(
                    pandas_dtypes[str(data[features].dtypes[element])], element
                )
                for element in data[features].columns
            ]
        )
        output_schema = Schema(
            [ColSpec(pandas_dtypes[str(data[target].dtypes)])]
        )
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        return signature



    def mlflow_training(self, features_train, target_train, features_test, target_test, signature, data_minmax_dict, feature_minmax_dict, target_minmax_dict, feature_dtypes_dict, model_name, model_typ="linear_regression"):
        
        
        if model_name is not None:
            model_name = model_name
        else:
            model_name = self.model_name


        sk_model = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(),
            "decision_tree": DecisionTreeRegressor(),
            "k_neighbors": KNeighborsRegressor(),
            "mlp": MLPRegressor(),
            "ridge": Ridge(),
            "elastic_net": ElasticNet(),
            "ada_boost": AdaBoostRegressor()
            }
        
        sk_model = sk_model[model_typ]
        print(sk_model)
        

        
        mlflow.set_experiment(model_name)
        print(f"MLFLow is tracking: {mlflow.is_tracking_uri_set()}")
        
        
        with mlflow.start_run():
            
            sk_model.fit(features_train, target_train)
            
            train_score = round(sk_model.score(features_train, target_train), 4)
            test_score = round(sk_model.score(features_test, target_test), 4)

            mlflow.autolog()

            mlflow.log_params(sk_model.get_params())
            mlflow.log_metric("train_score", train_score)
            mlflow.log_metric("test_score", test_score)

            mlflow.sklearn.log_model(sk_model, "model", signature=signature)


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



    def make_model(self, data=None, target=None, features=None, test_size = 0.2, scaler_expand_by="std", model_name=None, model_parameter=None, model_typ="linear_regression"):
        


        
        try:
        
            target_and_features = [target] + features
            
            df_tf = data[target_and_features].copy()

            
            dft = self.descriptiontable(df_tf)
            
            

            new_minmaxscalingdf = self.make_minmaxscalingtable_by_descriptiontable(
                descriptiontable=dft, 
                expand_by=scaler_expand_by)


            # target_minmaxscaling = list(target_minmaxscaling)
            # target_minmax_list = np.array(target_minmax_list)
            # target_minmax_list = target_minmax_list.reshape(-1, 1)
            
            
            data_minmax_dict = self.create_data_minmax_dict(new_minmaxscalingdf)
            
            feature_minmax_dict = self.create_data_minmax_dict(new_minmaxscalingdf.loc[:, features])
            target_minmax_dict = self.create_data_minmax_dict(new_minmaxscalingdf.loc[:, target])
            
            
            feature_dtypes_dict = self.create_data_dtype_dict(new_minmaxscalingdf.loc[:,features])
            
            
            print("MinMaxScaler Features")
            feature_minmaxscaler = MinMaxScaler()
            feature_minmaxscaler.fit(new_minmaxscalingdf.loc[:, features])
            fitted_df_features = feature_minmaxscaler.transform(data.loc[:,features])

            # print(f"fitted_df_features: {fitted_df_features}")
            
            
            
            print("MinMaxScaler Target")
            target_minmaxscaler = MinMaxScaler()
            target_minmax_list = list(new_minmaxscalingdf.loc[:, target])
            target_minmax_list = np.array(target_minmax_list)
            target_minmax_list = target_minmax_list.reshape(-1, 1)
            
            
            target_minmaxscaler.fit(target_minmax_list)
            fitted_target = target_minmaxscaler.transform(data.loc[:,[target]])
            
            # print(f"fitted_target: {fitted_target}")         


            print("Create Signature")
            signature = self.create_signature(data=data, target=target, features=features)



            print("Train Test Split")
            random_state = 2023 
            # test_size =0.2

            (
                features_train,
                features_test,
                target_train,
                target_test,
            ) = train_test_split(
                fitted_df_features,
                fitted_target,
                test_size=test_size,
                random_state=random_state
                )




            print("Create Model")

            MLFlow_Experiment = model_name  # "Project_name"

            self.mlflow_training(
                features_train=features_train,
                target_train=target_train,
                features_test=features_test,
                target_test=target_test,
                signature=signature,
                data_minmax_dict=data_minmax_dict,
                feature_minmax_dict=feature_minmax_dict,
                target_minmax_dict=target_minmax_dict,
                feature_dtypes_dict=feature_dtypes_dict,
                model_name=MLFlow_Experiment,
                model_typ=model_typ
                )
            
            print("Model created!")


        except Exception as e:
            print(e)
            print("Model not created!")
    
        return "Done!"



