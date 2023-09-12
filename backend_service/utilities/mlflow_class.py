




import os
import json
import mlflow
import pandas as pd
from pathlib import Path


from pathlib import PurePosixPath
from mlflow import MlflowClient


from sklearn.preprocessing import MinMaxScaler


from dotenv import load_dotenv


load_dotenv()




class mlflow_model():
    
    def __init__(self, model_name, staging="Staging"):


        load_dotenv()

        self.model_name = model_name
        self.staging = staging


        self.azure_model_dir = "models:/"

        if self.staging == "Staging":
            self.artifact_path = str(PurePosixPath(self.azure_model_dir).joinpath(self.model_name, "Staging"))
        elif self.staging == "Production":
            self.artifact_path = str(PurePosixPath(self.azure_model_dir).joinpath(self.model_name, "Production"))
        else:
            print("staging must be either 'Staging' or 'Production'")
            raise ValueError

        self.model = mlflow.pyfunc.load_model(self.artifact_path)
        print(f"Model {model_name} loaded")


    def list_registered_models(self):
        client = MlflowClient()
        output = []
        for rm in client.search_registered_models():
            output.append(rm.name)
        return output

    def get_model_version(self):
        client = MlflowClient()
        model_version = client.get_latest_versions(self.model_name, stages=[self.staging])[0].version
        return model_version
    
    def get_model(self):
        return self.model
    
    def get_model_artifact(self, artifact="feature_dtypes.json"):
        
        """
        
        aritfact: feature_names.json, feature_types.json, feature_limits.json, target_limits.json

        Returns:
            dictionary: dictionary with feature names and their data types
        """


        path_to_file = self.artifact_path + "/" + artifact

        path_to_file = path_to_file

        output = mlflow.artifacts.load_dict(path_to_file)

        return output


    def decode_df_mlflow_dtype(self, data, dtype):
        
        mlflow_dtypes = {
            "float": "float32",
            "integer": "int32",
            "boolean": "bool",
            "double": "double",
            "string": "object",
            "binary": "binary",
        }
    
        dtype_dict = self.get_model_artifact(artifact="feature_dtypes.json")
        
        for element in list(dtype_dict.keys()):
            try: 
                data[element] = data[element].astype(mlflow_dtypes[dtype_dict[element]])
            except BaseException:
                pass
        return data

    def make_minmax_df(self, dict):
        df = pd.DataFrame()
        for element in list(dict.keys()):
            df[element] = [dict[element]["max"], dict[element]["min"]]
        return df


    def get_feature_minmaxscaler(self):
        """
        Returns:
            dictionary: dictionary with feature names and their minmaxscaler
        """
        path_to_file = self.artifact_path + "/feature_limits.json"
        
        limits_df = mlflow.artifacts.load_dict(path_to_file)
        
        limits_df = self.make_minmax_df(limits_df)
        
        feature_minmaxscaler = MinMaxScaler()
        
        feature_minmaxscaler.fit(limits_df)
        
        return feature_minmaxscaler


    def get_target_minmaxscaler(self):
        """
        Returns:
            dictionary: dictionary with feature names and their minmaxscaler
        """
        path_to_file = self.artifact_path + "/target_limits.json"
        
        limits_df = mlflow.artifacts.load_dict(path_to_file)
        
        limits_df = self.make_minmax_df(limits_df)
        
        target_minmaxscaler = MinMaxScaler()
        
        target_minmaxscaler.fit(limits_df)
        
        return target_minmaxscaler


    def get_features(self):
        """
        Returns:
            dictionary: dictionary with feature names and their data types
        """
        path_to_file = self.artifact_path + "/feature_limits.json"
        
        features = mlflow.artifacts.load_dict(path_to_file)
        
        features = list(features.keys())
        
        return features
    
    def validate_data_columns(self, data):
        
        features = self.get_features()
        
        data_columns = list(data.columns)
        
        elements_to_test = [list_element for list_element in features if list_element in data_columns]
        
        if features == elements_to_test:
            return True
        else:
            return False
    
    def make_predictions(self, data):
        
        features = self.get_features()
        feature_scaler = self.get_feature_minmaxscaler()
        target_scaler = self.get_target_minmaxscaler()
        feature_dtypes = self.get_model_artifact(artifact="feature_dtypes.json")
        
        print(f"features: {features}")
        
        try:
            if self.validate_data_columns(data):
                scale_data = data[features]
                feature_data_scaled = feature_scaler.transform(scale_data)
                feature_data_scaled_df = pd.DataFrame(feature_data_scaled, columns=features)
                feature_data_scaled_df = self.decode_df_mlflow_dtype(feature_data_scaled_df, dtype=feature_dtypes)
                
                df_predictions = self.model.predict(feature_data_scaled_df)
                
                output = target_scaler.inverse_transform(df_predictions)
                
                output = list(output.flatten())

                return output
            
            else:
                features = self.get_features()
                data_columns = list(data.columns)
                
                missing_features = list(set(features) - set(data_columns))
                
                print(f"Missing features: {missing_features}")
                
                raise ValueError

        
        except BaseException as e:
            print(e)
            return None





