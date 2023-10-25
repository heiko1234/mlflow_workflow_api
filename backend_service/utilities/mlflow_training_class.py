



import pandas as pd

import os
import mlflow
import collections
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

    def transform_rawdata(self, data, transformation_dict=None):

        output = data.copy()

        if transformation_dict == None:
            print("No transformation dict given!")
        for column in output.columns:
            try:
                transformation = transformation_dict[column]
                output[column] = self.transform_column(data=data, column=column, transformation=transformation)
            except Exception as e:
                print(e)
                print(f"Column {column} not transformed! {column} is not in transformation_dict")
                pass

        return output

    def transform_column(self, data, column, transformation):

        if transformation == "no transformation":
            data_series = data[column]
        elif transformation == "log":
            data_series = data[column].apply(lambda x: np.log(x))
        elif transformation == "sqrt":
            data_series = data[column].apply(lambda x: np.sqrt(x))
        elif transformation == "1/x":
            data_series = data[column].apply(lambda x: 1/x)
        elif transformation == "x^2":
            data_series = data[column].apply(lambda x: x**2)
        elif transformation == "x^3":
            data_series = data[column].apply(lambda x: x**3)
        else:
            data_series = data[column]

        return data_series


    def update_nested_dict(self, original_dict, overwrite_dict):
        """This function updates a nested dictionary

        Args:
            original_dict (dict): any dictionary
            overwrite_dict (dict): any subset of the original_dict, that will overwrite the original_dict
        Returns:
            _type_: returns the original_dict updated with the overwrite_dict
        """


        for k, v in overwrite_dict.items():
            if isinstance(v, collections.abc.Mapping):
                original_dict[k] = self.update_nested_dict(original_dict.get(k, {}), v)
            else:
                original_dict[k] = v

        return original_dict


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



    def mlflow_training(self, features_train, target_train, features_test, target_test, signature, data_minmax_dict, feature_minmax_dict_original, feature_minmax_dict, target_minmax_dict, feature_dtypes_dict, model_name, additional_model_dict=None, transformation_dict=None, model_typ="linear_regression"):


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


            # new

            df_predictions = sk_model.predict(features_test)
            predicted_values = list(df_predictions.flatten())
            target_test_values = list(target_test.flatten())

            rmse = np.sqrt(mean_squared_error(target_test_values, predicted_values))
            mae = mean_absolute_error(target_test_values, predicted_values)
            r2 = r2_score(target_test_values, predicted_values)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # tags
            mlflow.set_tag("model_typ", model_typ)
            mlflow.set_tag("target", list(target_minmax_dict.keys())[0])
            mlflow.set_tag("features", list(feature_minmax_dict.keys()))

            metric_log_dict = {
                "r2_training": train_score,
                "r2_test": test_score,
                "rmse": rmse,
                "mae": mae
                }


            # artifacts
            mlflow.sklearn.log_model(
                sk_model=sk_model,
                artifact_path="model",
                signature=signature,
                # https://mlflow.org/docs/latest/model-registry.html#adding-an-mlflow-model-to-the-model-registry
                registered_model_name=model_name   # direct registration of model
                )

            mlflow.log_dict(
                metric_log_dict, "model/metrics.json"
            )

            mlflow.log_dict(
                data_minmax_dict, "model/data_limits.json"
            )
            mlflow.log_dict(
                feature_minmax_dict, "model/feature_limits.json"
            )
            mlflow.log_dict(
                feature_minmax_dict_original, "model/feature_limits_unscaled.json"
            )
            mlflow.log_dict(
                target_minmax_dict, "model/target_limits.json"
            )
            mlflow.log_dict(
                feature_dtypes_dict, "model/feature_dtypes.json"
            )

            mlflow.log_dict(
                transformation_dict, "model/transformation_dict.json"
            )
            if additional_model_dict is not None:
                mlflow.log_dict(
                    additional_model_dict, "model/additional_model_dict.json"
                )

        mlflow.end_run()


    def transition_model_verstion_stage(self, model_name, version, stage):

        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=version, stage=stage
        )

        return "Done, transition of model version {version} to stage {stage} completed!".format(version=version, stage=stage)


    def transition_model_stage_staging_to_production(self, model_name):

        client = MlflowClient()
        model_version_none = client.get_latest_versions(name=model_name, stages=["Staging"])[0].version
        self.transition_model_verstion_stage(model_name=model_name, version=model_version_none, stage="Production")


    def transition_model_stage_none_to_staging(self, model_name):

        client = MlflowClient()
        model_version_none = client.get_latest_versions(name=model_name, stages=["None"])[0].version
        self.transition_model_verstion_stage(model_name=model_name, version=model_version_none, stage="Staging")



    def make_model(self, data=None, target=None, features=None, test_size = 0.2, scaler_expand_by="std", transformation_dict=None, model_name=None, additional_model_dict= None, model_parameter=None, model_typ="linear_regression"):



        try:

            target_and_features = [target] + features

            df_tf = data[target_and_features].copy()


            df_tf_transformed = self.transform_rawdata(data=df_tf, transformation_dict=transformation_dict)
            dft = self.descriptiontable(df_tf_transformed)

            dft_original = self.descriptiontable(df_tf)



            new_minmaxscalingdf_scaled = self.make_minmaxscalingtable_by_descriptiontable(
                descriptiontable=dft,
                expand_by=scaler_expand_by)


            new_minmaxscalingdf_original = self.make_minmaxscalingtable_by_descriptiontable(
                descriptiontable=dft_original,
                expand_by=None)


            # target_minmaxscaling = list(target_minmaxscaling)
            # target_minmax_list = np.array(target_minmax_list)
            # target_minmax_list = target_minmax_list.reshape(-1, 1)


            # scaled
            data_minmax_dict_scaled = self.create_data_minmax_dict(new_minmaxscalingdf_scaled)
            feature_minmax_dict_scaled = self.create_data_minmax_dict(new_minmaxscalingdf_scaled.loc[:, features])
            target_minmax_dict_scaled = self.create_data_minmax_dict(new_minmaxscalingdf_scaled.loc[:, target])

            # original
            feature_minmax_dict_original = self.create_data_minmax_dict(new_minmaxscalingdf_original.loc[:, features])


            feature_dtypes_dict = self.create_data_dtype_dict(new_minmaxscalingdf_scaled.loc[:,features])



            print("MinMaxScaler Features")
            feature_minmaxscaler = MinMaxScaler()
            feature_minmaxscaler.fit(new_minmaxscalingdf_scaled.loc[:, features])
            fitted_df_features = feature_minmaxscaler.transform(df_tf_transformed.loc[:,features])

            # print(f"fitted_df_features: {fitted_df_features}")


            print("MinMaxScaler Target")
            target_minmaxscaler = MinMaxScaler()
            target_minmax_list = list(new_minmaxscalingdf_scaled.loc[:, target])
            target_minmax_list = np.array(target_minmax_list)
            target_minmax_list = target_minmax_list.reshape(-1, 1)


            target_minmaxscaler.fit(target_minmax_list)
            fitted_target = target_minmaxscaler.transform(df_tf_transformed.loc[:,[target]])

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
                data_minmax_dict=data_minmax_dict_scaled,
                feature_minmax_dict_original=feature_minmax_dict_original,
                feature_minmax_dict=feature_minmax_dict_scaled,
                target_minmax_dict=target_minmax_dict_scaled,
                feature_dtypes_dict=feature_dtypes_dict,
                transformation_dict=transformation_dict,
                model_name=MLFlow_Experiment,
                model_typ=model_typ
                )

            print("Model created!")

            self.transition_model_stage_none_to_staging(model_name=model_name)

            print("Model transitioned to staging!")


        except Exception as e:
            print(e)
            print("Model not created!")

        return "Done!"





