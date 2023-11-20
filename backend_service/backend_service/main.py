



import os
import yaml
import numpy as np
import pandas as pd


from dotenv import load_dotenv
from datetime import datetime

from typing import Dict, List, Union, Set, Optional

from fastapi import FastAPI, Depends, Request

from pydantic import BaseModel, Field, validator


# from backend_service.utilities.mlflow_training_class import mlflow_training
# from backend_service.utilities.mlflow_predict_class import mlflow_model
# from backend_service.backend_service.utilities.data_preprocess import data_preprocessing

from backend_service.utilities.data_preprocess import data_preprocessing
from backend_service.utilities.mlflow_training_class import mlflow_training
from backend_service.utilities.mlflow_predict_class import mlflow_model, list_all_registered_models

from backend_service.utilities.plots import validation_plot


import polars as pl

from upath import UPath



from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient, BlobBlock



from dotenv import load_dotenv

load_dotenv()







def read_configuration(configuration_file_path):

    with open(configuration_file_path) as file:
        config = yaml.full_load(file)

    return config





load_dotenv()

local_run = os.getenv("LOCAL_RUN", False)



class Data_load(BaseModel):
    blobcontainer: str | None = Field(example="chemical-data")
    subcontainer: str | None = Field(example="chemical-data")
    file_name: str | None = Field(example="ChemicalManufacturingProcess.parquet")
    account: str | None = Field(example="devstoreaccount1")


class Data_load_series(BaseModel):
    blobcontainer: str | None = Field(example="chemical-data")
    subcontainer: str | None = Field(example="chemical-data")
    file_name: str | None = Field(example="ChemicalManufacturingProcess.parquet")
    account: str | None = Field(example="devstoreaccount1")
    column_name: str | None = Field(example="Yield")


class Data_load_selected_features(BaseModel):
    blobcontainer: str | None = Field(example="chemical-data")
    subcontainer: str | None = Field(example="chemical-data")
    file_name: str | None = Field(example="ChemicalManufacturingProcess.parquet")
    account: str | None = Field(example="devstoreaccount1")
    features: List[str] | None = Field(example=["BioMaterial1", "BioMaterial2", "ProcessValue1"])


class Data_load_and_clean(BaseModel):
    blobcontainer: str | None = Field(example="chemical-data")
    subcontainer: str | None = Field(example="chemical-data")
    file_name: str | None = Field(example="ChemicalManufacturingProcess.parquet")
    account: str | None = Field(example="devstoreaccount1")
    features: List[str] | None = Field(example=["BioMaterial1", "BioMaterial2", "ProcessValue1"])
    spc_cleaning_dict: Dict[str, Dict[str, str]] | None = Field(example={"BioMaterial1": {"rule1": "no cleaning"}, "BioMaterial2": {"rule1":"remove data"}})
    limits_dict: Dict[str, Dict[str, Union[str, int, float]]] | None = Field(example={"BioMaterial1": {"min": 10, "max": 20}})
    transformation_dict: Dict[str, str] | None = Field(example={"Yield": "no transformation", "BioMaterial1": "log", "BioMaterial2": "sqrt", "ProcessValue1": "1/x"})



class train_modeling(BaseModel):
    blobcontainer: str | None = Field(example="chemical-data")
    subcontainer: str | None = Field(example="chemical-data")
    file_name: str | None = Field(example="ChemicalManufacturingProcess.parquet")
    account: str | None = Field(example="devstoreaccount1")
    features: List[str] | None = Field(example=["BioMaterial1", "BioMaterial2", "ProcessValue1"])
    spc_cleaning_dict: Dict[str, Dict[str, str]] | None = Field(example={"BioMaterial1": {"rule1": "no cleaning"}, "BioMaterial2": {"rule1":"remove data"}})
    limits_dict: Dict[str, Dict[str, Union[str, int, float]]] | None = Field(example={"BioMaterial1": {"min": 10, "max": 20}})
    transformation_dict: Dict[str, str] | None = Field(example={"Yield": "no transformation", "BioMaterial1": "log", "BioMaterial2": "sqrt", "ProcessValue1": "1/x"})
    target: str| None = Field(example="Yield")
    use_model_features: List[str] | None = Field(example=["BioMaterial1", "BioMaterial2", "ProcessValue1"])
    test_size: float | None = Field(example="0.2")
    scaler_expand_by: str | None = Field(example="std")
    use_model_name: str | None = Field(example="my_model_name")
    use_model_parameter: Dict[str, Union[str, int, float]] | None = Field(example={"alpha": 0.5})
    use_model_typ: str | None = Field(example="linear_regression")


class make_prediction(BaseModel):
    blobcontainer: str | None = Field(example="chemical-data")
    subcontainer: str | None = Field(example="chemical-data")
    file_name: str | None = Field(example="ChemicalManufacturingProcess.parquet")
    account: str | None = Field(example="devstoreaccount1")
    use_model_name: str | None = Field(example="my_model_name")



class make_prediction_with_data(BaseModel):
    account: str | None = Field(example="devstoreaccount1")
    use_model_name: str | None = Field(example="my_model_name")
    data_dict: List[Dict[str, Union[str, int, float]]] | Dict[str, Union[str, int, float]] | None = Field(example={"BioMaterial1": 10, "BioMaterial2": 20, "ProcessValue1": 30})




class model_artifact(BaseModel):
    account: str | None = Field(example="devstoreaccount1")
    use_model_name: str | None = Field(example="my_model_name")
    artifact: str | None = Field(example="target_limits.json")



class model_version(BaseModel):
    account: str | None = Field(example="devstoreaccount1")
    use_model_name: str | None = Field(example="my_model_name")
    staging: str | None = Field(example="staging or production")




app = FastAPI()


@app.get("/")
def read_root():
    return {"Connected to Backend Service"}


@app.get("/health")
def health():
    return True


@app.get("/testdata")
def testdata():
    return {"testdata": "testdata"}





@app.get("/list_available_accounts")
def get_available_blobs():

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:
        account = "devstoreaccount1"

    else:
        account = os.environ["AZURE_STORAGE_ACCOUNT"]

    return account


@app.get("/list_available_models")
def get_available_models():

    # local_run = os.getenv("LOCAL_RUN", False)
    # output = list_all_registered_models()

    my_mlflow_model = mlflow_model(model_name="project_name", staging="Staging")

    output = my_mlflow_model.list_registered_models()

    return output



@app.post("/list_available_blobs")
def get_available_blobs(query_input: Data_load):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:
        account = "devstoreaccount1"
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:
        # account = os.environ["AZURE_STORAGE_ACCOUNT"]
        account=query_input.account
        credential = DefaultAzureCredential(exclude_environment_credential=True)


    path = UPath("az://", account_name=account, anon=True, credential=credential)


    output = [str(p).split("/")[-1] for p in path.iterdir()]

    return output


@app.post("/list_available_subblobs")
def get_available_subblobs(query_input: Data_load):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:
        account = "devstoreaccount1"
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:
        # account = os.environ["AZURE_STORAGE_ACCOUNT"]
        account=query_input.account
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query_input.blobcontainer


    path = UPath(f"az://{blobcontainer}", account_name=account, anon=True, credential=credential)


    output = [str(p).split("/")[-1] for p in path.iterdir()]

    return output



@app.post("/list_available_files")
def get_available_files(query_input: Data_load):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:
        account = "devstoreaccount1"
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:
        # account = os.environ["AZURE_STORAGE_ACCOUNT"]
        account=query_input.account
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query_input.blobcontainer
    subcontainer=query_input.subcontainer


    path = UPath(f"az://{blobcontainer}/{subcontainer}", account_name=account, anon=True, credential=credential)


    output = [str(p).split("/")[-1] for p in path.iterdir()]

    return output





@app.post("/data_statistics")
def post_data_statistics(query_input: Data_load):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:

        # account = "devstoreaccount1"
        account=query_input.account
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:

        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query_input.blobcontainer
    subcontainer=query_input.subcontainer
    file=query_input.file_name

    if (blobcontainer is not None) and (subcontainer is not None) and (file is not None):

        print("blobcontainer, subcontainer, file for data statistics")
        master_data = pl.read_parquet(
            f"az://{blobcontainer}/{subcontainer}/{file}",
            storage_options={"account_name": account, "credential": credential}
            )
        df = master_data.to_pandas()

        dft=df.describe().reset_index(drop = True).T
        dft = dft.reset_index(drop=False)
        dft.columns= ["description", "counts", "mean", "std", "min", "25%", "50%", "75%", "max"]
        dft["nan"]=df.isna().sum().values


        digits = 2
        output_df=dft.round(digits).to_json(orient='split')

        # print("data statistics done")

    else:
        output_df = None

    return output_df




@app.post("/data_statistics_selected_features")
def post_data_statistics_selected_features(query_input: Data_load_selected_features):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:

        # account = "devstoreaccount1"
        account=query_input.account
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:

        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query_input.blobcontainer
    subcontainer=query_input.subcontainer
    file=query_input.file_name
    features = query_input.features

    if (blobcontainer is not None) and (subcontainer is not None) and (file is not None):

        print("blobcontainer, subcontainer, file for data statistics")
        master_data = pl.read_parquet(
            f"az://{blobcontainer}/{subcontainer}/{file}",
            storage_options={"account_name": account, "credential": credential}
            )
        df = master_data.to_pandas()

        df = df.loc[:,features]

        dft=df.describe().reset_index(drop = True).T
        dft = dft.reset_index(drop=False)
        dft.columns= ["description", "counts", "mean", "std", "min", "25%", "50%", "75%", "max"]
        dft["nan"]=df.isna().sum().values


        digits = 2
        output_df=dft.round(digits).to_json(orient='split')

        # print("data statistics done")

    else:
        output_df = None

    return output_df





@app.post("/data_columns")
def post_data_columns(query_input: Data_load):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:

        # account = "devstoreaccount1"
        account=query_input.account
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:

        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query_input.blobcontainer
    subcontainer=query_input.subcontainer
    file=query_input.file_name

    if (blobcontainer is not None) and (subcontainer is not None) and (file is not None):

        print("blobcontainer, subcontainer, file for data statistics")
        master_data = pl.read_parquet(
            f"az://{blobcontainer}/{subcontainer}/{file}",
            storage_options={"account_name": account, "credential": credential}
            )
        df = master_data.to_pandas()

        df_cnames = list(df.columns)

    else:
        df_cnames = None

    return df_cnames




@app.post("/data_series")
def post_data_series(query_input: Data_load_series):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:

        # account = "devstoreaccount1"
        account=query_input.account
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:

        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query_input.blobcontainer
    subcontainer=query_input.subcontainer
    file=query_input.file_name

    if (blobcontainer is not None) and (subcontainer is not None) and (file is not None):

        print("blobcontainer, subcontainer, file for data statistics")
        master_data = pl.read_parquet(
            f"az://{blobcontainer}/{subcontainer}/{file}",
            storage_options={"account_name": account, "credential": credential}
            )
        df = master_data.to_pandas()

        column_name=query_input.column_name

        df_output = df.loc[:,column_name]

    else:
        df_output = None

    return df_output




@app.post("/data_target_correlation")
def post_data_statistics(query_input: Data_load_series):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:

        # account = "devstoreaccount1"
        account=query_input.account
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:

        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query_input.blobcontainer
    subcontainer=query_input.subcontainer
    file=query_input.file_name
    column_name=query_input.column_name

    if (blobcontainer is not None) and (subcontainer is not None) and (file is not None):

        print("blobcontainer, subcontainer, file for data statistics")
        master_data = pl.read_parquet(
            f"az://{blobcontainer}/{subcontainer}/{file}",
            storage_options={"account_name": account, "credential": credential}
            )
        df = master_data.to_pandas()

        name_of_target = column_name

        correlations = []
        for i in df.columns:
            if i != name_of_target:
                corr = np.corrcoef(df[name_of_target], df[i])[0, 1]
                correlations.append(corr)
            else:
                correlations.append(1)

        correlations = [round(i, 4) for i in correlations]

        return correlations

    else:
        return None




@app.post("/all_data")
def post_data_load_and_clean(query_input: Data_load):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:

        # account = "devstoreaccount1"
        account=query_input.account
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:

        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query_input.blobcontainer
    subcontainer=query_input.subcontainer
    file=query_input.file_name


    if (blobcontainer is not None) and (subcontainer is not None) and (file is not None):

        print("blobcontainer, subcontainer, file for data statistics")
        master_data = pl.read_parquet(
            f"az://{blobcontainer}/{subcontainer}/{file}",
            storage_options={"account_name": account, "credential": credential}
            )
        df = master_data.to_pandas()

        output_df=df.to_json(orient='split')

        return output_df

    else:
        return None



@app.post("/data_selected_features")
def post_data_selected_features(query_input: Data_load_selected_features):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:

        # account = "devstoreaccount1"
        account=query_input.account
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:

        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query_input.blobcontainer
    subcontainer=query_input.subcontainer
    file=query_input.file_name

    # print(blobcontainer, subcontainer, file)


    if (blobcontainer is not None) and (subcontainer is not None) and (file is not None):

        # print("blobcontainer, subcontainer, file for data statistics")
        master_data = pl.read_parquet(
            f"az://{blobcontainer}/{subcontainer}/{file}",
            storage_options={"account_name": account, "credential": credential}
            )
        df = master_data.to_pandas()


        features = query_input.features

        df = df.loc[:,features]

        output_df=df.to_json(orient='split')


    else:
        output_df = None

    return output_df




@app.post("/data_load_and_clean")
def post_data_load_and_clean(query_input: Data_load_and_clean):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:

        # account = "devstoreaccount1"
        account=query_input.account
        credential = os.getenv("AZURE_STORAGE_KEY")


    else:

        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query_input.blobcontainer
    subcontainer=query_input.subcontainer
    file=query_input.file_name
    features=query_input.features
    spc_cleaning_dict=query_input.spc_cleaning_dict
    limits_dict=query_input.limits_dict
    transformation_dict=query_input.transformation_dict


    if (blobcontainer is not None) and (subcontainer is not None) and (file is not None):

        print("blobcontainer, subcontainer, file for data statistics")
        master_data = pl.read_parquet(
            f"az://{blobcontainer}/{subcontainer}/{file}",
            storage_options={"account_name": account, "credential": credential}
            )
        df = master_data.to_pandas()

        data_in_preprocessing = data_preprocessing(df = df, transformation_dict=transformation_dict)

        data_preprocessed = data_in_preprocessing.clean_up_data(dataframe=df, features=features, spc_cleaning_dict=spc_cleaning_dict, limits_dict=limits_dict)

        data_preprocessed = data_preprocessing(df = data_preprocessed, transformation_dict=transformation_dict).transform_rawdata()

        data_preprocessed = data_preprocessed.reset_index(drop=True)

        output_df=data_preprocessed.to_json(orient='split')



        return output_df

    else:
        return None




# class train_modeling(BaseModel):
#     blobcontainer: str | None = Field(example="chemical-data")
#     subcontainer: str | None = Field(example="chemical-data")
#     file_name: str | None = Field(example="ChemicalManufacturingProcess.parquet")
#     account: str | None = Field(example="devstoreaccount1")
#     features: List[str] | None = Field(example=["BioMaterial1", "BioMaterial2", "ProcessValue1"])
#     spc_cleaning_dict: Dict[str, Dict[str, str]] | None = Field(example={"BioMaterial1": {"rule1": "no cleaning"}, "BioMaterial2": {"rule1":"remove data"}})
#     limits_dict: Dict[str, Dict[str, Union[str, int, float]]] | None = Field(example={"BioMaterial1": {"min": 10, "max": 20}})
#     transformation_dict: Dict[str, str] | None = Field(example={"Yield": "no transformation", "BioMaterial1": "log", "BioMaterial2": "sqrt", "ProcessValue1": "1/x"})
#     target: str| None = Field(example="Yield")
#     features: List[str] | None = Field(example=["BioMaterial1", "BioMaterial2", "ProcessValue1"])
#     test_size: float | None = Field(example="0.2")
#     scaler_expand_by: str | None = Field(example="std")
#     model_name: str | None = Field(example="my_model_name")
#     model_parameter: Dict[str, Union[str, int, float]] | None = Field(example={"alpha": 0.5})
#     model_typ: str | None = Field(example="linear_regression")



@app.post("/train_model")
def train_model(query_input: train_modeling):


    try:
        local_run = os.getenv("LOCAL_RUN", False)

        if local_run:
            # account = "devstoreaccount1"
            account=query_input.account
            credential = os.getenv("AZURE_STORAGE_KEY")

        else:
            account = os.environ["AZURE_STORAGE_ACCOUNT"]
            # credential = os.environ["AZURE_STORAGE_KEY"]
            credential = DefaultAzureCredential(exclude_environment_credential=True)

        # load_data
        blobcontainer=query_input.blobcontainer
        subcontainer=query_input.subcontainer
        file=query_input.file_name

        # preprocess the data
        features=query_input.features
        spc_cleaning_dict=query_input.spc_cleaning_dict
        limits_dict=query_input.limits_dict
        transformation_dict=query_input.transformation_dict

        # make the model
        target=query_input.target
        test_size=query_input.test_size
        use_model_features=query_input.use_model_features
        scaler_expand_by=query_input.scaler_expand_by
        use_model_name=query_input.use_model_name
        use_model_parameter=query_input.use_model_parameter
        use_model_typ=query_input.use_model_typ



        if (blobcontainer is not None) and (subcontainer is not None) and (file is not None):

            print("blobcontainer, subcontainer, file for data statistics")
            master_data = pl.read_parquet(
                f"az://{blobcontainer}/{subcontainer}/{file}",
                storage_options={"account_name": account, "credential": credential}
                )
            df = master_data.to_pandas()

            data_in_preprocessing = data_preprocessing(df = df, transformation_dict=transformation_dict)

            data_preprocessed = data_in_preprocessing.clean_up_data(dataframe=df, features=features, spc_cleaning_dict=spc_cleaning_dict, limits_dict=limits_dict)

            data_preprocessed = data_preprocessing(df = data_preprocessed, transformation_dict=transformation_dict).transform_rawdata()

            data_preprocessed = data_preprocessed.reset_index(drop=True)


            mlflow_training(model_name=use_model_name).make_model(
                data=data_preprocessed,
                target=target,
                features=use_model_features,
                test_size = test_size,
                scaler_expand_by=scaler_expand_by,
                transformation_dict=transformation_dict,
                model_name=use_model_name,
                model_parameter=use_model_parameter,
                model_typ=use_model_typ
                )

            return "Done"


    except Exception as e:
        print(e)
        return "Failed"



@app.post("/model_validation")
def validation_model(query: make_prediction):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:
        # account = "devstoreaccount1"
        account=query.account
        credential = os.getenv("AZURE_STORAGE_KEY")

    else:
        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query.blobcontainer
    subcontainer=query.subcontainer
    file=query.file_name

    use_model_name = query.use_model_name

    if (blobcontainer is not None) and (subcontainer is not None) and (file is not None):

        print("blobcontainer, subcontainer, file for data statistics")
        master_data = pl.read_parquet(
            f"az://{blobcontainer}/{subcontainer}/{file}",
            storage_options={"account_name": account, "credential": credential}
            )
        df = master_data.to_pandas()

        loaded_model = mlflow_model(model_name=use_model_name, staging="Staging")

        output = loaded_model.make_predictions(data=df)
        target_name = list(loaded_model.get_model_artifact(artifact="target_limits.json").keys())[0]

        df_output = pd.DataFrame()

        target_name_output = "prediction"

        df_output[target_name_output] = output
        df_output["actual"] = df[target_name]

        # print(f"validation_model: {df_output.head()}")

        output_df = df_output.to_json(orient='split')

        return output_df

    else:
        return None


# TODO: make this work, Figure not Hashable
# @app.post("/model_validation_graphics")
# def validation_model_graphics(query: make_prediction):

#     local_run = os.getenv("LOCAL_RUN", False)

#     if local_run:
#         # account = "devstoreaccount1"
#         account=query.account
#         credential = os.getenv("AZURE_STORAGE_KEY")

#     else:
#         account = os.environ["AZURE_STORAGE_ACCOUNT"]
#         # credential = os.environ["AZURE_STORAGE_KEY"]
#         credential = DefaultAzureCredential(exclude_environment_credential=True)

#     blobcontainer=query.blobcontainer
#     subcontainer=query.subcontainer
#     file=query.file_name

#     use_model_name = query.use_model_name

#     if (blobcontainer is not None) and (subcontainer is not None) and (file is not None):

#         print("blobcontainer, subcontainer, file for data statistics")
#         master_data = pl.read_parquet(
#             f"az://{blobcontainer}/{subcontainer}/{file}",
#             storage_options={"account_name": account, "credential": credential}
#             )
#         df = master_data.to_pandas()

#         loaded_model = mlflow_model(model_name=use_model_name, staging="Staging")

#         output = loaded_model.make_predictions(data=df)
#         target_name = list(loaded_model.get_model_artifact(artifact="target_limits.json").keys())[0]

#         df_output = pd.DataFrame()

#         target_name_output = "prediction"

#         df_output[target_name_output] = output
#         df_output["actual"] = df[target_name]

#         print(f"validation_model: {df_output.head()}")


#         fig = validation_plot(df_output["actual"], df_output["prediction"])

#         print(f"validation_model: {fig}")

#         return {fig: fig}

#     else:
#         return None




@app.post("/model_prediction")
def predict_model(query: make_prediction):

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:
        # account = "devstoreaccount1"
        account=query.account
        credential = os.getenv("AZURE_STORAGE_KEY")

    else:
        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    blobcontainer=query.blobcontainer
    subcontainer=query.subcontainer
    file=query.file_name

    use_model_name = query.use_model_name

    if (blobcontainer is not None) and (subcontainer is not None) and (file is not None):

        print("blobcontainer, subcontainer, file for data statistics")
        master_data = pl.read_parquet(
            f"az://{blobcontainer}/{subcontainer}/{file}",
            storage_options={"account_name": account, "credential": credential}
            )
        df = master_data.to_pandas()

        loaded_model = mlflow_model(model_name=use_model_name, staging="Staging")

        output = loaded_model.make_predictions(data=df)
        target_name = list(loaded_model.get_model_artifact(artifact="target_limits.json").keys())[0]

        df_output = pd.DataFrame()

        target_name_output = target_name

        df_output[target_name_output] = output

        output = df_output.to_json(orient='split')

        return output

    else:
        return None



# TODO: work on this callback, not working actually
@app.post("/model_prediction_send_data")
def predict_model(query: make_prediction_with_data):

    local_run = os.getenv("LOCAL_RUN", False)

    print("### local_run ###")

    if local_run:
        # account = "devstoreaccount1"
        account=query.account
        credential = os.getenv("AZURE_STORAGE_KEY")

    else:
        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    if query.data_dict is not None:
        master_data=query.data_dict

        print(master_data)
        if isinstance(master_data, dict):
            df = pd.DataFrame.from_dict([master_data], orient="columns").reset_index()
        elif isinstance(master_data, list):
            df = pd.DataFrame.from_dict(master_data, orient="columns").reset_index()

        print(df.head())

        use_model_name = query.use_model_name
        loaded_model = mlflow_model(model_name=use_model_name, staging="Staging")

        output = loaded_model.make_predictions(data=df)
        target_name = list(loaded_model.get_model_artifact(artifact="target_limits.json").keys())[0]

        df_output = pd.DataFrame()

        target_name_output = target_name

        df_output[target_name_output] = output

        output = df_output.to_json(orient='split')

        # output = "200"


        return output

    else:
        return None



@app.post("/get_model_artifact")
def get_model_artifact(query: model_artifact):


    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:
        # account = "devstoreaccount1"
        account=query.account
        credential = os.getenv("AZURE_STORAGE_KEY")

    else:
        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    use_model_name = query.use_model_name
    artifact = query.artifact


    loaded_model = mlflow_model(model_name=use_model_name, staging="Staging")



    output = loaded_model.get_model_artifact(artifact=artifact)


    return output


@app.post("/get_model_version")
def get_model_version(query: model_version):


    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:
        # account = "devstoreaccount1"
        account=query.account
        credential = os.getenv("AZURE_STORAGE_KEY")

    else:
        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        # credential = os.environ["AZURE_STORAGE_KEY"]
        credential = DefaultAzureCredential(exclude_environment_credential=True)

    use_model_name = query.use_model_name
    staging = query.staging


    loaded_model = mlflow_model(model_name=use_model_name, staging=staging)


    output = loaded_model.get_model_version()

    print(f"get_model_version: {output}")


    return output









