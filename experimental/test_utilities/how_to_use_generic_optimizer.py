





import pandas as pd


# /home/heiko/Schreibtisch/Repos/dash_apps/mlflow_workflow_app/experimental/mflow_workflow/how_to_use_mlflowclass.py

from backend_service.backend_service.utilities.mlflow_predict_class import mlflow_model

from backend_service.backend_service.utilities.generic_optimizer import genetic_algorithm
from backend_service.backend_service.utilities.generic_optimizer import makedf, lossfunction




my_mlflow_model = mlflow_model(model_name="project_name", staging="Staging")



my_mlflow_model.list_registered_models()


my_mlflow_model.get_model_artifact(artifact="feature_dtypes.json")

my_mlflow_model.get_model_artifact(artifact="feature_limits.json")

my_mlflow_model.get_model_artifact(artifact="target_limits.json")





# >>> my_mlflow_model.get_model_artifact(artifact="feature_dtypes.json")
# {'BiologicalMaterial02': 'float', 'ManufacturingProcess06': 'float'}
# >>> my_mlflow_model.get_model_artifact(artifact="feature_limits.json")
# {'BiologicalMaterial02': {'min': 47.577262843692466, 'max': 68.45273715630753}, 'ManufacturingProcess06': {'min': 199.99953651551115, 'max': 230.40046348448885}}
# >>> my_mlflow_model.get_model_artifact(artifact="target_limits.json")
# {'Yield': {'min': 33.2392790595385, 'max': 48.35072094046151}}




bounds_dict = my_mlflow_model.get_model_artifact(artifact="feature_limits.json")


bounds_dict


bounds = [[bounds_dict[element]["min"], bounds_dict[element]["max"]] for element in bounds_dict.keys()]

bounds
# [[47.577262843692466, 68.45273715630753], [199.99953651551115, 230.40046348448885]]


dtype_dict = my_mlflow_model.get_model_artifact(artifact="feature_dtypes.json")
dtype_dict



target = 45

model = my_mlflow_model



model.get_features()
model.get_model_artifact(artifact="feature_dtypes.json")


# def makedf(liste, model):

#     model_features = model.get_features()
#     data = pd.DataFrame(data= liste)
#     data = data.T
#     data.columns = [element for element in model_features]
#     return data



# def lossfunction(target, X, model):
#     idata = makedf(liste=X, model = model)
#     modeloutput = model.make_predictions(idata)
#     diff = abs((target - modeloutput[0])**2)
#     return diff


target = 43
# makedf(liste=[55, 210], model = model)

# model.make_predictions(makedf(liste=[55, 210], model = model))

# lossfunction(target=target, X=[55, 210], model=model)




# >>> my_mlflow_model.get_model_artifact(artifact="feature_limits.json")
# {'BiologicalMaterial02': {'min': 47.577262843692466, 'max': 68.45273715630753}, 'ManufacturingProcess06': {'min': 199.99953651551115, 'max': 230.40046348448885}}


genetic_algorithm(
    target=43,
    bounds=[[49, 68], [200, 230]],
    model=my_mlflow_model,
    break_accuracy=0.005,
    digits=5,
    n_bits=16,
    n_iter=100,
    n_pop=100,
    r_cross=0.9,
)








