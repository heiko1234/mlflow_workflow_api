






import pandas as pd


from experimental.mlflow_training_class import mlflow_training





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









