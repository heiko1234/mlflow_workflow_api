




from experimental.data_preprocess import data_preprocessing



import pandas as pd







df = pd.DataFrame()

df["Yield"] = [44.0, 43.0, 46.0, 40.1, 42.2]
df["BioMaterial1"]=[5.5, 4.5, 3.5, 1.0, 6.0]
df["BioMaterial2"]=[9.5, 9, 5, 10, 12]
df["ProcessValue1"] = [20, 15, 10, 9, 2]






data_transformation = {"Yield": "no transformation", "BioMaterial1": "log", "BioMaterial2": "sqrt", "ProcessValue1": "1/x"}




data_preprocessed= data_preprocessing(df=df, transformation_dict=data_transformation)



data_preprocessed.transform_rawdata()


data_preprocessed.make_data_dtypes_dict()

data_preprocessed.make_data_minmax_dict()


data_preprocessed.descriptiontable()

data_preprocessed.make_minmaxscalingtable_by_descriptiontable(expand_by="std")

data_preprocessed.make_minmaxscalingtable_by_descriptiontable()


data_preprocessed.create_spc_cleaning_table()



dspctable =data_preprocessed.transform_cleaning_table_in_dict(cleaning_table=data_preprocessed.create_spc_cleaning_table())
dspctable

updating_spc_cleaning_dict={"Yield": {"rule4": "remove data"}}


data_preprocessed.update_nested_dict(dspctable, updating_spc_cleaning_dict)

















