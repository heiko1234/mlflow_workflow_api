


import pandas as pd
import numpy as np







df = pd.DataFrame()

df["Yield"] = [44.0, 43.0, 46.0, 40.1, 42.2]
df["BioMaterial1"]=[5.5, 4.5, 3.5, 1.0, 6.0]
df["BioMaterial2"]=[9.5, 9, 5, 10, 12]
df["ProcessValue1"] = [20, 15, 10, 9, 2]



data_transformation = {"Yield": "no transformation", "BioMaterial1": "log", "BioMaterial2": "sqrt", "ProcessValue1": "1/x"}




class data_preprocessing():
    def __init__(self, df, transformation_dict):
        self.df = df
        self.transformation_dict = transformation_dict
        
        
        
    def transform_rawdata(self):
        for column in self.df.columns:
            transformation = self.transformation_dict[column]
            self.df[column] = self.transform_column(column=column, transformation=transformation)
            output = self.df
        return output
    
    def transform_column(self, column, transformation):
        
        if transformation == "no transformation":
            data = df[column]
        elif transformation == "log":
            data = df[column].apply(lambda x: np.log(x))
        elif transformation == "sqrt":
            data = df[column].apply(lambda x: np.sqrt(x))
        elif transformation == "1/x":
            data = df[column].apply(lambda x: 1/x)
        elif transformation == "x^2":
            data = df[column].apply(lambda x: x**2)
        elif transformation == "x^3":
            data = df[column].apply(lambda x: x**3)
        else:
            data = df[column]
            
        return data
    
    
    def make_data_minmax_dict(self):
        output = {}
        for column in self.df.columns:
            output[column] = {"min": self.df[column].min(), "max": self.df[column].max()}
        return output
    
    def make_data_dtypes_dict(self):
        output = {}
        for column in self.df.columns:
            output[column] = str(self.df[column].dtype)
        return output
    
    def descriptiontable(self):
        
        dft=self.df.describe().reset_index(drop = True).T
        dft = dft.reset_index(drop=False)
        dft.columns= ["description", "counts", "mean", "std", "min", "25%", "50%", "75%", "max"]
        dft["nan"]=self.df.isna().sum().values
        
        return dft


    def make_minmaxscalingtable_by_descriptiontable(self, expand_by=None):
        
        output_df = pd.DataFrame()
        
        descriptiontable = self.descriptiontable()
        
        if expand_by == None:
            
            for row_index in range(descriptiontable.shape[0]):
                output_df[descriptiontable.loc[row_index, "description"]] = [descriptiontable.loc[row_index, "max"], descriptiontable.loc[row_index, "min"]]

        elif expand_by == "std":
                
                for row_index in range(descriptiontable.shape[0]):
                    output_df[descriptiontable.loc[row_index, "description"]] = [descriptiontable.loc[row_index, "max"]+ descriptiontable.loc[row_index, "std"], descriptiontable.loc[row_index, "min"]- descriptiontable.loc[row_index, "std"]]

        return output_df



    def create_data_dtype_dict(self):
        
        pandas_dtypes = {
            "float64": "float",
            "int64": "integer",
            "bool": "boolean",
            "double": "double",
            "object": "string",
            "binary": "binary",
        }

        output = {}

        for element in self.df.columns:
            output[element] = pandas_dtypes[str(self.df.dtypes[element])]

        return output
    
    
    def create_feature_minmax_dict(self, features, expand_by=None):
        
        output = {}
        
        if expand_by == None:
            for feature in features:
                output[feature] = {"min": self.df[feature].min(), "max": self.df[feature].max()}
        elif expand_by == "std":
            for feature in features:
                output[feature] = {"min": self.df[feature].min()-self.df[feature].std(), "max": self.df[feature].max()+self.df[feature].std()}
        return output
    
    def create_target_minmax_dict(self, target, expand_by=None):
        
        output = {}
        
        if expand_by == None:
            output[target] = {"min": self.df[target].min(), "max": self.df[target].max()}
        elif expand_by == "std":
            output[target] = {"min": self.df[target].min()-self.df[target].std(), "max": self.df[target].max()+self.df[target].std()}
        return output
    
    def create_feature_dtype_dict(self, features, pandas_dtypes):
        output = {}

        for element in features:
            output[element] = pandas_dtypes[str(self.df.dtypes[element])]

        return output










