


import pandas as pd
import numpy as np


import collections


from backend_service.utilities.nelson import (
    rule1,
    rule2,
    rule3,
    rule4,
    rule5,
    rule6,
    rule7,
    rule8,
)



# df = pd.DataFrame()

# df["Yield"] = [44.0, 43.0, 46.0, 40.1, 42.2]
# df["BioMaterial1"]=[5.5, 4.5, 3.5, 1.0, 6.0]
# df["BioMaterial2"]=[9.5, 9, 5, 10, 12]
# df["ProcessValue1"] = [20, 15, 10, 9, 2]



# data_transformation = {"Yield": "no transformation", "BioMaterial1": "log", "BioMaterial2": "sqrt", "ProcessValue1": "1/x"}




class data_preprocessing():
    def __init__(self, df, transformation_dict=None):
        self.df = df
        self.transformation_dict = transformation_dict


    def transform_rawdata(self, transformation_dict=None):


        if transformation_dict != None:
            self.transformation_dict = transformation_dict
        for column in self.df.columns:
            try:
                transformation = self.transformation_dict[column]
            except Exception as e:
                print(e)
                transformation = "no transformation"

            self.df[column] = self.transform_column(column=column, transformation=transformation)
            output = self.df
        return output


    def transform_column(self, column, transformation):

        if transformation == "no transformation":
            data = self.df[column]
        elif transformation == "log":
            data = self.df[column].apply(lambda x: np.log(x))
        elif transformation == "sqrt":
            data = self.df[column].apply(lambda x: np.sqrt(x))
        elif transformation == "1/x":
            data = self.df[column].apply(lambda x: 1/x)
        elif transformation == "x^2":
            data = self.df[column].apply(lambda x: x**2)
        elif transformation == "x^3":
            data = self.df[column].apply(lambda x: x**3)
        else:
            data = self.df[column]

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


    def create_spc_cleaning_table(self):

        dft=self.df.describe().reset_index(drop = True).T
        dft = dft.reset_index(drop=False)
        dft.columns= ["description", "counts", "mean", "std", "min", "25%", "50%", "75%", "max"]
        dft["nan"]=self.df.isna().sum().values

        dft = dft.drop(["25%", "50%", "75%"], axis=1)

        dft["rule1"] = "no cleaning"
        dft["rule2"] = "no cleaning"
        dft["rule3"] = "no cleaning"
        dft["rule4"] = "no cleaning"
        dft["rule5"] = "no cleaning"
        dft["rule6"] = "no cleaning"
        dft["rule7"] = "no cleaning"
        dft["rule8"] = "no cleaning"

        return dft


    def transform_cleaning_table_in_dict(self, cleaning_table):
        """ This function gives a dictionary of the usage of the nelson rules for each feature in the data cleaning table

        Args:
            dataframe (_type_): pandas dataframe

        Returns:
            _type_: returns a dictionary with separated rules for each feature
        """

        dict = {}

        list_of_rules = ["rule1", "rule2", "rule3", "rule4", "rule5", "rule6", "rule7", "rule8"]

        for element_in_description in cleaning_table["description"].unique():
            dict[element_in_description] = {}

            for rule in list_of_rules:
                if rule  in cleaning_table.columns:
                    dict[element_in_description][rule] = cleaning_table.loc[cleaning_table["description"]==element_in_description][rule].values[0]

        return dict


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



    def use_spc_cleaning_dict(self, dataframe, spc_cleaning_dict):
        """This function uses the dictionary of the usage of the nelson rules for each feature in the data cleaning table to clean the dataframe

        Args:
            dataframe: pandas dataframe
            spc_cleaning_dict: dictionary with separated rules for each feature

        Returns:
            pandas dataframe: returns a dataframe without the rows that have been cleaned
        """

        list_all_indexes = []

        for any_column in spc_cleaning_dict.keys():
            try:
                if any_column in dataframe.columns:
                    for any_rule in spc_cleaning_dict[any_column].keys():
                        if spc_cleaning_dict[any_column][any_rule] != "no cleaning":
                            try:

                                # print(f"any_column: {any_column}")
                                # print(f"any_rule: {any_rule}")
                                # print(dataframe.loc[eval(any_rule+"(original=dataframe[any_column])"), any_column])
                                # list_all_indexes.append(dataframe.loc[eval(any_rule+"(original=dataframe[any_column])"), any_column].index)
                                index_list_rule = dataframe.loc[eval(any_rule+"(original=dataframe[any_column])"), any_column].index
                                if len(index_list_rule) > 0:
                                    list_all_indexes.extend(index_list_rule)

                            except BaseException as be:
                                print(be)
                                pass
            except Exception as e:
                print(e)
                pass


        unique_list_all_indexes = list(set(list_all_indexes))

        dataframe_output = dataframe.drop(index=unique_list_all_indexes, axis=0)

        return dataframe_output



    def filter_dataframe_by_limits(self, dataframe, limits_dict):
        """This function filters the dataframe by the limits in the limits_dict

        Args:
            dataframe (_type_): pd.DataFrame
            limits_dict (_type_): dictionary with the limits for each feature

        Returns:
            _type_: returns a dataframe filtered by the limits in the limits_dict
        """
        filtered_dataframe = dataframe.copy()
        for each_column in dataframe.columns:
            if each_column in limits_dict.keys():
                filtered_dataframe = filtered_dataframe.loc[(filtered_dataframe[each_column] >= float(limits_dict[each_column]["min"])) & (filtered_dataframe[each_column] <= float(limits_dict[each_column]["max"])), :]
            else:
                print(f"column {each_column} not in limits_dict.keys()")
                pass
        return filtered_dataframe


    def clean_up_data(self, dataframe, features=None, spc_cleaning_dict=None, limits_dict=None):

        if dataframe is not None:
            output_df = dataframe.copy()

        else:
            output_df = self.df


        if spc_cleaning_dict is not None:
            # print(f"create_plot_preprocessed_data: spc_cleaning_dict:")
            output_df = self.use_spc_cleaning_dict(output_df, spc_cleaning_dict)
        if limits_dict is not None:
            # print(f"create_plot_preprocessed_data: limits_dict:")
            output_df = self.filter_dataframe_by_limits(output_df, limits_dict)

        output_df = output_df.reset_index(drop=True)

        if features is not None:
            output_df = output_df[features]

        return output_df






















