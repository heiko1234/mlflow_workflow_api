


import collections
import pandas as pd


try:
    from backend_service.utilities.nelson  import (
        rule1,
        rule2,
        rule3,
        rule4,
        rule5,
        rule6,
        rule7,
        rule8,
    )
except ModuleNotFoundError:
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





def transform_cleaning_table_in_dict(dataframe):
    """ This function gives a dictionary of the usage of the nelson rules for each feature in the data cleaning table

    Args:
        dataframe (_type_): pandas dataframe

    Returns:
        _type_: returns a dictionary with separated rules for each feature
    """
    
    dict = {}
    
    list_of_rules = ["rule1", "rule2", "rule3", "rule4", "rule5", "rule6", "rule7", "rule8"]

    for element_in_description in dataframe["description"].unique():
        dict[element_in_description] = {}
        
        for rule in list_of_rules:
            if rule  in dataframe.columns:
                dict[element_in_description][rule] = dataframe.loc[dataframe["description"]==element_in_description][rule].values[0]
                
    return dict





def use_spc_cleaning_dict(dataframe, spc_cleaning_dict):
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




def create_limits_dict(limits_table_df):
    """This function creates a dictionary with the limits for each feature

    Args:
        limits_table_df (pd.DataFrame): dataframe with the limits for each feature ["description", "mean", "std", "min", "max"]

    Returns:
        _type_: returns a dictionary with the limits for each feature
    """
    limits_dict = {}

    for each_row in limits_table_df["description"]:
        
        limits_dict[each_row] = {
            "mean": float(limits_table_df.loc[limits_table_df["description"]==each_row, "mean"].values[0]),
            "std": float(limits_table_df.loc[limits_table_df["description"]==each_row, "std"].values[0]),
            "min": float(limits_table_df.loc[limits_table_df["description"]==each_row, "min"].values[0]),
            "max": float(limits_table_df.loc[limits_table_df["description"]==each_row, "max"].values[0]),
        }
        
        
        # limits_dict.loc[each_row] = {
        #     "mean": limits_table_df.loc[each_row, "mean"],
        #     "std": limits_table_df.loc[each_row, "std"],
        #     "min": limits_table_df.loc[each_row, "min"],
        #     "max": limits_table_df.loc[each_row, "max"],
        # }
        
    return limits_dict





def filter_dataframe_by_limits(dataframe, limits_dict):
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




def update_nested_dict(original_dict, overwrite_dict):
    """This function updates a nested dictionary

    Args:
        original_dict (dict): any dictionary
        overwrite_dict (dict): any subset of the original_dict, that will overwrite the original_dict
    Returns:
        _type_: returns the original_dict updated with the overwrite_dict
    """
    
    
    for k, v in overwrite_dict.items():
        if isinstance(v, collections.abc.Mapping):
            original_dict[k] = update_nested_dict(original_dict.get(k, {}), v)
        else:
            original_dict[k] = v
            
    return original_dict






# limit_table_df = pd.DataFrame()

# limit_table_df["description"] = ["Yield", "Feature1", "Feature2"]
# limit_table_df["mean"] = [0.5, 0.5, 0.5]
# limit_table_df["std"] = [0.1, 0.1, 0.1]
# limit_table_df["min"] = [5, 10, 15]
# limit_table_df["max"] = [10, 15, 20]


# create_limits_dict(limit_table_df)

# # >>> create_limits_dict(limit_table_df)
# # {'Yield': {'mean': 0.5, 'std': 0.1, 'min': 5.0, 'max': 10.0}, 'Feature1': {'mean': 0.5, 'std': 0.1, 'min': 10.0, 'max': 15.0}, 'Feature2': {'mean': 0.5, 'std': 0.1, 'min': 15.0, 'max': 20.0}}






# dataframe_cleaning = pd.DataFrame()

# # no cleaning,  remove data  

# dataframe_cleaning["description"] = ["Yield", "Feature1", "Feature2"]
# dataframe_cleaning["rule1"] = ["no cleaning", "no cleaning", "no cleaning"]
# dataframe_cleaning["rule2"] = ["no cleaning", "no cleaning", "no cleaning"]
# dataframe_cleaning["rule3"] = ["no cleaning", "remove data", "no cleaning"]
# dataframe_cleaning["rule4"] = ["no cleaning", "no cleaning", "no cleaning"]
# dataframe_cleaning["rule5"] = ["no cleaning", "no cleaning", "no cleaning"]
# dataframe_cleaning["rule6"] = ["no cleaning", "no cleaning", "no cleaning"]
# dataframe_cleaning["rule7"] = ["no cleaning", "no cleaning", "no cleaning"]
# dataframe_cleaning["rule8"] = ["no cleaning", "no cleaning", "no cleaning"]




# transform_cleaning_table_in_dict(dataframe_cleaning)

# # >>> transform_cleaning_table_in_dict(dataframe_cleaning)
# # {'Yield': {'rule1': 'no cleaning', 'rule2': 'no cleaning', 'rule3': 'no cleaning', 'rule4': 'no cleaning', 'rule5': 'no cleaning', 'rule6': 'no cleaning', 'rule7': 'no cleaning', 'rule8': 'no cleaning'}, 'Feature1': {'rule1': 'no cleaning', 'rule2': 'no cleaning', 'rule3': 'remove data', 'rule4': 'no cleaning', 'rule5': 'no cleaning', 'rule6': 'no cleaning', 'rule7': 'no cleaning', 'rule8': 'no cleaning'}, 'Feature2': {'rule1': 'no cleaning', 'rule2': 'no cleaning', 'rule3': 'no cleaning', 'rule4': 'no cleaning', 'rule5': 'no cleaning', 'rule6': 'no cleaning', 'rule7': 'no cleaning', 'rule8': 'no cleaning'}}



# use_spc_cleaning_dict(dataframe, spc_cleaning_dict)






