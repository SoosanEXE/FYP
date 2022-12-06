"""
    Module to map attack to either 0 or 1
    Input:
        pandas dataframe with 'attack' col
    output:
        pandas dataframe with new col 'attack_class' with 0 (normal) or 1 (attack)
    exception handling
        exception will be thrown when given dataset doesnt have 'attack' col
"""

from pandas.core.frame import DataFrame
import constants

def BinMap(df, target):
    # map label to either 0 or 1
    try:
        if not isinstance(df, DataFrame):
            raise TypeError
        if target == constants.NSL_TARGET:
            df[target]  = df[target].apply(lambda v: 0 if v == "normal" else 1)
        elif target == constants.UNSW_TARGET:
            df[target]  = df[target].apply(lambda v: 0 if v == "Normal" else 1)
    except TypeError as ex:
        print(f"Expected type of arg: pandas.core.frame.DataFrame {str(ex)}")
    except KeyError:
        print("Please ensure df has col named 'label'")
    else:
        return df

def resultMap(df, target):
    # map label to either 0 or 1
    try:
        if not isinstance(df, DataFrame):
            raise TypeError
        df[target]  = df[target].apply(lambda v: "Normal" if v == 0 else "Attack")
    except TypeError as ex:
        print("Expected type of arg: pandas.core.frame.DataFrame")
    except KeyError:
        print("Please ensure df has col named 'label'")
    else:
        return df