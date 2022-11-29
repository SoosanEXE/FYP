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

def BinMap(df):
    # map label to either 0 or 1
    try:
        if not isinstance(df, DataFrame):
            raise TypeError
        df['attack_class']  = df['label'].apply(lambda v: 0 if v == "normal" else 1)
        df = df.drop('label', axis=1)
    except TypeError as ex:
        print(f"Expected type of arg: pandas.core.frame.DataFrame {str(ex)}")
    except KeyError:
        print("Please ensure df has col named 'label'")
    else:
        return df

def BinMap1(df):
    # map label to either 0 or 1
    try:
        if not isinstance(df, DataFrame):
            raise TypeError
        df['attack_cat']  = df['attack_cat'].apply(lambda v: 0 if v == "Normal" else 1)
        df=df.drop("label",axis=1)
    except TypeError as ex:
        print("Expected type of arg: pandas.core.frame.DataFrame")
    except KeyError:
        print("Please ensure df has col named 'label'")
    else:
        return df
