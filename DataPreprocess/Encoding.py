"""
    Module: Encoder: contains functions to perform label encoder and OHE
    Input: 
        pandas dataframe 
    Output:
        pandas dataframe that contains encoded categorical values
"""
from pandas import get_dummies
from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder

class Encoding:
    def __init__(self) -> None:
        return

    @staticmethod
    def OHencoding(df):
        try:
            if not isinstance(df, DataFrame):
                raise TypeError
            # select categorical columns
            categorical = df.select_dtypes(['object']).columns
            # perform OHE
            cat = get_dummies(df[categorical], columns=categorical)
            # drop the categorical columns after OHE
            df = df.drop(categorical, axis=1)
            # join the OHE dataframe with original Dataframe
            for i in cat.columns:
                df[i] = cat[i]
        except TypeError as ex:
            print("Expected type of arg: pandas.core.frame.DataFrame")
        except Exception as ex:
            print(f"An Exception Occured! {str(ex)}")
        else:
            return df
    
    @staticmethod
    def Labencoding(df):
        try:
            if not isinstance(df, DataFrame):
                raise TypeError
            enc = LabelEncoder()
            # Label encode all the categorical columns
            for i in df.select_dtypes(['object']).columns:
                df[i] = enc.fit_transform(df[i])
        except TypeError as ex:
            print("Expected type of arg: pandas.core.frame.DataFrame")
        except Exception as ex:
            print(f"An Exception Occured! {str(ex)}")
        else:
            return df