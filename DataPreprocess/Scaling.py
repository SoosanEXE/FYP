"""
    Module: Scaler: to scale given dataframe using MinMax scaler
    Input: 
        pandas dataframe and column list
        df, col
    Output:
        pandas dataframe scaled using minmax scaler
"""
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from pandas.core.frame import DataFrame

class Scaling:
    def __init__(self) -> None:
        return

    @staticmethod
    def scaling(df, col):
        #scale each columns using min max scaler
        try:
            if not isinstance(df, DataFrame):
                raise TypeError
            scaler = MinMaxScaler(feature_range=(0,1))
            for i in col:
                arr = df[i]
                arr = array(arr)
                df[i] = scaler.fit_transform(arr.reshape(len(arr), 1))
        except TypeError as ex:
            print("Expected type of arg: pandas.core.frame.DataFrame")
        except Exception as ex:
            print(f"Exception occured! {str(ex)}")
        else:
            return df