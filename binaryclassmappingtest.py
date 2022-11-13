"""
Module to test BinaryClassMapping.py
"""
from DataPreprocess.BinaryClassMapping import BinMap
from DataSets import constants as path_const
from DataPreprocess import constants
import pandas as pd

df = pd.read_csv(path_const.TRAIN20_PATH, header=None, names=constants.COL_NAMES)
print(df.head())
df = BinMap(df)
print("\n\nAfter mapping \n\n")
print(df. head())