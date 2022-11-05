import pandas as pd
from DataPreprocess import constants

def main():

    """Read Dataset(s)"""
    kdd_train = pd.read_csv("DataSets/KDDTrain+.txt", header=None, names=constants.COL_NAMES)
    kdd_test = pd.read_csv("DataSets/KDDTest+.txt", header=None, names=constants.COL_NAMES)

    """EDA?"""
    """Scaling"""
    """Encoding"""
    """Feature Selection"""
    """Model Trainig"""
    """Evaluation Metrics"""

    return