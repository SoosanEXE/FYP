import pandas as pd
from DataSets import constants as paths
from AnomalyClassifier.AnomalyClassifier import AnomalyClassifier_nsl, AnomalyClassifier_unsw
from EvalMetrics.EvalMetrics import EvalMetrics
from joblib import Parallel, delayed
import joblib

import warnings

warnings.filterwarnings('ignore')

##########################

#read datasets
print("Reading PreProcessed dataset")
df_new = pd.read_parquet(paths.PRE_DATA_NSL)
df_new1 = pd.read_parquet(paths.PRE_DATA_UNSW)
print("--------------------------------")
print("     NSL-KDD Evaluation        ")
print("--------------------------------")
clf, X_test, y_test = AnomalyClassifier_nsl.getAnomalyClassifier(df_new, 'attack_class')
EvalMetrics.metrics(clf, X_test, y_test)
print("--------------------------------")
print("     UNSW-NB15 Evaluation      ")
print("--------------------------------")
clf1, X_test1, y_test1 = AnomalyClassifier_unsw.getAnomalyClassifier(df_new1, 'attack_cat')
EvalMetrics.metrics(clf1, X_test1, y_test1)
joblib.dump(clf, "FYP/NSLAnoClf.pkl")
joblib.dump(clf1, "FYP/UNSWAnoClf.pkl")
print("     Model Saved      ")
print("---------------------------------------------------------------------")
