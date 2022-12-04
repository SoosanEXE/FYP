import pandas as pd
from DataSets import constants as paths
from AnomalyClassifier.AnomalyClassifier import AnomalyClassifier_nsl, AnomalyClassifier_unsw
from EvalMetrics.EvalMetrics import EvalMetrics
from DataPreprocess.BinaryClassMapping import resultMap, resultMap1
from joblib import Parallel, delayed
import joblib
from DataSets import constants as paths
from DataPreprocess import constants
from IPython.display import HTML
from report import report
import warnings

warnings.filterwarnings('ignore')

##########################
kdd_train = pd.read_csv("/home/soosan/fyp/FYP/DataSets/KDDTrain+.txt", header=None, names=constants.COL_NAMES)
kdd_test = pd.read_csv("/home/soosan/fyp/FYP/DataSets/KDDTest+.txt", header=None, names=constants.COL_NAMES)

unsw_train = pd.read_csv("/home/soosan/fyp/FYP/DataSets/UNSW_NB15_training-set.csv")
unsw_test = pd.read_csv("/home/soosan/fyp/FYP/DataSets/UNSW_NB15_testing-set.csv")

#make them as one
df = pd.concat([kdd_train, kdd_test])
df1 = pd.concat([unsw_train, unsw_test])
#read datasets
print("Reading PreProcessed dataset")
df_new = pd.read_parquet(paths.PRE_DATA_NSL)
df_new1 = pd.read_parquet(paths.PRE_DATA_UNSW)
print("--------------------------------")
print("     NSL-KDD Evaluation        ")
print("--------------------------------")
clf, X_test, y_test, y_pred = AnomalyClassifier_nsl.getAnomalyClassifier(df_new, 'attack_class')
EvalMetrics.metrics(clf, X_test, y_test)

df['attack_class'] = y_pred
df= resultMap(df)
print("--------------------------------")
print("     UNSW-NB15 Evaluation      ")
print("--------------------------------")
clf1, X_test1, y_test1, y_pred1= AnomalyClassifier_unsw.getAnomalyClassifier(df_new1, 'attack_cat')
EvalMetrics.metrics(clf1, X_test1, y_test1)
df1['attack_cat'] = y_pred1
df1 = resultMap1(df1)

joblib.dump(clf, "FYP/NSLAnoClf.pkl")
joblib.dump(clf1, "FYP/UNSWAnoClf.pkl")
print("     Model Saved      ")
print("---------------------------------------------------------------------")
report.report_nsl(df)
report.report_unsw(df1)
print("     Generated a report    ")