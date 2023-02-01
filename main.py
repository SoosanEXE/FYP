import pandas as pd
from DataSets import constants as paths
from AnomalyClassifier.AnomalyClassifier import AnomalyClassifier_nsl, AnomalyClassifier_unsw
from AnomalyClassifier import constants as modelpaths
from EvalMetrics.EvalMetrics import EvalMetrics
from DataPreprocess.BinaryClassMapping import resultMap
from joblib import Parallel, delayed
import joblib
from DataPreprocess import constants
from Report import constants as reportpaths
import warnings

warnings.filterwarnings('ignore')

##########################
kdd_train = pd.read_csv(paths.TRAIN_PATH, header=None, names=constants.COL_NAMES)
kdd_test = pd.read_csv(paths.TEST_PATH, header=None, names=constants.COL_NAMES)

unsw_train = pd.read_csv(paths.TRAIN_PATH_UNSW)
unsw_test = pd.read_csv(paths.TEST_PATH_UNSW)

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
df= resultMap(df, constants.NSL_TARGET)
df.to_parquet(reportpaths.NSL_PAR)
print("--------------------------------")
print("     UNSW-NB15 Evaluation      ")
print("--------------------------------")
clf1, X_test1, y_test1, y_pred1= AnomalyClassifier_unsw.getAnomalyClassifier(df_new1, 'attack_cat')
EvalMetrics.metrics(clf1, X_test1, y_test1)
df1['attack_cat'] = y_pred1
df1= resultMap(df1, constants.UNSW_TARGET)
df1.to_parquet(reportpaths.UNSW_PAR)

joblib.dump(clf, modelpaths.NSLANO)
joblib.dump(clf1, modelpaths.UNSWANO)
print("---------------------------------------------------------------------")
print("     Model Saved      ")
print("---------------------------------------------------------------------")