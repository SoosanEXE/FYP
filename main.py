import pandas as pd
from DataSets import constants as paths
from DataPreprocess import constants
from DataPreprocess.BinaryClassMapping import BinMap, BinMap1
from DataPreprocess.Scaling import Scaling
from DataPreprocess.Encoding import Encoding
from DataPreprocess.FeatureSelection import FeatureSelection
from AnomalyClassifier.AnomalyClassifier import AnomalyClassifier_nsl, AnomalyClassifier_unsw
from EvalMetrics.EvalMetrics import EvalMetrics
import joblib
import warnings

warnings.filterwarnings('ignore')

##########################

#read datasets
kdd_train = pd.read_csv(paths.TRAIN_PATH, header=None, names=constants.COL_NAMES)
kdd_test = pd.read_csv(paths.TEST_PATH, header=None, names=constants.COL_NAMES)

unsw_train = pd.read_csv(paths.TRAIN_PATH_UNSW)
unsw_test = pd.read_csv(paths.TEST_PATH_UNSW)

#make them as one
df = pd.concat([kdd_train, kdd_test])
df1 = pd.concat([unsw_train, unsw_test])
print("---------------------------------------------------------------------")
print("     NSL-KDD                         |     UNSW-NB15                 ")
print("---------------------------------------------------------------------")
print("     Dataset Loaded NSL-KDD          |    Dataset Loaded UNSW-NB15   ")
print("---------------------------------------------------------------------")
#map attack type to 0 or 1
df = BinMap(df)
df1 = BinMap1(df1)
#scale numerical columns
df = Scaling.scaling(df, df.select_dtypes(include=['float64', 'int64']).columns)
df1 = Scaling.scaling(df1, df1.select_dtypes(include=['float64', 'int64']).columns)

print("     Dataset Scaling is finished     |    Dataset Scaling is finished")
print("---------------------------------------------------------------------")

#for feature selection before make a copy

print("     Performing feature selection    |    Performing feature selection")
print("---------------------------------------------------------------------")

df_copy = df.copy()
df_copy1 = df1.copy()
#label encode categorical values
df_copy = Encoding.Labencoding(df_copy)
df_copy1 = Encoding.Labencoding(df_copy1)
#Feature Selection
X = df_copy.drop('attack_class', axis=1)
y = df_copy['attack_class']

X1 = df_copy1.drop('attack_cat', axis=1)
y1 = df_copy1['attack_cat']
#n = [10, 13, 15, 17, 21]

#for i in n:
new_fs = FeatureSelection.select_features(X, y, 'attack_class', 15)
#print("Selected features: ")
#print(new_fs)
df_new = df[new_fs]

new_fs1 = FeatureSelection.select_features(X1, y1, 'attack_cat', 15)
#print("Selected features: ")
#print(new_fs1)
df_new1 = df1[new_fs1]
print("     Performing One Hot encoding     |    Performing One Hot encoding")
print("---------------------------------------------------------------------")
df_new = Encoding.OHencoding(df)
df_new1 = Encoding.OHencoding(df1)
print("     Dataset preprocessed!           |    Dataset preprocessed!       ")
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
print("---------------------------------------------------------------------")