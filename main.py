import pandas as pd
from DataSets import constants as paths
from DataPreprocess import constants
from DataPreprocess.BinaryClassMapping import BinMap
from DataPreprocess.Scaling import Scaling
from DataPreprocess.Encoding import Encoding
from DataPreprocess.FeatureSelection import FeatureSelection
from AnomalyClassifier.AnomalyClassifier import AnomalyClassifier
from EvalMetrics.EvalMetrics import EvalMetrics
import joblib
import warnings

warnings.filterwarnings('ignore')

##########################

#read datasets
kdd_train = pd.read_csv(paths.TRAIN_PATH, header=None, names=constants.COL_NAMES)
kdd_test = pd.read_csv(paths.TEST_PATH, header=None, names=constants.COL_NAMES)

#make them as one
df = pd.concat([kdd_train, kdd_test])
print("Dataset Loaded")

#map attack type to 0 or 1
df = BinMap(df)

#scale numerical columns
df = Scaling.scaling(df, df.select_dtypes(include=['float64', 'int64']).columns)
print("Dataset Scaling is finished")

#for feature selection before make a copy
print("Performing feature selection....")
df_copy = df.copy()
#label encode categorical values
df_copy = Encoding.Labencoding(df_copy)
#Feature Selection
X = df_copy.drop('attack_class', axis=1)
y = df_copy['attack_class']

#n = [10, 13, 15, 17, 21]

#for i in n:
new_fs = FeatureSelection.select_features(X, y, 'attack_class', 15)
print("Selected features: ")
print(new_fs)
df_new = df[new_fs]

print("Performing One Hot encoding..")
df_new = Encoding.OHencoding(df)
print("Dataset preprocessed!")

clf, X_test, y_test = AnomalyClassifier.getAnomalyClassifier(df_new, 'attack_class')
EvalMetrics.metrics(clf, X_test, y_test)
