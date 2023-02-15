import pandas as pd
from DataSets import constants as paths
from AnomalyClassifier.AnomalyClassifier1 import AnomalyClassifier_nsl, AnomalyClassifier_unsw
from AnomalyClassifier import constants as modelpaths
from EvalMetrics.EvalMetrics import EvalMetrics
from DataPreprocess.BinaryClassMapping import resultMap
from joblib import Parallel, delayed
import joblib
from DataPreprocess import constants
from IPython.display import HTML
from Report import constants as reportpaths
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import numpy as np
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
df_new = pd.read_parquet(paths.PRE_DATA_NSLM)
df_new1 = pd.read_parquet(paths.PRE_DATA_UNSWM)
print("--------------------------------")
print("     NSL-KDD Evaluation        ")
print("--------------------------------")
mclf, X_test, y_test, y_pred = AnomalyClassifier_nsl.getAnomalyClassifier(df_new, 'attack_class')
print('\nNSL_Accuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,index = [0,1,2,3,4], columns = [0,1,2,3,4])
print(cm_df)
#EvalMetrics.metrics(mclf, X_test, y_test)
print("--------------------------------")
print("     UNSW-NB15 Evaluation      ")
print("--------------------------------")
mclf1, X_test1, y_test1, y_pred1= AnomalyClassifier_unsw.getAnomalyClassifier(df_new1, 'attack_cat')
# X=df_new1.drop('attack_cat',axis=1)
# y=df_new1['attack_cat']
# # Define the number of folds
# k = 10
# # Create an instance of StratifiedKFold
# skf = StratifiedKFold(n_splits=k)

# # Initialize an array to store the accuracy scores
# scores = []

# # Loop through the folds
# for train_index, test_index in skf.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
    
#     # Fit the model on the training data
#     mclf1.fit(X_train, y_train)
    
#     # Make predictions on the test data
#     y_pred = mclf1.predict(X_test)
    
#     # Calculate the accuracy score
#     score = accuracy_score(y_test, y_pred)
    
#     # Append the accuracy score to the scores array
#     scores.append(score)

# # Calculate the average accuracy score
# avg_score = np.mean(scores)
# # Print the average accuracy score
# print("Average accuracy:", avg_score)
print('\nUNSW_Accuracy: {:.2f}\n'.format(accuracy_score(y_test1, y_pred1)))
cm1 = confusion_matrix(y_test1, y_pred1)
cm_df1 = pd.DataFrame(cm1,index = [0,1,2,3,4,5,6,7,8,9], columns = [0,1,2,3,4,5,6,7,8,9])
print(cm_df1)
#EvalMetrics.metrics(mclf1, X_test1, y_test1)
joblib.dump(mclf, modelpaths.NSLANOM)
joblib.dump(mclf1, modelpaths.UNSWANOM)
print("---------------------------------------------------------------------")
print("     Model Saved      ")
print("---------------------------------------------------------------------")