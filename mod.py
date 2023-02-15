import pandas as pd
from DataSets import constants as paths
from AnomalyClassifier import constants as modelpaths
from EvalMetrics.EvalMetrics import EvalMetrics
from DataPreprocess.BinaryClassMapping import resultMap
from joblib import Parallel, delayed
import joblib
from DataPreprocess import constants
from Report import constants as reportpaths
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (precision_score, recall_score,\
    f1_score, confusion_matrix ,accuracy_score, \
        ConfusionMatrixDisplay, roc_curve, roc_auc_score)
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
df_new1 = pd.read_parquet('UNSWpreprocessedsmote.parquet')
X = df_new1.drop('attack_cat', axis=1)
y = df_new1['attack_cat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(sum(y_train))
# print(sum(y_test))
# print(y_train.value_counts())
# print(y_test.value_counts())
# print(y.value_counts())
# print(df_new1.head())
#clf =  RandomForestClassifier()
base_estimator = DecisionTreeClassifier(max_depth=100)
aode = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, learning_rate=1.0)
aode.fit(X_train, y_train)
y_pred = aode.predict(X_train)
acc = accuracy_score(y_train, y_pred)
print('Accuracy:', acc)
#clf=  lightgbm.LGBMClassifier(objective='multiclass', n_estimators= 500, n_jobs=-1)
#clf=AdaBoostClassifier(n_estimators=100)
#clf=xgboost.XGBClassifier(n_estimators=150, n_jobs=-1)
#clf=ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
#clf = GradientBoostingClassifier()
#clf = SVC(kernel='linear')
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
# cm1 = confusion_matrix(y_test, y_pred)
# cm_df1 = pd.DataFrame(cm1,index = [0,1,2,3,4,5,6,7,8], columns = [0,1,2,3,4,5,6,7,8])
# print(cm_df1)
#print("Accuracy:", accuracy_score(y_test,y_pred))
# param_grid = {
#     'n_estimators': [25, 50, 100, 150],
#     'max_features': ['sqrt', 'log2', None],
#     'max_depth': [3, 6, 9],
#     'max_leaf_nodes': [3, 6, 9],
# }
# random_search = RandomizedSearchCV(RandomForestClassifier(),param_grid)
# random_search.fit(X_train, y_train)
# print(random_search.best_estimator_)
