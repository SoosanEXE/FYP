import pandas as pd
from DataSets import constants as paths
from DataPreprocess import constants
from DataPreprocess.BinaryClassMapping import BinMap, BinMapM
from DataPreprocess.Scaling import Scaling
from DataPreprocess.Encoding import Encoding
from DataPreprocess.FeatureSelection import FeatureSelection
from EvalMetrics.EvalMetrics import EvalMetrics
import joblib
from imblearn.over_sampling import SMOTE
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import Counter
warnings.filterwarnings('ignore')

##########################

#read datasets
kdd_train = pd.read_csv("DataSets/KDDTrain+.txt", header=None, names=constants.COL_NAMES)
kdd_test = pd.read_csv("DataSets/KDDTest+.txt", header=None, names=constants.COL_NAMES)

unsw_train = pd.read_csv("DataSets/UNSW_NB15_training-set.csv")
unsw_test = pd.read_csv("DataSets/UNSW_NB15_testing-set.csv")

#make them as one
df = pd.concat([kdd_train, kdd_test])
df1 = pd.concat([unsw_train, unsw_test])
print("---------------------------------------------------------------------")
print("     NSL-KDD                         |     UNSW-NB15                 ")
print("---------------------------------------------------------------------")
print("     Dataset Loaded NSL-KDD          |    Dataset Loaded UNSW-NB15   ")
print("---------------------------------------------------------------------")
#map attack type to 0 or 1
df = BinMapM(df, constants.NSL_TARGET)
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
#df_new1=df_new1[df_new1.attack_cat!='Normal']
df_mul = df_new.drop('attack_class',axis=1)
df_mul_tar = pd.DataFrame(df_new['attack_class'])
le2 = preprocessing.LabelEncoder()
enc_label = df_mul_tar.apply(le2.fit_transform)
#unsw
df_mul1 = df_new1.drop('attack_cat',axis=1)
df_mul_tar1 = pd.DataFrame(df_new1['attack_cat'])
le3 = preprocessing.LabelEncoder()
enc_label1 = df_mul_tar1.apply(le3.fit_transform)
print("     Performing One Hot encoding     |    Performing One Hot encoding")
print("---------------------------------------------------------------------")
df_new = Encoding.OHencoding(df_mul)
df_new1 = Encoding.OHencoding(df_mul1)
df_new['attack_class']= enc_label
df_new1['attack_cat']= enc_label1

df_new_x = df_new.drop('attack_class',axis=1)
df_new_y = df_new['attack_class']
X_train, X_test, y_train, y_test = train_test_split(df_new_x, df_new_y, test_size=0.2, random_state=0)
# Use SMOTE to oversample the minority class
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)
X_train['attack_class']=y_train
f_df=X_train
#unsw smote
df_new1_x = df_new1.drop('attack_cat',axis=1)
df_new1_y = df_new1['attack_cat']
X_train1, X_test1, y_train1, y_test1 = train_test_split(df_new1_x, df_new1_y, test_size=0.2, random_state=0)
# Use SMOTE to oversample the minority class
smote = SMOTE()
X_train1, y_train1 = smote.fit_resample(X_train1, y_train1)
X_train1['attack_cat']=y_train1
f_df1=X_train1
# print(df_new['attack_class'].value_counts())
# print(f_df['attack_class'].value_counts())
# print(df_new1['attack_cat'].value_counts())
# print(f_df1['attack_cat'].value_counts())
f_df.to_parquet("KDDpreprocessedsmote.parquet")
f_df1.to_parquet("UNSWpreprocessedsmote.parquet")
print("     Dataset preprocessed!           |    Dataset preprocessed!       ")
print("---------------------------------------------------------------------")