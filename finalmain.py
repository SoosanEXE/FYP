from joblib import Parallel, delayed
import joblib
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')
model = joblib.load('FYP/AnomalyClassifier/NSLAnoClf.pkl')
df = pd.read_parquet("FYP/DataPreprocess/KDDpreprocessed.parquet")
df_mul = pd.read_parquet("FYP/KDDpreprocessed1.parquet")
df.drop("attack_class",axis=1,inplace=True)
predicted_labels =model.predict(df)
#Convert the predicted_labels to a Pandas DataFrame
data= pd.DataFrame(predicted_labels, columns=['attack_class'])
df['attack_class']=data
model1 = joblib.load('FYP/AnomalyClassifier/UNSWAnoClf.pkl')
df1 = pd.read_parquet("FYP/DataPreprocess/UNSWpreprocessed.parquet")
df1_mul = pd.read_parquet("FYP/UNSWpreprocessed1.parquet")
df1.drop("attack_cat",axis=1,inplace=True)
predicted_labels1 =model1.predict(df1)
#Convert the predicted_labels to a Pandas DataFrame
data= pd.DataFrame(predicted_labels1, columns=['attack_cat'])
df1['attack_cat']=data
df1=df1[df1.attack_cat!=0]
df=df[df.attack_class!=0]
col1=np.intersect1d(df1_mul.columns, df1.columns)
col=np.intersect1d(df.columns, df_mul.columns)
df3=pd.DataFrame(df.loc[:, col])
df4=pd.DataFrame(df1.loc[:, col1])
# print(df3)
# print(df3.shape)
# set1=set(df4.columns)
# set2=set(df1_mul.columns)
# #unique_to_df1 = set1 - set2
# unique_to_df2 = set2 - set1
# #print("Unique columns in df1:", unique_to_df1)
# print("Unique columns in df2:", unique_to_df2)
# df4['label'] = df1_mul['label']
# print(df4.shape)
# loaded_model = joblib.load('AnomalyClassifier/UNSWAnoClf1.pkl')
# feature_names = loaded_model.feature_names_in_
# print("Model features:", feature_names)
# print(len(feature_names))
# joblib.dump(loaded_model, 'AnomalyClassifier/UNSWAnoClf1.pkl')
# loaded_model = joblib.load('AnomalyClassifier/UNSWAnoClf1.pkl')
# feature_names = loaded_model.feature_names_in_
# print("Model features:", feature_names)
model3 = joblib.load('FYP/AnomalyClassifier/NSLAnoClf1.pkl')
df3.drop("attack_class",axis=1,inplace=True)
predicted_labels3 =model3.predict(df3)
#Convert the predicted_labels to a Pandas DataFrame
data3= pd.DataFrame(predicted_labels3, columns=['attack_class'])
df3['attack_class']=data3
print(df3['attack_class'].value_counts())
print(".................................................................")
model4 = joblib.load('FYP/AnomalyClassifier/UNSWAnoClf1.pkl')
df4.drop("attack_cat",axis=1,inplace=True)
predicted_labels4 =model4.predict(df4)
#Convert the predicted_labels to a Pandas DataFrame
data4= pd.DataFrame(predicted_labels4, columns=['attack_class'])
df4['attack_cat']=data4
print(df4['attack_cat'].value_counts())
