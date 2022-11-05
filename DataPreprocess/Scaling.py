from sklearn.preprocessing import MinMaxScaler
from numpy import array
minmax_scale = MinMaxScaler(feature_range=(0, 1))
def normalization(df,col):
  for i in col:
    arr = df[i]
    arr = array(arr)
    df[i] = minmax_scale.fit_transform(arr.reshape(len(arr),1))
  return df