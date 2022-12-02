import pandas as pd
from sklearn.model_selection import train_test_split
from EvalMetrics.EvalMetrics import EvalMetrics
import joblib
import warnings

warnings.filterwarnings('ignore')

df = pd.read_parquet('/home/soosan/fyp/FYP/KDDpreprocessed.parquet')
target = 'attack_class'
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50, shuffle=True)

clf = joblib.load('/home/soosan/fyp/FYP/NSLAnoClf.pkl')

EvalMetrics.metrics(clf, X_test=X_test, y_test=y_test)