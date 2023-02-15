"""
    Module: Classifier: returns a classfier object
        voting classifier made of rfc and adaboost
    Input: 
        pandas dataframe and target feature 
    Output:
        classifier object,
        X_test -- Test Data
        y_test -- Test Data
"""
from pandas.core.frame import DataFrame
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
import pandas as pd
import xgboost
from sklearn.ensemble import GradientBoostingClassifier
class AnomalyClassifier_nsl:
    def __init__():
        return
    
    @staticmethod
    def getAnomalyClassifier(df, target):
        try:
            if not isinstance(df, DataFrame):
                raise TypeError
            # create RFC and AdaBoost Instances
            rfc = RandomForestClassifier(n_estimators=100)
            adc = AdaBoostClassifier(n_estimators=100)
            # create list of estimators
            clfs = []
            clfs.append(('RandomForest', rfc))
            clfs.append(('AdaBoost', adc))

            # create voting classifier 
            anoClf = VotingClassifier(estimators=clfs, voting='soft')

            # seperate target and independent features
            X = df.drop(target, axis=1)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50, shuffle=True)
            # fit the model
            anoClf.fit(X_train, y_train)
            y_pred = anoClf.predict(X_test)
        except TypeError as ex:
            print("Expected type of arg: pandas.core.frame.DataFrame")
        except Exception as ex:
            print(f"An Exception Occured! {str(ex)}")
        else:
            return anoClf, X_test, y_test, y_pred

class AnomalyClassifier_unsw:
    def __init__():
        return
    
    @staticmethod
    def getAnomalyClassifier(df, target):
        try:
            if not isinstance(df, DataFrame):
                raise TypeError
            # create RFC and AdaBoost Instances
            rfc = xgboost.XGBClassifier(n_estimators=150, n_jobs=-1)
            ETC = GradientBoostingClassifier()
            # create list of estimators
            clfs = []
            clfs.append(('RandomForest', rfc))
            clfs.append(('ETC', ETC))
            # create voting classifier 
            anoClf = VotingClassifier(estimators=clfs, voting='soft')
            # seperate target and independent features
            X = df.drop(target, axis=1)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50, shuffle=True)
    
            # fit the model
            anoClf.fit(X_train, y_train)
            y_pred = anoClf.predict(X_test)
        except TypeError as ex:
            print("Expected type of arg: pandas.core.frame.DataFrame")
        except Exception as ex:
            print(f"An Exception Occured! {str(ex)}")
        else:
            return anoClf, X_test, y_test, y_pred