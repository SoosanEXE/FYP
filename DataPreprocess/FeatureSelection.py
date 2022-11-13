"""
    Module: FeatureSelection: Performs ANOVA F feature selection on given X, y
        X = Independent features
        y = Traget Feature
    Input: 
        X and y 
    Output:
        a python list of selected features
"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split


class FeatureSelection:
    def __init__(self) -> None:
        return

    @staticmethod
    def select_features(X, y, target, n):
        try:
            # Select K best features based on f-score
            fs = SelectKBest(score_func=f_classif, k=n)
            fs.fit(X, y)
            m = fs.get_support()
            new_features = X.columns[m]
            nfs = []
            for i in new_features:
                nfs.append(i)
            # append target feature and return
            nfs.append(target)
        except Exception as ex:
            print(f"An Exception Occured! {str(ex)}")
        else:
            return nfs
