"""
    Module: Evaluation: prints Evaluation metrics given a model and test data
    Input: clf (classification model), X_test and y_test (Test Data)
    Output: NaN
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, confusion_matrix ,accuracy_score)
class EvalMetrics:
    def __init__():
        return
    
    @staticmethod
    def metrics(clf, X_test, y_test):
        try:
            y_pred = clf.predict(X_test)
            
            print("Confusion Matrix:")
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            print(confusion_matrix(y_test, y_pred))

            print("\nAccuracy:")
            print(accuracy_score(y_test, y_pred))

            print("\nRecall:")
            print(recall_score(y_test, y_pred))

            print("\nPrecision:")
            print(precision_score(y_test, y_pred, average=None))

            print("\nF1 score:")
            print(f1_score(y_test, y_pred, average=None))

            print("\n False positive rate:")
            fpr = fp / (tn+fp)
            print(fpr)
        except AttributeError as ex:
            print(f"Please ensure an ML model is provided as an arg {str(ex)}")
        except Exception as ex:
            print(f"An Excpetion Occured {str(ex)}")


