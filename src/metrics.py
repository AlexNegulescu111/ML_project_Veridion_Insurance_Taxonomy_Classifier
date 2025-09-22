from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

def performance_label_metrics(y_true, y_pred, name):
    """
    Measure the main prediction performance metrics 
    """
    print("Entering the scores function")
    scores = {
        "name": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }
    

    return scores