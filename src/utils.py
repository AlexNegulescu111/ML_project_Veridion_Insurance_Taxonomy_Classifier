import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import pandas as pd

def load_data(data):
    df = pd.read_csv(ROOT / "data" / f"{data}.csv")
    return df

def save_model(model, path):
    import joblib
    joblib.dump(model, path)
    print("Model saved successfully")

def load_model(path):
    import joblib
    return joblib.load(path)
