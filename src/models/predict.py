import joblib
import pandas as pd


def predict_one(sample: pd.DataFrame, model_path="models/xgboost.joblib"):
    model = joblib.load(model_path)
    return model.predict(sample)
