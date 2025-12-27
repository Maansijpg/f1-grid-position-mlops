import numpy as np
import pandas as pd
import joblib
from pathlib import Path


MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "f1_grid_logreg.joblib"
PREPROCESS_PATH = MODELS_DIR / "preprocess_logreg.joblib"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocess = joblib.load(PREPROCESS_PATH)
    encoder = preprocess["encoder"]
    feature_cols = preprocess["feature_cols"]
    return model, encoder, feature_cols


def prepare_features(compound, lap_time_seconds, air_temp):
    """
    compound: string, one of HARD/MEDIUM/SOFT
    lap_time_seconds: float (e.g. 94.123)
    air_temp: float
    """
    model, encoder, feature_cols = load_artifacts()

    # Build single-row dataframe
    df = pd.DataFrame(
        {
            "LapTime": [lap_time_seconds],
            "Compound": [compound],
            "AirTemp": [air_temp],
        }
    )

    # Encode compound using the same encoder
    df[["Compound"]] = encoder.transform(df[["Compound"]])

    X = df[feature_cols].copy()
    X = X.fillna(0)

    pred_grid = model.predict(X)[0]
    return int(pred_grid)
