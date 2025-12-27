from pathlib import Path
import joblib
import pandas as pd

# Path to the saved model (relative to repo root)
MODEL_PATH = Path("data/models/f1_grid_xgb.joblib")

# Load model once at import time
_model = joblib.load(MODEL_PATH)


def predict_grid(lap_time_sec: float, compound: str, air_temp: float) -> int:
    """
    Predict grid position given lap time (seconds), tyre compound and air temperature.
    """
    compound_map = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}
    compound_code = compound_map[compound]

    X = pd.DataFrame([{
        "LapTime": lap_time_sec,
        "Compound_code": compound_code,
        "AirTemp": air_temp,
    }])

    pred = _model.predict(X)[0]
    return int(pred)


if __name__ == "__main__":
    # quick smoke test
    print(predict_grid(90.0, "MEDIUM", 30.0))
