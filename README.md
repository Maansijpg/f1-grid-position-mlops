# F1 Grid Position Prediction (MLOps)

Predict a Formula 1 driver's starting grid position tier from lap time, tyre compound, and air temperature using a Logistic Regression classifier.

The model bins the grid into three tiers:
- **Front** (positions 1–5)
- **Mid** (positions 6–13)
- **Back** (positions 14–20)

Data is fetched live from the [FastF1](https://theoehrly.github.io/Fast-F1/) library using real 2024 F1 season data.

---

## Project Structure

```
f1-grid-position-mlops/
├── app/
│   └── streamlit_app.py        # Streamlit web UI for predictions
├── configs/
│   └── config.yaml             # Configuration placeholder
├── data/
│   ├── f1_qatar_2024_driver_laps.csv   # Sample dataset (Qatar GP 2024)
│   └── models/
│       └── f1_grid_xgb.joblib          # Legacy XGBoost model
├── models/
│   ├── f1_grid_logreg.joblib           # Trained Logistic Regression model
│   ├── f1_grid_xgb.joblib              # Legacy XGBoost model (copy)
│   └── preprocess_logreg.joblib        # Preprocessing artifacts (encoder + feature columns)
├── notebooks/                  # Jupyter notebooks (placeholder)
├── src/
│   ├── __init__.py
│   ├── train.py                # Training pipeline
│   └── predict.py              # Inference module
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

---

## Usage

### Train the model

```bash
python src/train.py
```

Fetches all 2024 F1 race sessions via FastF1, builds a training set from lap times, tyre compounds, and air temperature, then trains a multinomial Logistic Regression classifier. The model and preprocessing artifacts are saved to `models/`.

### Run the web app

```bash
streamlit run app/streamlit_app.py
```

Opens an interactive UI where you can input lap time, select a tyre compound, and set air temperature to predict the starting grid tier.

### Programmatic inference

```python
from src.predict import prepare_features

tier = prepare_features(compound="SOFT", lap_time_seconds=90.0, air_temp=25.0)
# Returns 0 (Front), 1 (Mid), or 2 (Back)
```

---

## Features

| Feature       | Type      | Description                          |
|---------------|-----------|--------------------------------------|
| LapTime       | float (s) | Lap time in seconds                  |
| Compound      | categorical | Tyre compound: HARD, MEDIUM, or SOFT |
| AirTemp       | float (°C) | Average session air temperature      |

## Target

Grid position is binned into three ordinal classes:

| Tier | Label  | Grid Positions |
|------|--------|----------------|
| 0    | Front  | 1–5            |
| 1    | Mid    | 6–13           |
| 2    | Back   | 14–20          |

---

## Model

- **Algorithm:** Logistic Regression with `multinomial` softmax output
- **Solver:** L-BFGS
- **Train/Test split:** 70/30 stratified
- **Feature encoding:** Ordinal encoding of tyre compound (HARD=0, MEDIUM=1, SOFT=2)

---

## Known Limitations

- Air temperature is averaged per session (all drivers get the same value), reducing its discriminative power.
- No feature scaling is applied (Logistic Regression with L-BFGS generally benefits from scaling).
- No hyperparameter tuning or cross-validation is performed.
- The training set includes race lap times, which may introduce data leakage since grid position is determined pre-race.
- No CI/CD, containerization, experiment tracking, or automated testing is configured yet.

---

## Dependencies

- `streamlit` — Web UI
- `fastf1` — F1 data fetching
- `pandas` / `numpy` — Data manipulation
- `scikit-learn` — ML pipeline
- `joblib` — Model serialization
- `pyyaml` — Config (reserved)

---

## License

MIT
