import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path


def load_session(year=2024, gp="Qatar", session_type="R"):
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def build_training_dataframe(session):
    laps = session.laps
    weather = session.weather_data

    # Simple weather feature
    avg_air_temp = weather["AirTemp"].mean()
    laps["AirTemp"] = avg_air_temp

    results = session.results
    grid_pos = results[["Abbreviation", "GridPosition"]]

    laps = laps.rename(columns={"Driver": "Abbreviation"})
    laps = laps.merge(grid_pos, on="Abbreviation", how="left")

    df = laps[["Abbreviation", "LapTime", "Compound", "GridPosition", "AirTemp"]].copy()

    # Drop rows with missing grid position or laptime
    df = df.dropna(subset=["GridPosition", "LapTime"])

    # Convert LapTime to seconds
    df["LapTime"] = df["LapTime"].dt.total_seconds()

    # Encode categorical
    columns_to_encode = ["Compound"]
    encoder = OrdinalEncoder(categories=[["HARD", "MEDIUM", "SOFT"]])
    df[columns_to_encode] = encoder.fit_transform(df[columns_to_encode])

    # Features and target
    feature_cols = ["LapTime", "Compound", "AirTemp"]
    X = df[feature_cols].copy()
    y = df["GridPosition"].astype(int)

    X = X.fillna(0)

    return X, y, encoder, feature_cols, df


def train_model():
    session = load_session()
    X, y, encoder, feature_cols, df = build_training_dataframe(session)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Multinomial logistic regression for multi‑class grid positions
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.3f}")

    # Build a small prediction table for inspection (optional)
    x_test_with_id = X_test.copy()
    x_test_with_id["Abbreviation"] = df.loc[X_test.index, "Abbreviation"]
    pred_table = x_test_with_id[["Abbreviation"]].copy()
    pred_table["predicted_grid"] = y_pred
    print(pred_table.head(20))

    # Save model and encoder
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "f1_grid_logreg.joblib")
    joblib.dump(
        {"encoder": encoder, "feature_cols": feature_cols},
        models_dir / "preprocess_logreg.joblib",
    )


if __name__ == "__main__":
    train_model()

