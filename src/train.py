import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path


def load_season_sessions(year=2024):
    """Return a list of race sessions (all GPs in a season)."""
    schedule = fastf1.get_event_schedule(year)  # full season calendar [web:119]
    sessions = []

    for _, event in schedule.iterrows():
        if event["EventName"] is None:
            continue  # skip tests etc.
        try:
            session = fastf1.get_session(year, event["EventName"], "R")
            session.load()
            sessions.append(session)
        except Exception:
            # If a race fails to load for any reason, skip it
            continue

    return sessions


def build_training_dataframe_for_session(session):
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

    return df


def build_training_dataframe(year=2024):
    sessions = load_season_sessions(year)
    df_list = []

    for session in sessions:
        df_session = build_training_dataframe_for_session(session)
        df_list.append(df_session)

    if not df_list:
        raise RuntimeError("No session data collected for the season.")

    df_all = pd.concat(df_list, ignore_index=True)  # combine all races [web:143]

    # Encode categorical
    columns_to_encode = ["Compound"]
    encoder = OrdinalEncoder(categories=[["HARD", "MEDIUM", "SOFT"]])
    df_all[columns_to_encode] = encoder.fit_transform(df_all[columns_to_encode])

    feature_cols = ["LapTime", "Compound", "AirTemp"]
    X = df_all[feature_cols].copy()
    y = df_all["GridPosition"].astype(int)

    X = X.fillna(0)

    return X, y, encoder, feature_cols, df_all


def train_model():
    X, y, encoder, feature_cols, df_all = build_training_dataframe(year=2024)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(
        multi_class="multinomial",  # remove if your sklearn is too old
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.3f}")

    x_test_with_id = X_test.copy()
    x_test_with_id["Abbreviation"] = df_all.loc[X_test.index, "Abbreviation"]
    pred_table = x_test_with_id[["Abbreviation"]].copy()
    pred_table["predicted_grid"] = y_pred
    print(pred_table.head(20))

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "f1_grid_logreg.joblib")
    joblib.dump(
        {"encoder": encoder, "feature_cols": feature_cols},
        models_dir / "preprocess_logreg.joblib",
    )


if __name__ == "__main__":
    train_model()
