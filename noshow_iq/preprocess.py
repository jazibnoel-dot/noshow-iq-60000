import pandas as pd


def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("-", "_")
        .str.replace(" ", "_")
    )
    df["no_show"] = df["no_show"].map({"Yes": 1, "No": 0})
    df = df[df["age"] >= 0]
    df["scheduledday"] = pd.to_datetime(df["scheduledday"], utc=True)
    df["appointmentday"] = pd.to_datetime(df["appointmentday"], utc=True)
    df["days_in_advance"] = (df["appointmentday"] - df["scheduledday"]).dt.days
    df = df[df["days_in_advance"] >= 0]
    df["hour_of_booking"] = df["scheduledday"].dt.hour
    return df


def get_features_and_target(df: pd.DataFrame):
    feature_cols = [
        "age", "days_in_advance", "hour_of_booking",
        "scholarship", "hipertension", "diabetes",
        "alcoholism", "handcap", "sms_received",
    ]
    X = df[feature_cols]
    y = df["no_show"]
    return X, y