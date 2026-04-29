import pandas as pd

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file and clean/prepare the dataset for analysis.

    Steps performed:
    1. Reads the CSV file into a pandas DataFrame.
    2. Cleans column names (removes spaces, makes lowercase, replaces symbols).
    3. Converts 'no_show' column from Yes/No to 1/0.
    4. Removes invalid age values (age < 0).
    5. Converts date columns to proper datetime format.
    6. Creates a new feature: days between booking and appointment.
    7. Removes records where appointment is before booking.
    8. Extracts booking hour from the scheduled day.

    Parameters:
    filepath (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Cleaned and processed dataset.
    """

    # Load dataset from CSV file
    df = pd.read_csv(filepath)

    # Clean column names:
    # - Remove extra spaces
    # - Convert to lowercase
    # - Replace '-' and spaces with '_'
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("-", "_")
        .str.replace(" ", "_")
    )

    # Convert 'no_show' column:
    # Yes → 1 (patient did not show up)
    # No → 0 (patient showed up)
    df["no_show"] = df["no_show"].map({"Yes": 1, "No": 0})

    # Remove invalid ages (age cannot be negative)
    df = df[df["age"] >= 0]

    # Convert date columns into datetime format (UTC timezone)
    df["scheduledday"] = pd.to_datetime(df["scheduledday"], utc=True)
    df["appointmentday"] = pd.to_datetime(df["appointmentday"], utc=True)

    # Create a new column:
    # Number of days between scheduling and appointment
    df["days_in_advance"] = (df["appointmentday"] - df["scheduledday"]).dt.days

    # Remove rows where appointment date is before scheduled date
    df = df[df["days_in_advance"] >= 0]

    # Extract the hour (0–23) when the appointment was booked
    df["hour_of_booking"] = df["scheduledday"].dt.hour

    return df


def get_features_and_target(df: pd.DataFrame):
    """
    Separate the dataset into input features (X) and target variable (y).

    Features (X): Variables used to predict no-show.
    Target (y): Whether the patient showed up or not.

    Parameters:
    df (pd.DataFrame): Cleaned dataset.

    Returns:
    tuple:
        X (pd.DataFrame): Feature variables
        y (pd.Series): Target variable (no_show)
    """

    # List of columns used as input features for the model
    feature_cols = [
        "age", "days_in_advance", "hour_of_booking",
        "scholarship", "hipertension", "diabetes",
        "alcoholism", "handcap", "sms_received",
    ]

    # X = input features (independent variables)
    X = df[feature_cols]

    # y = target variable (what we want to predict)
    y = df["no_show"]

    return X, y