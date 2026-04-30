import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Path where the trained model will be saved
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "noshow_model.joblib")

# List of features used for training and prediction
FEATURE_COLS = [
    "age", "days_in_advance", "hour_of_booking",
    "scholarship", "hipertension", "diabetes",
    "alcoholism", "handcap", "sms_received",
]


def train(X, y, db=None):
    """
    Train a machine learning model to predict patient no-shows.

    Steps performed:
    1. Split data into training and testing sets.
    2. Handle class imbalance using SMOTE (creates synthetic samples).
    3. Train a Random Forest model.
    4. Evaluate the model on test data.
    5. Save the trained model to disk.
    6. Optionally store training results in MongoDB.

    Parameters:
    X (pd.DataFrame): Input features
    y (pd.Series): Target variable (no_show)
    db: Optional database connection (MongoDB)

    Returns:
    tuple:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """

    # Split dataset into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE to balance the dataset (important if classes are uneven)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )

    # Train the model using resampled (balanced) data
    model.fit(X_train_res, y_train_res)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Generate evaluation report (precision, recall, f1-score)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Print evaluation results
    print("\n=== Model Evaluation ===")
    print(classification_report(y_test, y_pred))
    print("========================\n")

    # Create folder if it doesn't exist and save the model
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # If database is provided, store training details in MongoDB
    if db is not None:
        db["training_runs"].insert_one({
            "timestamp": datetime.utcnow(),
            "training_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "imbalance_technique": "SMOTE",
            "metrics": report,
        })
        print("Training run saved to MongoDB.")

    return model, X_test, y_test


def load_model():
    if not os.path.exists(MODEL_PATH):   # ← fix: call os.path.exists(), not .exists()
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)


def predict(model, X):
    """
    Make a prediction for patient no-show risk.

    Steps:
    1. Ensure input is in DataFrame format.
    2. Predict probability of no-show.
    3. Assign risk level (high or low).
    4. Provide recommendation based on risk.

    Parameters:
    model: Trained machine learning model
    X: Input data (single row or DataFrame)

    Returns:
    dict: Prediction result including risk level, probability, and recommendation
    """

    # Ensure input is a DataFrame with correct columns
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame([X], columns=FEATURE_COLS)
    else:
        X = pd.DataFrame(X.values, columns=FEATURE_COLS)

    # Get probability of class "1" (no-show)
    probability = float(model.predict_proba(X)[:, 1][0])

    # Decide risk level based on probability threshold
    risk_level = "high" if probability >= 0.5 else "low"

    # Provide recommendation based on risk level
    recommendation = (
        "High no-show risk. Call the patient and send an SMS reminder."
        if risk_level == "high"
        else "Low no-show risk. A standard SMS reminder is sufficient."
    )

    return {
        "risk_level": risk_level,
        "probability": round(probability, 4),
        "recommendation": recommendation,
    }


def evaluate(model, X_test, y_test):
    """
    Evaluate the trained model on test data.

    Parameters:
    model: Trained machine learning model
    X_test (pd.DataFrame): Test features
    y_test (pd.Series): True labels

    Returns:
    dict: Classification report with evaluation metrics
    """

    # Predict on test data
    y_pred = model.predict(X_test)

    # Return evaluation metrics (precision, recall, f1-score)
    return classification_report(y_test, y_pred, output_dict=True)