import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("models") / "noshow_model.joblib"
FEATURE_COLS = [
    "age", "days_in_advance", "hour_of_booking",
    "scholarship", "hipertension", "diabetes",
    "alcoholism", "handcap", "sms_received",
]


def train(X, y, db=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\n=== Model Evaluation ===")
    print(classification_report(y_test, y_pred))
    print("========================\n")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
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
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)


def predict(model, X):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame([X], columns=FEATURE_COLS)
    else:
        X = pd.DataFrame(X.values, columns=FEATURE_COLS)
    probability = float(model.predict_proba(X)[:, 1][0])
    risk_level = "high" if probability >= 0.5 else "low"
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
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True)