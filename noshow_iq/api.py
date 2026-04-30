import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

"""
This file creates a web API using FastAPI.

What this API does:
1. Loads a trained machine learning model.
2. Accepts patient data as input.
3. Predicts whether the patient will miss their appointment.
4. Stores prediction results in MongoDB (optional).
5. Provides endpoints to check health, view history, and see stats.

This is the "backend" of your project.
"""

# Load environment variables (like MongoDB connection string)
load_dotenv()

# Create FastAPI application with basic info
app = FastAPI(
    title="NoShowIQ",
    description="Predicts which patients will miss their appointment",
    version="0.1.0",
)

# Get MongoDB URI from environment, or use default local database
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")


def get_db():
    """
    Connect to MongoDB database.

    Returns:
    Database object (noshowiq)
    """
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return client["noshowiq"]


def get_model():
    """
    Load the trained machine learning model.

    Returns:
    Loaded model from file
    """
    from noshow_iq.model import load_model
    return load_model()


# Global variable to store the model
model = None


class AppointmentInput(BaseModel):
    """
    This class defines the structure of input data for prediction.

    Each field represents patient information required by the model.
    """
    age: int
    scholarship: int
    hipertension: int
    diabetes: int
    alcoholism: int
    handcap: int
    sms_received: int
    scheduled_day: datetime
    appointment_day: datetime
    gender: str | None = None


@app.get("/health")
def health():
    """
    Health check endpoint.

    Used to verify:
    - API is running
    - Model is loaded or not
    """
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(record: AppointmentInput):
    """
    Predict whether a patient will miss their appointment.

    Steps:
    1. Check if model is loaded.
    2. Convert input data into DataFrame.
    3. Run prediction using trained model.
    4. Store result in MongoDB (if available).
    5. Return prediction result.

    Parameters:
    record (AppointmentInput): Patient input data

    Returns:
    dict: Prediction result (risk level, probability, recommendation)
    """

    # If model is not loaded, return error
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    # Import prediction function
    from noshow_iq.model import predict as model_predict

    # Convert input object to dictionary
    input_dict = record.dict()

    # Compute model features from appointment timestamps
    days_in_advance = int((input_dict["appointment_day"] - input_dict["scheduled_day"]).days)
    hour_of_booking = input_dict["scheduled_day"].hour

    feature_dict = {
        "age": input_dict["age"],
        "days_in_advance": days_in_advance,
        "hour_of_booking": hour_of_booking,
        "scholarship": input_dict["scholarship"],
        "hipertension": input_dict["hipertension"],
        "diabetes": input_dict["diabetes"],
        "alcoholism": input_dict["alcoholism"],
        "handcap": input_dict["handcap"],
        "sms_received": input_dict["sms_received"],
    }

    # Convert dictionary to DataFrame (model expects this format)
    df = pd.DataFrame([feature_dict])

    # Get prediction result
    result = model_predict(model, df)

    # Try to save prediction to MongoDB
    try:
        db = get_db()
        db["predictions"].insert_one({
            "timestamp": datetime.utcnow(),
            "input": input_dict,
            "risk_level": result["risk_level"],
            "probability": result["probability"],
            "recommendation": result["recommendation"],
        })
    except Exception:
        # Ignore errors if DB is not available
        pass

    return result


@app.get("/history")
def history():
    """
    Get last 20 prediction records from database.

    Returns:
    list: Recent prediction records
    """

    try:
        db = get_db()

        # Fetch last 20 predictions sorted by latest timestamp
        docs = list(db["predictions"].find({}, {"_id": 0}).sort("timestamp", -1).limit(20))

        return docs
    except Exception:
        # Return empty list if database fails
        return []


@app.get("/stats")
def stats():
    """
    Get summary statistics of predictions.

    Includes:
    - Total predictions
    - High risk count
    - Low risk count
    - Average probability
    - Last model training time

    Returns:
    dict: Statistics summary
    """

    try:
        db = get_db()

        # MongoDB aggregation pipeline to calculate stats
        pipeline = [{"$group": {
            "_id": None,
            "total_predictions": {"$sum": 1},
            "high_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "high"]}, 1, 0]}},
            "low_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "low"]}, 1, 0]}},
            "average_probability": {"$avg": "$probability"},
        }}]

        # Execute aggregation
        result = list(db["predictions"].aggregate(pipeline))

        # Get last training timestamp
        last_run = db["training_runs"].find_one({}, {"_id": 0, "timestamp": 1}, sort=[("timestamp", -1)])

        if result:
            data = result[0]

            # Remove MongoDB internal ID
            data.pop("_id", None)

            # Round probability value
            data["average_probability"] = round(data["average_probability"], 4)

            # Add last training time
            data["last_trained"] = last_run["timestamp"].isoformat() if last_run else None

            return data

    except Exception:
        # Ignore errors
        pass

    # Default empty stats if something fails
    return {
        "total_predictions": 0,
        "high_risk_count": 0,
        "low_risk_count": 0,
        "average_probability": 0.0,
        "last_trained": None
    }
@app.on_event("startup")
def load_model_on_startup():
    global model
    from noshow_iq.model import load_model
    try:
        model = load_model()
        print("✅ Model loaded")
    except Exception as e:
        print("❌ Model load failed:", e)