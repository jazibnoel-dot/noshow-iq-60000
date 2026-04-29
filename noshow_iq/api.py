import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="NoShowIQ",
    description="Predicts which patients will miss their appointment",
    version="0.1.0",
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")


def get_db():
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return client["noshowiq"]


def get_model():
    from noshow_iq.model import load_model
    return load_model()


model = None


class AppointmentInput(BaseModel):
    age: int
    scholarship: int
    hipertension: int
    diabetes: int
    alcoholism: int
    handcap: int
    sms_received: int
    days_in_advance: int
    hour_of_booking: int


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(record: AppointmentInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    from noshow_iq.model import predict as model_predict
    input_dict = record.dict()
    df = pd.DataFrame([input_dict])
    result = model_predict(model, df)
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
        pass
    return result


@app.get("/history")
def history():
    try:
        db = get_db()
        docs = list(db["predictions"].find({}, {"_id": 0}).sort("timestamp", -1).limit(20))
        return docs
    except Exception:
        return []


@app.get("/stats")
def stats():
    try:
        db = get_db()
        pipeline = [{"$group": {
            "_id": None,
            "total_predictions": {"$sum": 1},
            "high_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "high"]}, 1, 0]}},
            "low_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "low"]}, 1, 0]}},
            "average_probability": {"$avg": "$probability"},
        }}]
        result = list(db["predictions"].aggregate(pipeline))
        last_run = db["training_runs"].find_one({}, {"_id": 0, "timestamp": 1}, sort=[("timestamp", -1)])
        if result:
            data = result[0]
            data.pop("_id", None)
            data["average_probability"] = round(data["average_probability"], 4)
            data["last_trained"] = last_run["timestamp"].isoformat() if last_run else None
            return data
    except Exception:
        pass
    return {"total_predictions": 0, "high_risk_count": 0, "low_risk_count": 0,
            "average_probability": 0.0, "last_trained": None}