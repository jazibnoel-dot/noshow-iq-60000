import os
from dotenv import load_dotenv
from noshow_iq.preprocess import load_and_clean, get_features_and_target
from noshow_iq.model import train

load_dotenv()

DATASET_PATH = "data/raw/KaggleV2-May-2016.csv"
print("Loading dataset...")
df = load_and_clean(DATASET_PATH)
print(f"Rows after cleaning: {len(df)}")
X, y = get_features_and_target(df)
print(f"No-show rate: {y.mean():.1%}")

db = None
mongo_uri = os.getenv("MONGO_URI")
if mongo_uri:
    from pymongo import MongoClient
    client = MongoClient(mongo_uri)
    db = client["noshowiq"]
    print("Connected to MongoDB.")

print("Training model - this takes 1-2 minutes...")
model, X_test, y_test = train(X, y, db=db)
print("Done! Model saved to models/noshow_model.joblib")