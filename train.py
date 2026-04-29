import os
from dotenv import load_dotenv
from noshow_iq.preprocess import load_and_clean, get_features_and_target
from noshow_iq.model import train

"""
Main script to:
1. Load environment variables (like database connection).
2. Load and clean the dataset.
3. Prepare features and target variable.
4. Optionally connect to MongoDB.
5. Train the machine learning model.
6. Save the trained model.

This script is usually run once to train the model.
"""

# Load environment variables from a .env file (if available)
load_dotenv()

# Path to the dataset file
DATASET_PATH = "data/raw/KaggleV2-May-2016.csv"

# Step 1: Load and clean dataset
print("Loading dataset...")
df = load_and_clean(DATASET_PATH)

# Show number of rows after cleaning
print(f"Rows after cleaning: {len(df)}")

# Step 2: Separate features (X) and target (y)
X, y = get_features_and_target(df)

# Show percentage of patients who did not show up
print(f"No-show rate: {y.mean():.1%}")

# Step 3: Setup database connection (optional)
db = None
mongo_uri = os.getenv("MONGO_URI")

# If MongoDB URI is provided, connect to database
if mongo_uri:
    from pymongo import MongoClient
    client = MongoClient(mongo_uri)
    db = client["noshowiq"]  # database name
    print("Connected to MongoDB.")

# Step 4: Train the model
print("Training model - this takes 1-2 minutes...")

# Train model and optionally store training info in DB
model, X_test, y_test = train(X, y, db=db)

# Final message after training is complete
print("Done! Model saved to models/noshow_model.joblib")