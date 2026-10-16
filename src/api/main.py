
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# 1. Locate the project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 2. Locate the saved model and threshold
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_pipeline.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "decision_threshold.txt"

# 3. Load the trained model and threshold
model = joblib.load(MODEL_PATH)

threshold = float(
    THRESHOLD_PATH.read_text(encoding="utf-8").strip()
)

# 4. Create the FastAPI application
app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Predict fraudulent transactions using Random Forest.",
    version="1.0.0",
)


# 5. Define the transaction input format
class Transaction(BaseModel):
    step: int = Field(ge=0)
    age: str
    gender: str
    merchant: str
    category: str
    amount: float = Field(ge=0)


# 6. Create a basic health-check endpoint
@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "model": "Random Forest",
    }


# 7. Create the fraud prediction endpoint
@app.post("/predict")
def predict_fraud(transaction: Transaction):

    # Convert incoming transaction into a DataFrame
    transaction_df = pd.DataFrame(
        [transaction.model_dump()]
    )

    # Calculate the fraud probability
    fraud_probability = float(
        model.predict_proba(transaction_df)[0][1]
    )

    # Apply the saved threshold
    prediction = int(fraud_probability >= threshold)

    # Convert numeric prediction into a readable result
    result = "FRAUD" if prediction == 1 else "LEGITIMATE"

    # Return the prediction response
    return {
        "fraud_probability": round(fraud_probability, 4),
        "threshold": threshold,
        "prediction": prediction,
        "result": result,
    }
