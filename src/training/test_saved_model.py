
from pathlib import Path

import joblib
import pandas as pd


# 1. Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 2. Define the paths to the saved files
model_path = PROJECT_ROOT / "models" / "random_forest_pipeline.joblib"
threshold_path = PROJECT_ROOT / "models" / "decision_threshold.txt"

# 3. Load the saved model and threshold
model = joblib.load(model_path)
threshold = float(threshold_path.read_text(encoding="utf-8").strip())

print("Saved model loaded successfully.")
print(f"Decision threshold: {threshold}")

# 4. Load sample transactions from the test dataset
test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
test_df = pd.read_csv(test_path)

# Select two example transactions, without their labels or customer IDs
sample_transactions = test_df.drop(
    columns=["fraud", "customer"]
).head(2)

# 5. Predict fraud probabilities
probabilities = model.predict_proba(sample_transactions)[:, 1]

# 6. Convert probabilities into predictions
predictions = (probabilities >= threshold).astype(int)

# 7. Display the results
print("\nSample Prediction Results")
print("=" * 40)

for index, probability, prediction in zip(
    sample_transactions.index,
    probabilities,
    predictions,
):
    result = "FRAUD" if prediction == 1 else "LEGITIMATE"

    print(f"Transaction row: {index}")
    print(f"Fraud probability: {probability:.4f}")
    print(f"Prediction: {result}")
    print("-" * 40)
