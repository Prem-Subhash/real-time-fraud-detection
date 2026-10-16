
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load the prepared dataset
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fraud_transactions.csv"

# Load the data
df = pd.read_csv(DATA_PATH)

# Separate input features and target
X = df.drop("fraud", axis=1)
y = df["fraud"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display the results
print("Training input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

# Save the split datasets
processed_path = PROJECT_ROOT / "data" / "processed"
processed_path.mkdir(parents=True, exist_ok=True)

X_train.assign(fraud=y_train).to_csv(
    processed_path / "train.csv", index=False
)
X_test.assign(fraud=y_test).to_csv(
    processed_path / "test.csv", index=False
)

print("\nTraining and testing datasets saved successfully.")
