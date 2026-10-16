
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# 1. Locate and load the training dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

df = pd.read_csv(TRAIN_PATH)

# 2. Separate input features and target
X = df.drop(columns=["fraud", "customer"])
y = df["fraud"]

# 3. Create the same validation split used in earlier experiments
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 4. Define preprocessing
categorical_features = ["age", "gender", "merchant", "category"]
numerical_features = ["step", "amount"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features,
        ),
    ]
)

# 5. Define the two candidate models
models = {
    "Logistic Regression": (
        Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        0.90,
    ),
    "Random Forest": (
        Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        0.60,
    ),
}

# 6. Train and evaluate each model on the same validation set
results = []

for model_name, (model, threshold) in models.items():
    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ).ravel()

    results.append(
        {
            "Model": model_name,
            "Threshold": threshold,
            "Precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_val, predictions, zero_division=0
            ),
            "True Positives": tp,
            "False Positives": fp,
            "False Negatives": fn,
            "True Negatives": tn,
        }
    )

# 7. Display the comparison
results_df = pd.DataFrame(results)

print("\n" + "=" * 90)
print("MODEL COMPARISON - VALIDATION DATA")
print("=" * 90)
print(results_df.round(4).to_string(index=False))
