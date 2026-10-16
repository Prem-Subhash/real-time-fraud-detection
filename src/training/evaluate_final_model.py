
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. LOCATE PROJECT AND LOAD DATASETS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Training dataset loaded:", train_df.shape)
print("Test dataset loaded:", test_df.shape)


# ============================================================
# 2. SEPARATE INPUT FEATURES AND TARGET
# ============================================================

X_train = train_df.drop(columns=["fraud", "customer"])
y_train = train_df["fraud"]

X_test = test_df.drop(columns=["fraud", "customer"])
y_test = test_df["fraud"]


# ============================================================
# 3. DEFINE PREPROCESSING
# ============================================================

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


# ============================================================
# 4. CREATE THE RANDOM FOREST PIPELINE
# ============================================================

model = Pipeline(
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
)


# ============================================================
# 5. TRAIN THE MODEL
# ============================================================

print("\nTraining Random Forest on the complete training dataset...")

model.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# 6. PREDICT ON THE UNTOUCHED TEST DATASET
# ============================================================

threshold = 0.60

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= threshold).astype(int)


# ============================================================
# 7. EVALUATE THE MODEL
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1],
).ravel()

print("\n" + "=" * 60)
print("FINAL RANDOM FOREST TEST EVALUATION")
print("=" * 60)

print(f"Test transactions: {len(X_test)}")
print(f"Actual fraud cases: {int(y_test.sum())}")
print(f"Decision threshold: {threshold:.2f}")

print("\nConfusion Matrix:")
print(f"True Negatives  (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives  (TP): {tp}")

print("\nEvaluation Metrics:")
print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(
    f"Precision: "
    f"{precision_score(y_test, predictions, zero_division=0):.4f}"
)
print(
    f"Recall:    "
    f"{recall_score(y_test, predictions, zero_division=0):.4f}"
)
print(f"F1 Score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions, zero_division=0))


# ============================================================
# 8. SAVE THE TRAINED MODEL AND THRESHOLD
# ============================================================

models_dir = PROJECT_ROOT / "models"

models_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "random_forest_pipeline.joblib"
threshold_path = models_dir / "decision_threshold.txt"

joblib.dump(model, model_path)

threshold_path.write_text(
    str(threshold),
    encoding="utf-8",
)

print("\n" + "=" * 60)
print("MODEL SAVING")
print("=" * 60)

print(f"Saved model: {model_path}")
print(f"Saved threshold: {threshold_path}")

print("\nFinal evaluation and model saving completed successfully.")
