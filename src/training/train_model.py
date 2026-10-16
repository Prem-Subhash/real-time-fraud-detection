
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"


def main():
    # 1. Load the existing training data
    df = pd.read_csv(TRAIN_PATH)

    # 2. Separate input features and target
    X = df.drop(columns=["fraud", "customer"])
    y = df["fraud"]

    # 3. Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 4. Identify categorical and numerical columns
    categorical_columns = ["age", "gender", "merchant", "category"]
    numerical_columns = ["step", "amount"]

    # 5. Preprocess the input features
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_columns
            )
        ]
    )

    # 6. Create the Logistic Regression pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    # 7. Train the model
    model.fit(X_train, y_train)

    # 8. Get fraud probabilities for validation data
    fraud_probabilities = model.predict_proba(X_val)[:, 1]

    print("=" * 60)
    print("LOGISTIC REGRESSION - THRESHOLD EVALUATION")
    print("=" * 60)

    print(f"\nValidation set size: {len(y_val)}")
    print(f"Actual fraud cases: {(y_val == 1).sum()}")

    # 9. Compare four decision thresholds
    thresholds = [0.50, 0.70, 0.80, 0.90]

    for threshold in thresholds:

        # Convert probabilities into predictions
        y_pred = (fraud_probabilities >= threshold).astype(int)

        # Calculate confusion matrix values
        tn, fp, fn, tp = confusion_matrix(
            y_val,
            y_pred,
            labels=[0, 1]
        ).ravel()

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(
            y_val, y_pred, zero_division=0
        )
        recall = recall_score(
            y_val, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_val, y_pred, zero_division=0
        )

        # Display results
        print("\n" + "=" * 60)
        print(f"THRESHOLD: {threshold}")
        print("-" * 60)

        print("Confusion Matrix:")
        print(f"True Negatives  (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Positives  (TP): {tp}")

        print("\nEvaluation Metrics:")
        print(f"Accuracy:        {accuracy:.4f}")
        print(f"Fraud Precision: {precision:.4f}")
        print(f"Fraud Recall:    {recall:.4f}")
        print(f"Fraud F1 Score:  {f1:.4f}")

    print("\nThreshold evaluation completed.")


if __name__ == "__main__":
    main()
