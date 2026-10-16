
from pathlib import Path

import pandas as pd

# Locate the project folder and original dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bs140513_032310.csv"


def main():
    # Load the original transaction data
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("FRAUD ANALYSIS BY TRANSACTION CATEGORY")
    print("=" * 60)

    # Count transactions in each category
    print("\nTransaction count by category:")
    print(df["category"].value_counts())

    # Count legitimate and fraudulent transactions per category
    print("\nFraud counts by category:")
    fraud_counts = pd.crosstab(df["category"], df["fraud"])
    print(fraud_counts)

    # Calculate fraud percentage in each category
    print("\nFraud rate by category (%):")
    fraud_rate = df.groupby("category")["fraud"].mean() * 100
    print(fraud_rate.sort_values(ascending=False).round(2))

    # Combine counts and fraud rates into one summary
    summary = df.groupby("category").agg(
        total_transactions=("fraud", "count"),
        fraud_cases=("fraud", "sum"),
        average_amount=("amount", "mean")
    )

    summary["fraud_rate_percent"] = (
        summary["fraud_cases"] / summary["total_transactions"] * 100
    )

    print("\nComplete category summary:")
    print(
        summary.sort_values(
            "fraud_rate_percent", ascending=False
        ).round(2)
    )

    print("\nCategory investigation completed.")


if __name__ == "__main__":
    main()
