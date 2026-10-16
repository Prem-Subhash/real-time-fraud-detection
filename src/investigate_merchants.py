
from pathlib import Path

import pandas as pd

# Locate the project folder and original dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bs140513_032310.csv"


def main():
    # Load the transaction dataset
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("MERCHANT-LEVEL FRAUD ANALYSIS")
    print("=" * 60)

    # 1. Count transactions for each merchant
    print("\nTransaction count by merchant:")
    print(df["merchant"].value_counts().head(15))

    # 2. Calculate transaction and fraud counts per merchant
    merchant_summary = df.groupby("merchant").agg(
        total_transactions=("fraud", "count"),
        fraud_cases=("fraud", "sum"),
        average_amount=("amount", "mean")
    )

    # 3. Calculate the fraud rate for each merchant
    merchant_summary["fraud_rate_percent"] = (
        merchant_summary["fraud_cases"]
        / merchant_summary["total_transactions"]
        * 100
    )

    # 4. Show merchants with the highest fraud rates
    print("\nTop 15 merchants by fraud rate:")
    print(
        merchant_summary.sort_values(
            "fraud_rate_percent", ascending=False
        ).head(15).round(2)
    )

    # 5. Show merchants with the most fraud cases
    print("\nTop 15 merchants by number of fraud cases:")
    print(
        merchant_summary.sort_values(
            "fraud_cases", ascending=False
        ).head(15).round(2)
    )

    print("\nMerchant investigation completed.")


if __name__ == "__main__":
    main()
