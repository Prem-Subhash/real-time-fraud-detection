
from pathlib import Path

import pandas as pd

# Locate the project and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bs140513_032310.csv"


def main():
    # Load the original dataset
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("INVESTIGATING UNUSUAL TRANSACTION AMOUNTS")
    print("=" * 60)

    # Select transactions with zero or negative amounts
    unusual = df[df["amount"] <= 0]

    print(f"\nTotal unusual transactions: {len(unusual)}")

    # Show the transaction amounts and fraud labels
    print("\nFraud label distribution:")
    print(unusual["fraud"].value_counts())

    # Calculate the fraud percentage in these transactions
    print("\nFraud percentage:")
    print(unusual["fraud"].value_counts(normalize=True) * 100)

    # Compare unusual transactions with the full dataset
    print("\nComparison with all transactions:")
    print(
        df.groupby("fraud")["amount"]
        .agg(["count", "mean", "min", "max"])
    )

    # Display a sample of unusual transactions
    print("\nSample unusual transactions:")
    print(
        unusual[
            ["step", "customer", "category", "amount", "fraud"]
        ].head(10).to_string(index=False)
    )

    print("\nInvestigation completed.")


if __name__ == "__main__":
    main()
