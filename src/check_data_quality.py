
from pathlib import Path

import pandas as pd

# Find the project folder and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bs140513_032310.csv"


def main():
    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATA QUALITY CHECK")
    print("=" * 60)

    # 1. Check duplicate rows
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicate_count}")

    # 2. Check unique values in each column
    print("\nUnique values per column:")
    print(df.nunique())

    # 3. Check categorical values
    categorical_columns = [
        "age", "gender", "category", "merchant"
    ]

    for column in categorical_columns:
        if column in df.columns:
            print(f"\nUnique values in '{column}':")
            print(df[column].value_counts().head(15))

    # 4. Summarize transaction amounts
    print("\nTransaction amount summary:")
    print(df["amount"].describe())

    # 5. Check unusual transaction amounts
    print("\nMinimum transaction amount:", df["amount"].min())
    print("Maximum transaction amount:", df["amount"].max())
    print("Transactions with amount <= 0:", (df["amount"] <= 0).sum())

    # 6. Check target labels
    print("\nFraud labels:")
    print(df["fraud"].value_counts(dropna=False))

    print("\nData quality check completed.")


if __name__ == "__main__":
    main()
