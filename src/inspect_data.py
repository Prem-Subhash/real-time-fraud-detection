
from pathlib import Path

import pandas as pd

# Locate the project folder and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bs140513_032310.csv"


def main():
    print("=" * 60)
    print("BANKSIM DATASET INSPECTION")
    print("=" * 60)

    # Check whether the dataset exists
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at: {DATA_PATH}")
        return

    # Load the original CSV file
    df = pd.read_csv(DATA_PATH)

    # Display basic information
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Check whether the target column exists
    if "fraud" in df.columns:
        print("\nFraud label distribution:")
        print(df["fraud"].value_counts(dropna=False))

        print("\nFraud label percentages:")
        print(df["fraud"].value_counts(normalize=True, dropna=False) * 100)
    else:
        print("\nWARNING: The 'fraud' column was not found.")

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()
