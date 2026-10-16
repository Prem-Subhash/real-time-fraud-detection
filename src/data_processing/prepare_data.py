
from pathlib import Path
import pandas as pd

# Find the project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define the input and output paths
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bs140513_032310.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fraud_transactions.csv"


def main():
    # Load the original dataset
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    # Step 1: Check missing values
    print("\nMissing values:")
    print(df.isnull().sum().sum())

    if df.isnull().sum().sum() > 0:
        raise ValueError("Missing values found. Investigate before proceeding.")

    # Step 2: Check and remove exact duplicate rows
    duplicate_count = df.duplicated().sum()
    print("\nExact duplicate rows:", duplicate_count)

    df = df.drop_duplicates()

    # Step 3: Validate the target column
    valid_target_values = {0, 1}

    if not set(df["fraud"].unique()).issubset(valid_target_values):
        raise ValueError("Invalid values found in the fraud column.")

    print("\nTarget validation passed.")

    # Step 4: Identify constant columns
    constant_columns = [
        column for column in df.columns
        if df[column].nunique() == 1
    ]

    print("\nConstant columns excluded:")
    print(constant_columns)

    # Step 5: Exclude constant columns from the prepared dataset
    df = df.drop(columns=constant_columns)

    # Step 6: Create the output folder if needed
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step 7: Save the prepared dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Step 8: Verify the saved dataset
    prepared_df = pd.read_csv(OUTPUT_PATH)

    print("\nOriginal dataset shape: (594643, 10)")
    print("Prepared dataset shape:", prepared_df.shape)

    print("\nPrepared columns:")
    print(prepared_df.columns.tolist())

    print("\nPrepared target counts:")
    print(prepared_df["fraud"].value_counts().sort_index())

    print("\nSaved to:", OUTPUT_PATH)
    print("\nData preparation completed successfully.")


if __name__ == "__main__":
    main()
