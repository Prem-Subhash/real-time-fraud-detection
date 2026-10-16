
from pathlib import Path

import pandas as pd

# Locate the project folder and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bs140513_032310.csv"


def main():
    # Load the original dataset
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("INVESTIGATING AGE AND GENDER CATEGORIES")
    print("=" * 60)

    # 1. Check the age categories
    print("\nAge category counts:")
    print(df["age"].value_counts(dropna=False))

    # 2. Check the gender categories
    print("\nGender category counts:")
    print(df["gender"].value_counts(dropna=False))

    # 3. Compare fraud counts across age categories
    print("\nFraud counts by age category:")
    print(pd.crosstab(df["age"], df["fraud"]))

    # 4. Calculate fraud percentages for each age category
    print("\nFraud percentage by age category:")
    age_fraud_rate = df.groupby("age")["fraud"].mean() * 100
    print(age_fraud_rate.round(2))

    # 5. Compare fraud counts across gender categories
    print("\nFraud counts by gender:")
    print(pd.crosstab(df["gender"], df["fraud"]))

    # 6. Calculate fraud percentages for each gender category
    print("\nFraud percentage by gender:")
    gender_fraud_rate = df.groupby("gender")["fraud"].mean() * 100
    print(gender_fraud_rate.round(2))

    print("\nCategory investigation completed.")


if __name__ == "__main__":
    main()
