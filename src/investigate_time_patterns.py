
from pathlib import Path

import pandas as pd

# Locate the project folder and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bs140513_032310.csv"


def main():
    # Load the original dataset
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("FRAUD ANALYSIS OVER TIME")
    print("=" * 60)

    # 1. Inspect the time-step column
    print("\nTime-step summary:")
    print(df["step"].describe())

    print("\nNumber of unique time steps:")
    print(df["step"].nunique())

    # 2. Count transactions and fraud cases at each step
    time_summary = df.groupby("step").agg(
        total_transactions=("fraud", "count"),
        fraud_cases=("fraud", "sum")
    )

    # 3. Calculate the fraud rate at each time step
    time_summary["fraud_rate_percent"] = (
        time_summary["fraud_cases"]
        / time_summary["total_transactions"]
        * 100
    )

    print("\nFirst 10 time steps:")
    print(time_summary.head(10).round(2))

    # 4. Find time steps with the highest fraud rates
    print("\nTop 10 time steps by fraud rate:")
    print(
        time_summary.sort_values(
            "fraud_rate_percent", ascending=False
        ).head(10).round(2)
    )

    # 5. Compare fraud rates across broader time periods
    df["time_period"] = pd.cut(
        df["step"],
        bins=6,
        labels=False,
        include_lowest=True
    )

    period_summary = df.groupby("time_period", observed=True).agg(
        total_transactions=("fraud", "count"),
        fraud_cases=("fraud", "sum")
    )

    period_summary["fraud_rate_percent"] = (
        period_summary["fraud_cases"]
        / period_summary["total_transactions"]
        * 100
    )

    print("\nFraud summary across six time periods:")
    print(period_summary.round(2))

    print("\nTime-pattern investigation completed.")


if __name__ == "__main__":
    main()
