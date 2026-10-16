
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load the training and testing datasets
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


def main():
    # Read the datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Separate input features (X) and target (y)
    X_train = train_df.drop(columns=["fraud", "customer"])
    y_train = train_df["fraud"]

    X_test = test_df.drop(columns=["fraud", "customer"])
    y_test = test_df["fraud"]

    # Identify categorical and numerical columns
    categorical_columns = ["age", "gender", "merchant", "category"]
    numerical_columns = ["step", "amount"]

    # Create the preprocessing transformer
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            ("numerical", "passthrough", numerical_columns)
        ]
    )

    # Learn the encodings from training data and transform it
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply the same encodings to testing data
    X_test_processed = preprocessor.transform(X_test)

    # Display the results
    print("Original training shape:", X_train.shape)
    print("Processed training shape:", X_train_processed.shape)

    print("\nOriginal testing shape:", X_test.shape)
    print("Processed testing shape:", X_test_processed.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting target distribution:")
    print(y_test.value_counts().sort_index())

    print("\nFeature preprocessing completed successfully.")


if __name__ == "__main__":
    main()
