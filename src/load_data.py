import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict
from config.config_loader import load_config


def load_train_data() -> pd.DataFrame:
    """Load training data from config-specified path."""
    config = load_config()
    train_path = os.path.join(
        config["data"]["raw_data_path"], config["data"]["train_file"]
    )
    return pd.read_csv(train_path)


def load_test_data() -> pd.DataFrame:
    """Load test data from config-specified path."""
    config = load_config()
    test_path = os.path.join(
        config["data"]["raw_data_path"], config["data"]["test_file"]
    )
    return pd.read_csv(test_path)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return both training and test datasets."""
    train_df = load_train_data()
    test_df = load_test_data()

    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    return train_df, test_df


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed datasets.

    Raises:
        FileNotFoundError: If processed files don't exist
    """
    config = load_config()
    processed_path = config["data"]["processed_data_path"]

    train_processed_path = os.path.join(processed_path, "train_processed.csv")
    test_processed_path = os.path.join(processed_path, "test_processed.csv")

    if not os.path.exists(train_processed_path) or not os.path.exists(
        test_processed_path
    ):
        raise FileNotFoundError(
            "Processed data files not found. Run preprocess_data() first."
        )

    train_df = pd.read_csv(train_processed_path)
    test_df = pd.read_csv(test_processed_path)

    print(f"Loaded processed training data: {train_df.shape}")
    print(f"Loaded processed test data: {test_df.shape}")

    return train_df, test_df


def load_label_encoders() -> Dict[str, LabelEncoder]:
    """Load saved label encoders.

    Raises:
        FileNotFoundError: If encoders file doesn't exist
    """
    config = load_config()
    encoders_path = os.path.join(
        config["data"]["processed_data_path"], "label_encoders.pkl"
    )

    if not os.path.exists(encoders_path):
        raise FileNotFoundError(
            "Label encoders file not found. Run preprocess_data() first."
        )

    return joblib.load(encoders_path)


if __name__ == "__main__":
    train_df, test_df = load_data()
    print("Data loaded successfully!")
    print(f"Training columns: {list(train_df.columns)}")
    print(f"Test columns: {list(test_df.columns)}")
