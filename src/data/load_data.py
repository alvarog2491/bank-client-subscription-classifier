import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict
from config.config_loader import load_config


def load_train_data() -> pd.DataFrame:
    """Load training data from the raw data folder.

    Returns:
        pd.DataFrame: Training data
    """
    config = load_config()
    train_path = os.path.join(
        config["data"]["raw_data_path"], config["data"]["train_file"]
    )
    return pd.read_csv(train_path)


def load_test_data() -> pd.DataFrame:
    """Load test data from the raw data folder.

    Returns:
        pd.DataFrame: Test data
    """
    config = load_config()
    test_path = os.path.join(
        config["data"]["raw_data_path"], config["data"]["test_file"]
    )
    return pd.read_csv(test_path)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both training and test data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and test data
    """
    train_df = load_train_data()
    test_df = load_test_data()

    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    return train_df, test_df


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load already processed training and test data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed training and test data
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

    Returns:
        Dict[str, LabelEncoder]: Dictionary of label encoders
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
