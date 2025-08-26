import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Tuple
from .load_data import load_data
from config.config_loader import load_config


def preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply label encoding to categorical features and save encoders.

    Fits encoders on combined train+test data to ensure consistency.
    Saves processed datasets and encoders to configured paths.
    """
    config = load_config()

    # Load raw data
    train_df, test_df = load_data()

    # Get features configuration from config
    categorical_features = config["features"]["categorical_features"]
    features_to_drop = config["features"].get("features_to_drop", [])

    print(f"Label encoding categorical features: {categorical_features}")
    if features_to_drop:
        print(f"Features to drop: {features_to_drop}")

    # Initialize label encoders dictionary
    label_encoders = {}

    # Process each categorical feature
    for feature in categorical_features:
        if feature in train_df.columns:
            print(f"Processing feature: {feature}")

            # Create label encoder
            le = LabelEncoder()

            # Fit on combined data to ensure consistent encoding
            combined_values = pd.concat(
                [train_df[feature].astype(str), test_df[feature].astype(str)]
            ).unique()

            le.fit(combined_values)

            # Transform both datasets
            train_df[feature] = le.transform(train_df[feature].astype(str))
            test_df[feature] = le.transform(test_df[feature].astype(str))

            # Store encoder for later use
            label_encoders[feature] = le

            print(f"  - {feature}: {len(le.classes_)} unique values")
        else:
            print(f"Warning: Feature '{feature}' not found in data")

    # Drop specified features (only from training data
    # to preserve id for predictions)
    for feature in features_to_drop:
        if feature in train_df.columns:
            train_df = train_df.drop(columns=[feature])
            print(f"Dropped feature '{feature}' from training data")
            # Note: Keep id in test data for prediction output

    # Save processed data
    processed_path = config["data"]["processed_data_path"]
    os.makedirs(processed_path, exist_ok=True)

    train_processed_path = os.path.join(processed_path, "train_processed.csv")
    test_processed_path = os.path.join(processed_path, "test_processed.csv")
    encoders_path = os.path.join(processed_path, "label_encoders.pkl")

    # Save processed datasets
    train_df.to_csv(train_processed_path, index=False)
    test_df.to_csv(test_processed_path, index=False)

    # Save label encoders
    joblib.dump(label_encoders, encoders_path)

    print("\nProcessed data saved:")
    print(f"  - Training: {train_processed_path} (shape: {train_df.shape})")
    print(f"  - Test: {test_processed_path} (shape: {test_df.shape})")
    print(f"  - Label encoders: {encoders_path}")

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = preprocess_data()
    print("\nData preprocessing completed successfully!")
