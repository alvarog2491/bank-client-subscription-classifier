import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Tuple
from load_data import load_data
from config.config_loader import load_config


def create_job_conversion_groups(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create job category groups based on conversion rates from EDA."""
    print("  Creating job conversion groups...")

    high_conversion_jobs = ["student", "retired"]  # 34.1%, 30.8%
    medium_conversion_jobs = [
        "self-employed",
        "unemployed",
        "management",
    ]  # ~12-15%

    def categorize_job(job):
        if job in high_conversion_jobs:
            return "high_conversion"
        elif job in medium_conversion_jobs:
            return "medium_conversion"
        else:
            return "low_conversion"

    train_df["job_conversion_group"] = train_df["job"].map(categorize_job)
    test_df["job_conversion_group"] = test_df["job"].map(categorize_job)

    return train_df, test_df


def create_monthly_success_patterns(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create monthly success pattern features based on EDA seasonal analysis."""
    print("  Creating monthly success patterns...")

    high_success_months = ["mar"]  # 57.1% success
    medium_success_months = ["sep", "oct", "dec"]  # ~15-20% success

    def categorize_month(month):
        if month in high_success_months:
            return "high_success"
        elif month in medium_success_months:
            return "medium_success"
        else:
            return "low_success"

    train_df["month_success_group"] = train_df["month"].map(categorize_month)
    test_df["month_success_group"] = test_df["month"].map(categorize_month)

    return train_df, test_df


def create_previous_campaign_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create previous campaign success features (76.4% conversion rate)."""
    print("  Creating previous campaign features...")

    train_df["previous_success"] = (train_df["poutcome"] == "success").astype(
        int
    )
    test_df["previous_success"] = (test_df["poutcome"] == "success").astype(
        int
    )

    return train_df, test_df


def create_high_value_segments(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create high-value customer segment flags."""
    print("  Creating high-value segments...")

    high_conversion_jobs = ["student", "retired"]
    train_df["high_value_segment"] = (
        train_df["job"].isin(high_conversion_jobs).astype(int)
    )
    test_df["high_value_segment"] = (
        test_df["job"].isin(high_conversion_jobs).astype(int)
    )

    return train_df, test_df


def apply_categorical_feature_engineering(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply categorical feature engineering based on EDA insights.

    Orchestrates all categorical feature engineering steps:
    - Job category groups by conversion rates
    - Seasonal/monthly success patterns
    - Previous campaign success encoding
    - High-value customer segments
    """
    print("Applying categorical feature engineering...")

    # Apply each categorical feature engineering step
    train_df, test_df = create_job_conversion_groups(train_df, test_df)
    train_df, test_df = create_monthly_success_patterns(train_df, test_df)
    train_df, test_df = create_previous_campaign_features(train_df, test_df)
    train_df, test_df = create_high_value_segments(train_df, test_df)

    print("Categorical feature engineering completed:")
    print(
        "  - job_conversion_group: high/medium/low conversion job categories"
    )
    print("  - month_success_group: high/medium/low success month categories")
    print("  - previous_success: binary flag for previous campaign success")
    print("  - high_value_segment: binary flag for high-conversion jobs")

    return train_df, test_df


def apply_duration_feature_treatment(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply duration feature treatment based on EDA insights.

    Duration shows strongest correlation (r=0.519) with target:
    - Subscribers: avg 525 seconds
    - Non-subscribers: avg 212 seconds

    Creates duration bins and features to capture this relationship.
    """
    print("Applying duration feature treatment...")

    # Create duration bins based on quartiles and business logic
    duration_bins = [0, 120, 300, 600, float("inf")]
    duration_labels = ["very_short", "short", "medium", "long"]

    train_df["duration_bin"] = pd.cut(
        train_df["duration"],
        bins=duration_bins,
        labels=duration_labels,
        include_lowest=True,
    )
    test_df["duration_bin"] = pd.cut(
        test_df["duration"],
        bins=duration_bins,
        labels=duration_labels,
        include_lowest=True,
    )

    # Create log transformation of duration to handle skewness
    train_df["duration_log"] = np.log1p(train_df["duration"])
    test_df["duration_log"] = np.log1p(test_df["duration"])

    # Create binary feature for high engagement calls (>300 seconds)
    train_df["duration_high_engagement"] = (train_df["duration"] > 300).astype(
        int
    )
    test_df["duration_high_engagement"] = (test_df["duration"] > 300).astype(
        int
    )

    print("Duration features created:")
    print(
        "  - duration_bin: categorical bins (very_short, short, medium, long)"
    )
    print("  - duration_log: log-transformed duration")
    print("  - duration_high_engagement: binary flag for calls > 300 seconds")

    return train_df, test_df


def create_balance_transformations(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply arcsinh transformation to balance feature."""
    print("  Creating balance transformations...")
    
    train_df["balance_arcsinh"] = np.arcsinh(train_df["balance"])
    test_df["balance_arcsinh"] = np.arcsinh(test_df["balance"])
    
    return train_df, test_df


def create_age_segments(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create age demographic segments."""
    print("  Creating age segments...")
    
    age_bins = [0, 30, 50, 65, float("inf")]
    age_labels = ["young", "middle_aged", "mature", "senior"]
    
    train_df["age_group"] = pd.cut(
        train_df["age"], bins=age_bins, labels=age_labels, include_lowest=True
    )
    test_df["age_group"] = pd.cut(
        test_df["age"], bins=age_bins, labels=age_labels, include_lowest=True
    )
    
    return train_df, test_df


def create_campaign_patterns(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create campaign frequency optimization features."""
    print("  Creating campaign patterns...")
    
    train_df["optimal_contact"] = (
        (train_df["campaign"] >= 2) & (train_df["campaign"] <= 3)
    ).astype(int)
    test_df["optimal_contact"] = (
        (test_df["campaign"] >= 2) & (test_df["campaign"] <= 3)
    ).astype(int)
    
    train_df["over_contacted"] = (train_df["campaign"] > 3).astype(int)
    test_df["over_contacted"] = (test_df["campaign"] > 3).astype(int)
    
    return train_df, test_df


def apply_numerical_feature_enhancements(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply numerical feature enhancements based on EDA insights."""
    print("Applying numerical feature enhancements...")
    
    train_df, test_df = create_balance_transformations(train_df, test_df)
    train_df, test_df = create_age_segments(train_df, test_df)
    train_df, test_df = create_campaign_patterns(train_df, test_df)
    
    print("Numerical feature enhancements completed:")
    print("  - balance_arcsinh: arcsinh-transformed balance")
    print("  - age_group: demographic segments (young/middle_aged/mature/senior)")
    print("  - optimal_contact: optimal campaign frequency flag (2-3 contacts)")
    print("  - over_contacted: excessive contact attempts flag (>3)")
    
    return train_df, test_df


def create_previous_success_duration_interactions(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create interactions between previous campaign success and call duration."""
    print("  Creating previous success + duration interactions...")
    
    # Previous success with high engagement duration
    train_df["prev_success_high_engagement"] = (
        train_df["previous_success"] * train_df["duration_high_engagement"]
    )
    test_df["prev_success_high_engagement"] = (
        test_df["previous_success"] * test_df["duration_high_engagement"]
    )
    
    # Previous success with duration quartiles
    train_df["prev_success_duration_log"] = (
        train_df["previous_success"] * train_df["duration_log"]
    )
    test_df["prev_success_duration_log"] = (
        test_df["previous_success"] * test_df["duration_log"]
    )
    
    return train_df, test_df


def create_job_education_interactions(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create professional segment interactions between job and education."""
    print("  Creating job + education interactions...")
    
    # High education professionals (management, technician with higher education)
    train_df["high_ed_professional"] = (
        ((train_df["job"].isin([1, 8])) & (train_df["education"] == 3)) |  # management, technician with tertiary
        ((train_df["job"] == 9) & (train_df["education"].isin([2, 3])))     # unemployed with secondary/tertiary
    ).astype(int)
    
    test_df["high_ed_professional"] = (
        ((test_df["job"].isin([1, 8])) & (test_df["education"] == 3)) |
        ((test_df["job"] == 9) & (test_df["education"].isin([2, 3])))
    ).astype(int)
    
    # Blue collar with higher education (potential career transition)
    train_df["blue_collar_educated"] = (
        (train_df["job"] == 0) & (train_df["education"].isin([2, 3]))  # blue-collar with secondary/tertiary
    ).astype(int)
    
    test_df["blue_collar_educated"] = (
        (test_df["job"] == 0) & (test_df["education"].isin([2, 3]))
    ).astype(int)
    
    return train_df, test_df


def create_seasonal_contact_interactions(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create seasonal communication effectiveness interactions."""
    print("  Creating month + contact interactions...")
    
    # High success months with cellular contact
    train_df["peak_month_cellular"] = (
        (train_df["month_success_group"] == 0) & (train_df["contact"] == 0)  # high success month + cellular
    ).astype(int)
    
    test_df["peak_month_cellular"] = (
        (test_df["month_success_group"] == 0) & (test_df["contact"] == 0)
    ).astype(int)
    
    # Medium success months with optimal contact type
    train_df["medium_month_contact"] = (
        (train_df["month_success_group"] == 2) & (train_df["contact"] != 2)  # medium month + not unknown contact
    ).astype(int)
    
    test_df["medium_month_contact"] = (
        (test_df["month_success_group"] == 2) & (test_df["contact"] != 2)
    ).astype(int)
    
    return train_df, test_df


def apply_feature_interactions(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply feature interactions based on domain insights."""
    print("Applying feature interactions...")
    
    train_df, test_df = create_previous_success_duration_interactions(train_df, test_df)
    train_df, test_df = create_job_education_interactions(train_df, test_df)
    train_df, test_df = create_seasonal_contact_interactions(train_df, test_df)
    
    print("Feature interactions completed:")
    print("  - prev_success_high_engagement: previous success with high engagement calls")
    print("  - prev_success_duration_log: previous success weighted by call duration")
    print("  - high_ed_professional: professional segments with higher education")
    print("  - blue_collar_educated: blue-collar workers with higher education")
    print("  - peak_month_cellular: high success months with cellular contact")
    print("  - medium_month_contact: medium success months with optimal contact")
    
    return train_df, test_df


def preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply label encoding to categorical features and save encoders.

    Fits encoders on combined train+test data to ensure consistency.
    Saves processed datasets and encoders to configured paths.
    """
    config = load_config()

    # Load raw data
    train_df, test_df = load_data()

    # Apply duration feature treatment (step 2)
    train_df, test_df = apply_duration_feature_treatment(train_df, test_df)

    # Apply categorical feature engineering (step 3)
    train_df, test_df = apply_categorical_feature_engineering(
        train_df, test_df
    )

    # Apply numerical feature enhancements (step 4)
    train_df, test_df = apply_numerical_feature_enhancements(
        train_df, test_df
    )

    # Apply feature interactions (step 5)
    train_df, test_df = apply_feature_interactions(
        train_df, test_df
    )

    # Get features configuration from config
    categorical_features = config["features"]["categorical_features"]
    # Add new engineered features to categorical features for encoding
    categorical_features = categorical_features + [
        "duration_bin",
        "job_conversion_group",
        "month_success_group",
        "age_group",
    ]
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
