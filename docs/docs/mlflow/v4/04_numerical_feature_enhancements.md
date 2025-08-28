# Numerical Feature Enhancements (Step 4)

## Overview

This step enhances numerical features beyond duration to address distribution issues and capture demographic patterns. The approach targets balance transformation, age segmentation, and campaign frequency optimization based on EDA insights.

## Key Findings

- Balance distribution spans wide range (-8,019 to 99,717) with negative values present
- Campaign frequency shows optimal performance at 2-3 contacts with diminishing returns beyond
- Age patterns reveal distinct demographic segments with different conversion behaviors
- Most numerical features exhibit right-skewed distributions requiring normalization

## Implementation

### 1. Balance Transformation

Applied arcsinh transformation to handle the wide range and negative values:

```python
def create_balance_transformations(train_df, test_df):
    """Apply arcsinh transformation to balance feature."""
    
    train_df["balance_arcsinh"] = np.arcsinh(train_df["balance"])
    test_df["balance_arcsinh"] = np.arcsinh(test_df["balance"])
    
    return train_df, test_df
```

**Arcsinh advantages over log1p:**
- Handles negative values directly (debt accounts)
- Symmetric transformation preserves sign
- Better numerical stability for optimization
- Similar variance reduction without domain restrictions

### 2. Age Demographic Segmentation

Created life-stage based segments:

```python
def create_age_segments(train_df, test_df):
    """Create age demographic segments."""
    
    age_bins = [0, 30, 50, 65, float("inf")]
    age_labels = ["young", "middle_aged", "mature", "senior"]
    
    train_df["age_group"] = pd.cut(
        train_df["age"], bins=age_bins, labels=age_labels, include_lowest=True
    )
    test_df["age_group"] = pd.cut(
        test_df["age"], bins=age_bins, labels=age_labels, include_lowest=True
    )
    
    return train_df, test_df
```

### 3. Campaign Frequency Patterns

Enhanced campaign features with engagement optimization:

```python
def create_campaign_patterns(train_df, test_df):
    """Create campaign frequency optimization features."""
    
    train_df["optimal_contact"] = (
        (train_df["campaign"] >= 2) & (train_df["campaign"] <= 3)
    ).astype(int)
    test_df["optimal_contact"] = (
        (test_df["campaign"] >= 2) & (test_df["campaign"] <= 3)
    ).astype(int)
    
    train_df["over_contacted"] = (train_df["campaign"] > 3).astype(int)
    test_df["over_contacted"] = (test_df["campaign"] > 3).astype(int)
    
    return train_df, test_df
```

### 4. Coordination Function

The main orchestrator coordinates all numerical enhancements:

```python
def apply_numerical_feature_enhancements(train_df, test_df):
    """Apply numerical feature enhancements based on EDA insights."""
    
    train_df, test_df = create_balance_transformations(train_df, test_df)
    train_df, test_df = create_age_segments(train_df, test_df)
    train_df, test_df = create_campaign_patterns(train_df, test_df)
    
    return train_df, test_df
```

## Features Created

| Feature | Type | Description |
|---------|------|-------------|
| `balance_arcsinh` | Numerical | Arcsinh-transformed balance handling negatives and outliers |
| `age_group` | Categorical | 4-tier demographic segments (young/middle_aged/mature/senior) |
| `optimal_contact` | Binary | Optimal campaign frequency flag (2-3 contacts) |
| `over_contacted` | Binary | Excessive contact attempts flag (>3 contacts) |

## Data Transformation Example

### Before Processing
| age | balance | campaign |
|-----|---------|----------|
| 25  | -150    | 1        |
| 45  | 5500    | 3        |
| 60  | 15000   | 5        |
| 70  | 2500    | 2        |

### After Processing
| age | balance | campaign | balance_arcsinh | age_group | optimal_contact | over_contacted |
|-----|---------|----------|-----------------|-----------|----------------|----------------|
| 25  | -150    | 1        | -5.00           | young     | 0              | 0              |
| 45  | 5500    | 3        | 8.61            | middle_aged | 1            | 0              |
| 60  | 15000   | 5        | 9.62            | mature    | 0              | 1              |
| 70  | 2500    | 2        | 7.82            | senior    | 1              | 0              |

## Expected Impact

- Handle distribution skewness through arcsinh transformation
- Capture demographic conversion patterns via age segmentation
- Optimize contact strategy with campaign frequency flags
- Maintain feature interpretability for business applications

## Results

### MLflow Performance
Numerical feature enhancements results:

<a href="../images/04_mlflow_num_engineering.png" target="_blank">
  <img src="../images/04_mlflow_num_engineering.png" alt="MLflow Results - Numerical Enhancements" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

**Single Model Performance (80/20 split):**
- **Test AUC: 0.9683**

**K-Fold Cross-Validation (5 folds):**
- **Average AUC: 0.9686**

Performance shows slight decline compared to categorical feature engineering, with consistent cross-validation results.

### Classification Metrics
Performance comparison with categorical feature engineering results:

<a href="../images/04_num_engineering_metrics.png" target="_blank">
  <img src="../images/04_num_engineering_metrics.png" alt="MLflow Numerical Enhancement Metrics" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

- **False Positives**: Increased from 3,848 to 5,252 (+1,404)
- **False Negatives**: Reduced from 5,657 to 4,452 (-1,205)
- **Trade-off**: Higher recall but lower precision compared to categorical engineering

The numerical enhancements show different error patterns with increased false positives but fewer false negatives.

### Kaggle Competition Results
Competition submission results:

<a href="../images/04_numerical_kaggle_submit.png" target="_blank">
  <img src="../images/04_numerical_kaggle_submit.png" alt="Kaggle Submission - Numerical Enhancements" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

- **Competition Score**: 0.96893
- **Performance**: Decline from previous categorical engineering score (0.96907)