# Feature Interactions (Step 5)

## Overview

This step creates feature interactions to capture non-linear relationships and domain-specific patterns that individual features cannot express. The approach targets combinations where business logic suggests multiplicative effects between features.

XGBoost optimization continues with 20 trials per run, focusing on extracting maximum value from feature combinations rather than exploring multiple model types.

## Key Findings

- Previous campaign success shows amplified effects when combined with call engagement
- Professional segments emerge from job-education combinations
- Seasonal timing effectiveness varies by contact method
- Interaction features can capture complex decision patterns

## Implementation

### 1. Previous Success + Duration Interactions

Created interactions between campaign history and engagement:

```python
def create_previous_success_duration_interactions(train_df, test_df):
    """Create interactions between previous campaign success and call duration."""
    
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
```

### 2. Job + Education Professional Segments

Enhanced professional segmentation through job-education combinations:

```python
def create_job_education_interactions(train_df, test_df):
    """Create professional segment interactions between job and education."""
    
    # High education professionals
    train_df["high_ed_professional"] = (
        ((train_df["job"].isin([1, 8])) & (train_df["education"] == 3)) |
        ((train_df["job"] == 9) & (train_df["education"].isin([2, 3])))
    ).astype(int)
    
    # Blue collar with higher education
    train_df["blue_collar_educated"] = (
        (train_df["job"] == 0) & (train_df["education"].isin([2, 3]))
    ).astype(int)
    
    return train_df, test_df
```

### 3. Seasonal + Contact Method Interactions

Combined timing and communication channel effectiveness:

```python
def create_seasonal_contact_interactions(train_df, test_df):
    """Create seasonal communication effectiveness interactions."""
    
    # High success months with cellular contact
    train_df["peak_month_cellular"] = (
        (train_df["month_success_group"] == 0) & (train_df["contact"] == 0)
    ).astype(int)
    
    # Medium success months with optimal contact type
    train_df["medium_month_contact"] = (
        (train_df["month_success_group"] == 2) & (train_df["contact"] != 2)
    ).astype(int)
    
    return train_df, test_df
```

### 4. Coordination Function

The main orchestrator applies all interaction types:

```python
def apply_feature_interactions(train_df, test_df):
    """Apply feature interactions based on domain insights."""
    
    train_df, test_df = create_previous_success_duration_interactions(train_df, test_df)
    train_df, test_df = create_job_education_interactions(train_df, test_df)
    train_df, test_df = create_seasonal_contact_interactions(train_df, test_df)
    
    return train_df, test_df
```

## Features Created

| Feature | Type | Description |
|---------|------|-------------|
| `prev_success_high_engagement` | Binary | Previous success combined with high engagement calls |
| `prev_success_duration_log` | Numerical | Previous success weighted by call duration |
| `high_ed_professional` | Binary | Professional segments with higher education |
| `blue_collar_educated` | Binary | Blue-collar workers with higher education |
| `peak_month_cellular` | Binary | High success months with cellular contact |
| `medium_month_contact` | Binary | Medium success months with optimal contact |

## Data Transformation Example

### Before Processing
| previous_success | duration_high_engagement | job | education | month_success_group | contact |
|------------------|-------------------------|-----|-----------|-------------------|---------|
| 1                | 1                       | 1   | 3         | 0                 | 0       |
| 0                | 1                       | 0   | 2         | 1                 | 1       |
| 1                | 0                       | 8   | 3         | 2                 | 0       |

### After Processing
| previous_success | duration_high_engagement | prev_success_high_engagement | high_ed_professional | peak_month_cellular |
|------------------|-------------------------|------------------------------|---------------------|---------------------|
| 1                | 1                       | 1                            | 1                   | 1                   |
| 0                | 1                       | 0                            | 0                   | 0                   |
| 1                | 0                       | 0                            | 1                   | 0                   |

## Expected Impact

- Capture multiplicative effects between campaign history and engagement
- Identify high-value professional segments through education-job combinations
- Optimize timing and communication channel strategies
- Enable more nuanced customer segmentation

## Results

### MLflow Performance
Feature interactions results:

**Single Model Performance (80/20 split):**
- **Test AUC: 0.96840**

**K-Fold Cross-Validation (5 folds):**
- **Average AUC: 0.9686**

Performance comparable to numerical enhancements, with consistent cross-validation results.

### Classification Metrics
Performance comparison with numerical enhancements results:

- **False Positives**: Reduced from 5,252 to 5,120 (-132)
- **False Negatives**: Increased from 4,452 to 4,547 (+95)
- **Trade-off**: Lower precision with higher recall

Feature interactions show minor changes in error patterns compared to numerical enhancements.

### Kaggle Competition Results
Competition submission results:

- **Competition Score**: 0.96916
- **Leaderboard Position**: 1152