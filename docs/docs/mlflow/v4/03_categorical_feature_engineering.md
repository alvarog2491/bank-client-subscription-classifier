# Categorical Feature Engineering (Step 3)

## Overview

The EDA analysis revealed substantial differences in conversion rates across categorical variables, particularly job types and seasonal patterns. This step transforms these insights into engineered features that capture the underlying segmentation patterns.

## Key Findings from EDA

Students and retirees show significantly higher conversion rates at 34.1% and 30.8% respectively, compared to other job categories. March campaigns achieve remarkable success at 57.1% while August drops to just 4.7%. Additionally, customers with previous successful campaigns convert at 76.4%, indicating strong relationship continuity.

## Feature Engineering Approach

### Job Performance Segmentation

Created three-tier job groupings based on observed conversion patterns:

- High performers: students and retirees
- Medium performers: self-employed, unemployed, management roles
- Standard performers: remaining job categories

### Seasonal Campaign Effectiveness  

Grouped months by historical success rates:

- Peak performance: March campaigns
- Solid performance: September, October, December
- Standard performance: remaining months

### Campaign History Encoding

Previous campaign outcomes serve as strong predictors, with successful prior interactions showing 76.4% conversion rates. This relationship warranted direct binary encoding.

### Customer Segment Identification

Combined high-performing job categories into a single high-value customer flag for targeting purposes.

## Implementation

The approach uses modular functions for each feature type, allowing independent testing and validation. Each transformation maintains the original categorical values while adding the derived features.

```python
def create_job_conversion_groups(train_df, test_df):
    """Create job category groups based on conversion rates from EDA."""
    high_conversion_jobs = ["student", "retired"]  # 34.1%, 30.8%
    medium_conversion_jobs = ["self-employed", "unemployed", "management"]
    
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
```

The main orchestrator function coordinates all transformations:

```python
def apply_categorical_feature_engineering(train_df, test_df):
    """Apply categorical feature engineering based on EDA insights."""
    
    train_df, test_df = create_job_conversion_groups(train_df, test_df)
    train_df, test_df = create_monthly_success_patterns(train_df, test_df)
    train_df, test_df = create_previous_campaign_features(train_df, test_df)
    train_df, test_df = create_high_value_segments(train_df, test_df)
    
    return train_df, test_df
```

## Data Transformation Example

### Before Processing
| job | month | poutcome | y | Customer Profile |
|-----|-------|----------|---|------------------|
| technician | aug | unknown | 0 | Blue-collar worker, August campaign |
| student | may | unknown | 0 | Student, May campaign |
| blue-collar | jun | success | 1 | Previous successful contact |
| retired | mar | failure | 1 | Retired customer, March timing |

### After Processing
| job | month | poutcome | job_conversion_group | month_success_group | previous_success | high_value_segment | y | Insights |
|-----|-------|----------|---------------------|--------------------|-----------------|--------------------|---|-----------|
| technician | aug | unknown | low_conversion | low_success | 0 | 0 | 0 | Standard segment, poor timing |
| student | may | unknown | high_conversion | low_success | 0 | 1 | 0 | High-value segment, average timing |
| blue-collar | jun | success | low_conversion | low_success | 1 | 0 | 1 | Previous relationship success |
| retired | mar | failure | high_conversion | high_success | 0 | 1 | 1 | Premium segment, peak timing |


## Features Created

| Feature | Type | Description |
|---------|------|-------------|
| `job_conversion_group` | Categorical | 3-tier grouping (high/medium/low conversion) based on job success rates |
| `month_success_group` | Categorical | 3-tier seasonal grouping based on campaign timing effectiveness |
| `previous_success` | Binary | Flag for customers with previous successful campaigns (76.4% conversion) |
| `high_value_segment` | Binary | Flag for high-performing job categories (students, retirees) |

## Expected Impact

- **Capture conversion patterns**: Job and seasonal groupings reveal underlying success drivers
- **Leverage relationship history**: Previous campaign success as strong predictor
- **Enable targeted segmentation**: High-value customer identification for focused campaigns
- **Preserve original features**: Keep original categorical values alongside engineered features

## Results

### MLflow Performance
Categorical feature engineering results:

<a href="../images/03_mlflow_categorical_engineering.png" target="_blank">
  <img src="../images/03_mlflow_categorical_engineering.png" alt="MLflow Results - Categorical Engineering" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

**Single Model Performance (80/20 split):**
- **Test AUC: 0.9685**

**K-Fold Cross-Validation (5 folds):**
- **Average AUC: 0.9687**

Performance is comparable to duration feature treatment, with consistent cross-validation results.

### Classification Metrics
Performance comparison with duration feature treatment results:

<a href="../images/03_mlflow_cat_engineering_metrics.png" target="_blank">
  <img src="../images/03_mlflow_cat_engineering_metrics.png" alt="MLflow Categorical Engineering Metrics" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

- **False Positives**: Reduced from 5,374 to 3,848 (-1,526)
- **False Negatives**: Increased from 4,409 to 5,657 (+1,248)
- **Trade-off**: Higher precision but lower recall compared to duration treatment

The categorical engineering approach shows different error patterns compared to duration treatment.

### Kaggle Competition Results
Competition submission results:

<a href="../images/03_categorical_treatment_kaggle_submit.png" target="_blank">
  <img src="../images/03_categorical_treatment_kaggle_submit.png" alt="Kaggle Submission - Categorical Treatment" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

- **Competition Score**: 0.96907
- **Leaderboard Position**: 1159

!!! warning "Spoiler Warning"
    The following section contains additional details about the model and its evaluation. Click to expand to see more.

??? info "Additional Model Information"
    Based on the strong performance results, model V19 was identified as the best performing model and promoted to production with the champion alias in the MLflow Model Registry.

    <a href="../images/Promoted_model_production.png" target="_blank">
      <img src="../images/Promoted_model_production.png" alt="Model V19 Promoted to Production" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
    </a>

    - **Model Version**: V19
    - **Status**: Champion (Production)
    - **Deployment**: Ready for inference and business use
    - **Registry**: Versioned and tracked in MLflow Model Registry