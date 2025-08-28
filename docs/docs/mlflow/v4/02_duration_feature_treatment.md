# Duration Feature Treatment (Step 2)

## Overview

Duration emerges as the strongest predictor from the EDA analysis, showing the highest correlation (r=0.519) with the target variable. This step focuses on extracting maximum value from this critical feature while addressing potential data leakage concerns.

## Key EDA Findings

- **Strongest correlation**: Duration shows r=0.519 with target (highest among all features)
- **Clear engagement pattern**: Subscribers average 525 seconds vs 212 seconds for non-subscribers
- **Business insight**: Call duration serves as an early success indicator during campaigns
- **Data quality**: Duration has right-skewed distribution requiring transformation

## Implementation Strategy

### 1. Duration Binning
Created categorical bins based on engagement patterns:

- `very_short` (0-120s): Quick rejections
- `short` (120-300s): Standard interactions  
- `medium` (300-600s): Engaged prospects
- `long` (>600s): High engagement calls

### 2. Log Transformation
Applied `np.log1p()` transformation to handle right-skewed distribution and reduce impact of extreme outliers.

### 3. High Engagement Flag
Binary feature `duration_high_engagement` flags calls exceeding 300 seconds, capturing the threshold where conversion likelihood increases significantly.

## Code Implementation

```python
def apply_duration_feature_treatment(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply duration feature treatment based on EDA insights."""
    
    # Duration bins based on engagement patterns
    duration_bins = [0, 120, 300, 600, float('inf')]
    duration_labels = ['very_short', 'short', 'medium', 'long']
    
    train_df['duration_bin'] = pd.cut(train_df['duration'], bins=duration_bins, labels=duration_labels, include_lowest=True)
    test_df['duration_bin'] = pd.cut(test_df['duration'], bins=duration_bins, labels=duration_labels, include_lowest=True)
    
    # Log transformation for skewed distribution
    train_df['duration_log'] = np.log1p(train_df['duration'])
    test_df['duration_log'] = np.log1p(test_df['duration'])
    
    # High engagement binary flag
    train_df['duration_high_engagement'] = (train_df['duration'] > 300).astype(int)
    test_df['duration_high_engagement'] = (test_df['duration'] > 300).astype(int)
    
    return train_df, test_df
```

## Features Created

| Feature | Type | Description |
|---------|------|-------------|
| `duration_bin` | Categorical | 4 engagement levels (very_short, short, medium, long) |
| `duration_log` | Numerical | Log-transformed duration to handle skewness |
| `duration_high_engagement` | Binary | Flag for calls > 300 seconds |

## Data Transformation Example

### Before Processing
| duration | y | Description |
|----------|---|-------------|
| 117      | 0 | 117 seconds, no subscription |
| 185      | 0 | 185 seconds, no subscription |
| 111      | 0 | 111 seconds, no subscription |
| 10       | 0 | 10 seconds, no subscription |

### After Processing
| duration | duration_bin | duration_log | duration_high_engagement | y | Analysis |
|----------|--------------|--------------|-------------------------|---|----------|
| 117      | very_short   | 4.77         | 0                       | 0 | Short call, low engagement |
| 185      | short        | 5.23         | 0                       | 0 | Standard call, low engagement |
| 111      | very_short   | 4.72         | 0                       | 0 | Quick rejection, very low engagement |
| 10       | very_short   | 2.40         | 0                       | 0 | Immediate hang-up, no engagement |


## Expected Impact

- **Capture non-linear patterns**: Binning reveals engagement thresholds
- **Handle distribution skewness**: Log transformation improves model performance
- **Create interpretable features**: Business-meaningful engagement levels
- **Preserve original signal**: Keep original duration alongside engineered features

## Results

### MLflow Performance
Duration feature treatment results:

<a href="../images/02_mlflow_duration.png" target="_blank">
  <img src="../images/02_mlflow_duration.png" alt="MLflow Results - Duration Treatment" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

**Single Model Performance (80/20 split):**
- **Test AUC: 0.96844**

**K-Fold Cross-Validation (5 folds):**
- **Average AUC: 0.9686**

Results show consistent performance with low variance across cross-validation folds. The improvement over baseline is marginal.

### Classification Metrics
Performance comparison with baseline results:

<a href="../images/02_mlflow_duration_metrics.png" target="_blank">
  <img src="../images/02_mlflow_duration_metrics.png" alt="MLflow Duration Treatment Metrics" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

- **False Positives**: Increased from 3,702 to 5,374 (+1,672)
- **False Negatives**: Reduced from 6,187 to 4,409 (-1,778)
- **Trade-off**: Higher recall but more false positives

### Kaggle Competition Results
Competition submission results:

<a href="../images/02_duration_treatment_kaggle_submit.png" target="_blank">
  <img src="../images/02_duration_treatment_kaggle_submit.png" alt="Kaggle Submission - Duration Treatment" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

- **Competition Score**: 0.969
- **Leaderboard Position**: 1170