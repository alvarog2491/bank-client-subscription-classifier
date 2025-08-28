# MLflow V3 Overview

## Foundation Complete

With V2, the MLflow infrastructure is now stable and reliable. Time to focus on feature engineering and data exploration.

!!! info "XGboost as base model"
    XGBoost was identified as the most appropriate model for this dataset through previous experiments. All trials in this section will be conducted with XGBoost only, as testing all models would be time-intensive. Focus is on optimizing XGBoost performance with 20 trials per run to achieve the best results efficiently.


## Approach

V3 implements systematic feature engineering based on EDA insights. Each improvement is tracked separately to measure individual impact on model performance.

Pipeline structure:
```
Raw Data -> Duration Treatment -> Categorical Engineering -> Numerical Enhancement -> Feature Interactions -> Model Training
```

## Class Imbalance Handling

The first priority is addressing the high number of false negatives identified in [V2 best model results](../v2/05_best_model.md). 

Added `scale_pos_weight` parameter to XGBoost hyperparameter optimization (range 1.0-15.0) to handle the 7.3:1 class imbalance by giving more weight to positive subscription cases.
