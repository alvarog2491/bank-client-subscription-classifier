# MLflow V3 Overview

## K-fold Cross-Validation Implementation

V3 implements K-fold cross-validation on the best model found by Optuna hyperparameter optimization from V2.

## What is K-fold Cross-Validation?

K-fold cross-validation splits the dataset into k equal parts (folds). The model is trained k times, each time using k-1 folds for training and 1 fold for validation. This provides more reliable performance estimates by testing the model on multiple data splits rather than a single validation set.

V3 uses stratified 5-fold cross-validation (defined in config file) which maintains the same class distribution across all folds, important for imbalanced datasets.

## Implementation

After Optuna finds the best hyperparameters, K-fold cross-validation is applied to evaluate the optimized model's performance across different data splits, this evaluation is performed only on the best model due to computational resource limitations. Only cross-validation metrics are logged to MLflow. 