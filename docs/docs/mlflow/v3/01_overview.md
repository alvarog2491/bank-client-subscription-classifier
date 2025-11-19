# System Scope (V3.0)

## K-fold Cross-Validation Implementation

The V3.0 implementation applies K-fold cross-validation to the best model identified by the Optuna hyperparameter optimization process from V2.

## Cross-Validation Strategy

K-fold cross-validation involves splitting the dataset into *k* equal parts (folds). The model is trained *k* times, using *k-1* folds for training and 1 fold for validation in each iteration. This approach yields more reliable performance estimates by evaluating the model on multiple data splits.

The system employs **stratified 5-fold cross-validation** (configurable), which preserves the class distribution across all folds—a critical factor for handling imbalanced datasets.

## Implementation

After Optuna finds the best hyperparameters, K-fold cross-validation is applied to evaluate the optimized model's performance across different data splits, this evaluation is performed only on the best model due to computational resource limitations. Only cross-validation metrics are logged to MLflow. 