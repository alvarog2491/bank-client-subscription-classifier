# Results & Analysis

## V1.0 Performance Results

Results from initial model implementations with hardcoded hyperparameters.

## Model Comparison

### Current Baselines
All models trained on processed data with train/validation split (test_size=0.2, random_state=42).

**Evaluation Metrics:**
- Accuracy Score
- ROC AUC Score

### LightGBM Results
- **Hyperparameters**: n_estimators=200, learning_rate=0.1, max_depth=6, num_leaves=100
- **Performance**: Results available in MLFlow UI after training

### XGBoost Results  
- **Hyperparameters**: n_estimators=200, learning_rate=0.1, max_depth=6, min_child_weight=3
- **Performance**: Results available in MLFlow UI after training

### CatBoost Results
- **Hyperparameters**: iterations=200, learning_rate=0.1, depth=6, l2_leaf_reg=5  
- **Performance**: Results available in MLFlow UI after training

## MLFlow Experiment Tracking

View detailed results in the MLFlow UI:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
# Navigate to http://localhost:5000
```

## Model Registry

All trained models are automatically registered in MLFlow model registry with naming convention:
- `bank-client-subscription-classifier-lightgbm`
- `bank-client-subscription-classifier-xgboost` 
- `bank-client-subscription-classifier-catboost`

## Next Steps

V2.0 will focus on hyperparameter optimization to improve upon these baseline results.