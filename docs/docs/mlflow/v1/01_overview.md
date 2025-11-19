# V1.0 - Initial Implementation Overview

This documentation covers the **first iteration (V1.0)** of the bank client subscription classifier project, representing the foundational implementation focused on establishing core architecture and achieving baseline model performance.

## System Scope (V1.0)

The V1.0 implementation provides:

1. **Architecture Foundation**: Core class structure and MLflow integration patterns supporting future iterations
2. **Model Pipeline**: Complete training-to-prediction pipeline with proper experiment tracking
3. **Model Comparison**: Training and evaluation of multiple algorithms (LightGBM, XGBoost, CatBoost) to identify the best performer
4. **Competition Readiness**: Generation and submission of predictions to validate the end-to-end workflow

## Key Features

### Architecture Implementation
The V1.0 implementation prioritizes a functional system with baseline performance. **Hardcoded hyperparameters** are utilized based on common defaults for each algorithm, establishing the workflow structure prior to performance fine-tuning.

### Evaluation Metrics
While the system tracks different metrics (accuracy, precision, recall, F1-score), **AUC (Area Under Curve)** serves as the primary optimization target since it aligns with the competition's evaluation criteria and handles class imbalance effectively.

!!! info "AUC Methodology"
    AUC quantifies the model's discriminative capacity by evaluating the trade-off between true positive rate and false positive rate across all classification thresholds. It represents the probability that the model assigns a higher score to a randomly selected positive instance than to a randomly selected negative instance, with values ranging from 0.5 (random performance) to 1.0 (perfect discrimination).

### MLflow Integration
Every model training run is tracked in MLflow with:

- Model parameters and hyperparameters
- Performance metrics across validation sets
- Model artifacts and metadata
- Model registry for production flagging

### Production Readiness
The best-performing model (highest AUC) will be flagged for production use in the MLflow model registry and used to generate competition predictions, providing immediate feedback on the system's effectiveness.
