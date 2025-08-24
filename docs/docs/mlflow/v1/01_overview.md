# V1.0 - Initial Implementation Overview

This documentation covers the **first iteration (V1.0)** of the bank client subscription classifier project, representing the foundational implementation focused on establishing core architecture and achieving baseline model performance.

## Project Goals for V1.0

The primary objectives for this initial version are:

1. **Architecture Foundation**: Establish the core class structure and MLflow integration patterns that will support future iterations
2. **Model Pipeline**: Create a complete training-to-prediction pipeline with proper experiment tracking
3. **Model Comparison**: Train and evaluate multiple algorithms (LightGBM, XGBoost, CatBoost) to identify the best performer
4. **Competition Readiness**: Generate and submit predictions to validate the end-to-end workflow

## Focus Areas

### Achitecture development
The V1.0 implementation emphasizes getting a working system with baseline performance rather than optimization. **Hardcoded hyperparameters** are used based on common defaults for each algorithm, focusing on establishing the workflow rather than fine-tuning performance.

### Evaluation Metrics
While the system tracks different metrics (accuracy, precision, recall, F1-score), **AUC (Area Under Curve)** serves as the primary optimization target since it aligns with the competition's evaluation criteria and handles class imbalance effectively.

### MLflow Integration
Every model training run is tracked in MLflow with:

- Model parameters and hyperparameters
- Performance metrics across validation sets
- Model artifacts and metadata
- Model registry for production flagging

### Production Readiness
The best-performing model (highest AUC) will be flagged for production use in the MLflow model registry and used to generate competition predictions, providing immediate feedback on the system's effectiveness.
