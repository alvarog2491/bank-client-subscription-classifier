# Conclusions

## V1 Summary

The V1.0 system establishes a **MLflow-based infrastructure** for experiment tracking, model registry, and reproducible ML pipelines, prioritizing architectural stability over immediate maximization of the competition score.

## Key Achievements

The champion XGBoost model achieved a **0.96715 AUC score** on the Kaggle competition, placing only **1% behind the leader**. This demonstrates that the model selection process and MLflow pipeline are working effectively. More importantly, a complete and successful end-to-end workflow has been built, with experiment tracking and model registry using algorithm-specific flavors.

The project effectively demonstrates MLflow's capabilities through consistent model comparison across LightGBM, XGBoost, and CatBoost, with all experiments being fully reproducible. The prediction system generates Kaggle-compatible output while maintaining native `predict_proba` support for optimal performance.

## Current Limitations

The models currently use **hardcoded hyperparameters** rather than optimized values. This was intentional for v1 since the primary goal was establishing the MLflow structure and tracking capabilities, but it clearly limits the performance potential.

## Future Improvements

The subsequent iteration is planned to integrate **Optuna** for systematic hyperparameter optimization, replacing the current hardcoded parameters to improve model performance.
