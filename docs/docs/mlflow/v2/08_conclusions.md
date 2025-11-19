# V2 Conclusions

The **Optuna integration** has been successfully incorporated into the MLflow pipeline, enabling both traditional and optimized training modes with persistent studies that build optimization knowledge over time.

Results demonstrated an improvement in competition score to 0.96878 (from leaderboard position 1,427 to 1,193) and local validation AUC to 0.968. These consistent gains across metrics validate the effectiveness of the optimization strategy.

The current architecture supports full model potential extraction through automated hyperparameter optimization. Future iterations (V3 and V4) are planned to introduce K-Fold cross-validation for reliable metrics and focus on data quality and feature engineering improvements.