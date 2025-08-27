# V2 Conclusions

The **Optuna integration was successfully implemented** into the MLflow pipeline. The system now supports both traditional and optimized training modes with persistent studies that build optimization knowledge over time.

Results showed improvement from leaderboard position 1,427 to 1,193 with a competition score of 0.96878. Local validation AUC improved from 0.966 to 0.968. While the gains are modest, they're consistent across validation and competition metrics.

The current architecture is capable of extracting the full potential from the models through automated hyperparameter optimization. With this foundation in place, the **focus now shifts to data and feature engineering**. V3 will concentrate on improving the input data quality and creating more informative features based on the EDA analysis.