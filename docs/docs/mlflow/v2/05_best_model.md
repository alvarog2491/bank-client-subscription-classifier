# Best Model Selection with Optuna

## Results

<a href="../images/Optuna_results.png" target="_blank">
  <img src="../images/Optuna_results.png" alt="Optuna Results Comparison" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

**XGBoost** wins again with AUC **0.968** (improved from V1's 0.966)

Other results:
- **CatBoost**: AUC **0.967**
- **LightGBM**: AUC **0.967**

## Winning XGBoost Parameters

<a href="../images/Xgboost_optuna_metrics_params.png" target="_blank">
  <img src="../images/Xgboost_optuna_metrics_params.png" alt="XGBoost Optuna Parameters" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

!!! warning "Class Imbalance Impact"
    The winning model shows a **high number of false negatives**, likely due to the 7.3:1 class imbalance in the dataset. While achieving strong AUC performance, the model tends to under-predict positive cases (subscription = 1). This issue will be addressed in V3 through class imbalance handling techniques.

