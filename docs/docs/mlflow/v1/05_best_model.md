# Best Model Selection

## Performance Analysis

After training three gradient boosting models with hardcoded hyperparameters, comparative performance analysis reveals consistent results across all implementations. Each model achieved AUC scores ranging from 0.96 to 0.966, indicating strong predictive capability.

<a href="../images/model_comparison_graph.png" target="_blank">
  <img src="../images/model_comparison_graph.png" alt="Model Comparison Graph" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

## Model Comparison Results

The evaluation demonstrates similar performance metrics across all three model types, with marginal differences in AUC scores:

<a href="../images/model_comparison_list.png" target="_blank">
  <img src="../images/model_comparison_list.png" alt="Model Comparison List" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

## Champion Model Selection

Based on the initial evaluation results, the best performing model was identified as:

**Run ID:** f2c64e293c9246fa904dd6f66bce8c9f `(treasured-mole-567)`  
**Model Type:** XGBoost  
**AUC Score:** 0.966

This model represents the current champion configuration for the bank client subscription prediction task.

## Best Model Performance

Detailed metrics and parameters for the selected champion model:

<a href="../images/model_metrics_v1.png" target="_blank">
  <img src="../images/model_metrics_v1.png" alt="Best Model Performance Metrics" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

While all models demonstrate strong performance within a narrow AUC range (0.96-0.966), the selected XGBoost model achieved the highest validation AUC score.