# Generating Test Predictions with Optimized Model

## Model Inference

Using the champion XGBoost model (version 10) to generate competition predictions:

**File**: `src/models/predict_model.py`

```bash
python -m src.models.predict_model \
     --model-uri models:/bank-client-subscription-classifier-xgboost/10 \
     --model-type xgboost \
     --input-path data/processed/test_processed.csv \
     --output-path data/predictions/xgboost_predictions.csv
```


## Competition Results

<a href="../images/optuna_result_competition_v2.png" target="_blank">
  <img src="../images/optuna_result_competition_v2.png" alt="Optuna Competition Results V2" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

## Performance Analysis

The Optuna-optimized XGBoost model showed improvements in the competition leaderboard:

- **Leaderboard Position**: Improved from rank 1,427 to 1,193 (234 position improvement)
- **Competition Score**: Enhanced from V1's baseline to 0.96878
- **Local Validation**: AUC improved from 0.966 to 0.968

While this isn't a dramatic gain, it's still a solid improvement that validates the hyperparameter optimization approach.