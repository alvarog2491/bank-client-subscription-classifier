# Upload Predictions

## Initial Submission Results

After generating predictions using the champion XGBoost model, the submission was uploaded to Kaggle for evaluation.

### Submission Performance

<a href="../images/prediction_submited_v1.png" target="_blank">
  <img src="../images/prediction_submited_v1.png" alt="First Prediction Submission Results" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

**Initial Score:** 0.96715  
**Performance:** Strong baseline performance with AUC score very close to 1.0

### Competition Standing

<a href="../images/leaderbord_comparation_v1.png" target="_blank">
  <img src="../images/leaderbord_comparation_v1.png" alt="Leaderboard Position Comparison" width="800" style="cursor: pointer; border: 1px solid #ddd; border-radius: 4px; transition: 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
</a>

**Current Position:** 1,427th place  
**Best Score:** 0.97829  
**Gap to Leader:** 0.01114 (approximately 1.1%)

## Performance Analysis

The initial submission achieved a score of **0.96715**, indicating a high level of predictive accuracy with an AUC near 1.0. The **1% gap to the leader** validates the effectiveness of the selected model and MLflow pipeline.

While the rank of 1,427th reflects the high participation level of the competition, the **small performance differential** suggests the current approach provides a solid foundation for further optimization.