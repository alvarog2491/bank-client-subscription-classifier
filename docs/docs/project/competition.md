# Competition Details

This is for Kaggle's [Playground Series S5E8](https://kaggle.com/competitions/playground-series-s5e8) - a binary classification challenge with banking data.

## The task

Predict whether bank clients will subscribe to a term deposit after a direct marketing campaign. Pretty straightforward business problem - banks want to know who to target with their marketing efforts.

**Timeline**: August 1-31, 2025 (plenty of time to experiment)

## Dataset Advantages

- Banking domain provides rich opportunities for feature engineering
- Playground Series datasets are clean and well-structured
- ROC AUC scoring is appropriate for business classification problems
- Synthetic data ensures fair competition without privacy concerns

## The dataset

Synthetic data based on real banking patterns - which is nice because it means no privacy concerns and no weird data leakage issues to worry about.

**Files you get**:
- `train.csv` - training data with features and target
- `test.csv` - test features (for final predictions)  
- `sample_submission.csv` - shows the submission format

**Target variable**: `y` where 1 = client subscribes, 0 = doesn't

The features are typical banking/demographic stuff - age, job, education, financial status, previous campaign interactions, etc.

## Scoring

Competition is scored on **ROC AUC**, which makes sense for this type of problem. It handles class imbalance well and focuses on ranking/probability calibration rather than just hard classifications.

You submit probabilities (0.0 to 1.0) for each test case, not binary predictions.

AUC ranges:
- 0.90+ = excellent (probably won't happen but would be nice)
- 0.80-0.90 = good, competitive score
- 0.70-0.80 = decent starting point
- Below 0.70 = need to rethink approach

## Submission format

Standard Kaggle format - CSV with id and probability:

```csv
id,y
750000,0.5
750001,0.8
750002,0.2
```

Make sure to include all test IDs and submit probabilities, not classifications.

## Technical Approach

This project implements a comprehensive MLFlow-based machine learning pipeline with the following objectives:

1. **Complete MLFlow Integration** - systematic experiment tracking, model registry, and reproducible pipelines
2. **Systematic Model Comparison** - evaluate multiple algorithms with consistent methodology
3. **Model Interpretability** - comprehensive analysis using SHAP and feature importance
4. **Production-Ready Architecture** - maintainable, documented, and scalable codebase

The project serves as a reference implementation for MLFlow best practices in competitive machine learning.

## Implementation Timeline

**Phase 1**: Data exploration and MLFlow infrastructure setup  
**Phase 2**: Feature engineering and baseline model development  
**Phase 3**: Advanced modeling and hyperparameter optimization  
**Phase 4**: Model interpretation, ensemble methods, and final submission

### Detailed Implementation Steps

1. **Data Analysis**: Exploratory data analysis and quality assessment
2. **MLFlow Setup**: Configure experiment tracking, model registry, and project structure
3. **Baseline Models**: Establish performance benchmarks with simple algorithms
4. **Feature Engineering**: Domain-specific feature creation and selection
5. **Algorithm Comparison**: Systematic evaluation of multiple ML algorithms
6. **Hyperparameter Optimization**: Automated tuning using Optuna
7. **Model Interpretation**: SHAP analysis and feature importance evaluation
8. **Production Pipeline**: Finalize reproducible training and prediction workflows

## Useful links

- [Competition page](https://kaggle.com/competitions/playground-series-s5e8)
- [MLFlow docs](https://mlflow.org/docs/latest/index.html)
- [Material for MKDocs](https://squidfunk.github.io/mkdocs-material/getting-started)