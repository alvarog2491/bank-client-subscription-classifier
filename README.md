# Bank Client Subscription Classifier

This project's **primary goal is to demonstrate MLflow workflows** for experiment tracking, model registry, reproducible ML pipelines, and hands-on familiarity with the MLflow UI and processes, while participating in Kaggle's Playground Series S5E8 competition.

## Documentation

[Full project analysis and documentation](https://bank-client-subscription-classifier.alvaro-ai-ml-ds-lab.com/)

### Navigation Guide

The documentation follows the complete MLFlow journey through different versions (V1, V2, V3, V4) showing the evolution of the process. **Navigate sequentially** through the left navigation panel from top to bottom to follow the development progression and understand how each iteration builds upon the previous version.


**Competition Details:**

- Competition overview: [Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8/overview)
- Dataset: [S5E8 synthetic banking data](https://www.kaggle.com/competitions/playground-series-s5e8/data)
- Task: Binary classification predicting term deposit subscriptions  
- Evaluation: ROC AUC score
- Timeline: August 2025

## Quick Setup

```bash
# Install dependencies
pip install -e .

# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
# MLflow UI will be available at http://localhost:5000

# Run MKDocs
mkdocs serve
```

## Project Structure

```
├── .github/                  # GitHub configuration
├── catboost_info/            # CatBoost training artifacts
├── config/                   # Configuration files and loaders
├── data/                     # Data files (raw, processed, predictions)
│   ├── predictions/          # Model predictions output
│   ├── processed/            # Processed datasets
│   └── raw/                  # Raw competition data
├── docs/                     # MkDocs documentation
├── images/                   # Project images and assets
├── mlartifacts/              # MLflow artifacts storage
├── mlruns/                   # MLflow experiment tracking data
├── notebooks/                # Jupyter notebooks for EDA and experiments
├── src/                      # Source code
│   ├── data/                 # Data loading and preprocessing modules
│   └── models/               # Model training, evaluation, and prediction
│       ├── core/             # Core model abstractions
│       └── implementations/  # Model implementations (LightGBM, XGBoost, CatBoost)
├── tests/                    # Unit tests
├── main.py                   # Main entry point
├── MLproject                 # MLflow project definition
├── mkdocs.yml                # MkDocs configuration
├── pyproject.toml            # Python project configuration
└── python_env.yaml           # Conda environment specification (Needed for MLFlow)
```

## Development Status

The core project structure is complete with full MLflow implementation across multiple iterations (V1-V4). The Kaggle competition has concluded, but the project remains open for further enhancements such as SHAP values for feature importance analysis, additional model interpretability tools, and extended MLflow workflow demonstrations.
