# Bank Client Subscription Classifier

MLFlow project for Kaggle's Playground Series S5E8 competition focused on predicting bank client subscription to term deposits.

## Project Overview

This project demonstrates MLFlow workflows for experiment tracking and model management using a realistic banking dataset from the Kaggle competition. The dataset contains client demographic information, previous campaign interactions, and economic indicators to predict subscription outcomes.

**Competition Details:**

- Competition overview: [Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8/overview)
- Dataset: [S5E8 synthetic banking data](https://www.kaggle.com/competitions/playground-series-s5e8/data)
- Task: Binary classification predicting term deposit subscriptions  
- Evaluation: ROC AUC score
- Timeline: August 2025

## Quick Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start MLFlow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
# MLFlow UI will be available at http://localhost:5000

# Run MKDocs
mkdocs serve
```

## Project Structure

```
├── config/                 # Configuration files
├── data/                  # Data files (raw, processed)
├── experiments/           # Experiment scripts
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── data/            # Data processing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training
│   └── utils/           # Utilities
├── tests/               # Unit tests
└── MLproject           # MLFlow project definition
```

## Development Status

Initial project structure established with data exploration completed. Ready for feature engineering and model development phases.