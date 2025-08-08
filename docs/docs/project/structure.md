# Project Structure

This document explains the organization and purpose of each directory and file in the project.

## Directory Structure

```
bank-client-subscription-classifier/
├── config/                     # Configuration files
│   └── config.yaml            # Main configuration
├── data/                      # Data storage
│   ├── raw/                  # Raw competition data
│   ├── processed/            # Processed datasets
│   └── external/             # External data sources
├── docs/                     # Documentation
│   ├── docs/                 # MKDocs source files
│   └── overrides/            # Theme customizations
├── experiments/              # Experiment scripts
│   ├── experiment_runner.py  # Main experiment runner
│   └── hyperparameter_tuning.py # HP optimization
├── notebooks/               # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── src/                     # Source code
│   ├── data/               # Data processing
│   │   ├── __init__.py
│   │   ├── load_data.py    # Data loading utilities
│   │   └── preprocess.py   # Data preprocessing
│   ├── features/           # Feature engineering
│   │   ├── __init__.py
│   │   └── build_features.py # Feature creation
│   ├── models/             # Model training/evaluation
│   │   ├── __init__.py
│   │   ├── train_model.py  # Model training
│   │   ├── predict_model.py # Predictions
│   │   └── evaluate.py     # Model evaluation
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── mlflow_utils.py # MLFlow utilities
├── tests/                  # Unit tests
├── .gitignore             # Git ignore rules
├── main.py                # Main execution script
├── MLproject              # MLFlow project definition
├── mkdocs.yml             # Documentation configuration
├── python_env.yaml        # Python environment for MLFlow
└── requirements.txt       # Project dependencies
```

## Key Files Explained

### Configuration Files

- **`config/config.yaml`**: Central configuration for all project parameters
- **`MLproject`**: MLFlow project definition with entry points
- **`python_env.yaml`**: Python environment specification for MLFlow
- **`requirements.txt`**: All project dependencies

### Source Code Organization

#### `src/data/`
- **`load_data.py`**: Functions to load raw data from various sources
- **`preprocess.py`**: Data cleaning and preprocessing pipelines

#### `src/features/`
- **`build_features.py`**: Feature engineering and transformation functions

#### `src/models/`
- **`train_model.py`**: Model training with MLFlow integration
- **`predict_model.py`**: Generate predictions on new data
- **`evaluate.py`**: Model evaluation and metrics calculation

#### `src/utils/`
- **`mlflow_utils.py`**: MLFlow helper functions and decorators

### Experiments and Analysis

- **`experiments/`**: Production-ready experiment scripts
- **`notebooks/`**: Interactive analysis and prototyping

### Documentation

- **`docs/docs/`**: MKDocs documentation source
- **`mkdocs.yml`**: Documentation site configuration

## Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Reproducibility**: All experiments tracked with MLFlow
3. **Scalability**: Easy to add new models or features
4. **Documentation**: Comprehensive docs for all components
5. **Testing**: Unit tests for critical functions

## Workflow Integration

The structure supports the complete ML workflow:

1. **Data Pipeline**: `src/data/` → `data/processed/`
2. **Feature Pipeline**: `src/features/` → enhanced datasets
3. **Model Pipeline**: `src/models/` → MLFlow experiments
4. **Evaluation Pipeline**: `src/models/evaluate.py` → results
5. **Documentation**: All steps documented in MKDocs

This organization ensures clear separation of concerns while maintaining easy navigation and understanding of the project flow.