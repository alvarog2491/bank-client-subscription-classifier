# API Reference

This section provides detailed documentation for all modules, classes, and functions in the project.

## Core Modules

### Data Processing
::: src.data.load_data
::: src.data.preprocess

### Feature Engineering
::: src.features.build_features

### Model Training and Evaluation
::: src.models.train_model
::: src.models.predict_model
::: src.models.evaluate

### Utilities
::: src.utils.mlflow_utils

## Experiment Scripts

### Experiment Runner
::: experiments.experiment_runner

### Hyperparameter Tuning
::: experiments.hyperparameter_tuning

## Usage Examples

### Data Loading

```python
from src.data.load_data import load_competition_data

# Load training data
train_df = load_competition_data("data/raw/train.csv")

# Load test data
test_df = load_competition_data("data/raw/test.csv")
```

### Model Training

```python
from src.models.train_model import train_lightgbm_model
from src.utils.mlflow_utils import mlflow_run

@mlflow_run(experiment_name="lightgbm-baseline")
def train_baseline():
    model = train_lightgbm_model(
        train_data=train_df,
        target_column="y",
        model_params={
            "n_estimators": 100,
            "learning_rate": 0.1
        }
    )
    return model

model = train_baseline()
```

### Feature Engineering

```python
from src.features.build_features import create_features

# Create new features
enhanced_df = create_features(train_df)
```

### Model Evaluation

```python
from src.models.evaluate import evaluate_model

# Evaluate model performance
metrics = evaluate_model(
    model=model,
    X_test=X_test,
    y_test=y_test
)
```

## Configuration

The project uses a centralized configuration system. See `config/config.yaml` for all available parameters.

```python
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Access configuration
model_params = config["hyperparameters"]["lightgbm"]
```

## MLFlow Integration

All model training and experiments are integrated with MLFlow:

```python
import mlflow

# Set experiment
mlflow.set_experiment("bank-subscription-prediction")

# Start run and log everything
with mlflow.start_run():
    # Training code here
    mlflow.log_param("model_type", "lightgbm")
    mlflow.log_metric("roc_auc", auc_score)
    mlflow.sklearn.log_model(model, "model")
```

## Custom Decorators and Utilities

The project includes several utility decorators and functions to streamline the ML workflow:

```python
from src.utils.mlflow_utils import mlflow_run, log_model_metrics

@mlflow_run(experiment_name="my-experiment")
@log_model_metrics
def train_and_evaluate(X_train, y_train, X_val, y_val):
    # Training code
    return model, predictions
```

*This API documentation is automatically generated from docstrings. Keep docstrings up to date for accurate documentation.*