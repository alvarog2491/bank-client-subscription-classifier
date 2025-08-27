# MLFlow Guidelines

Essential MLFlow commands and patterns for the project.

## Server Management

[Official Documentation: MLflow Tracking Server](https://mlflow.org/docs/latest/tracking/server.html)

```bash
# Start MLFlow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 5001

# Access UI
http://localhost:5001
```

## Run MLFlow project entry points

[Official Documentation: MLflow Projects](https://mlflow.org/docs/latest/projects.html)

```bash
# Run MLFLow CLI entry points
mlflow run . -e data_preprocessing
mlflow run . -e train -P model_type=lightgbm # before Optuna
mlflow run . -e train -P model-type=lightgbm -P optimize=True -P n-trials=3 # After Optuna
mlflow run . -e predict -P model_uri="models:/ModelName/Production"
mlflow run . -e main

# Python 
python -m src.models.train_model --model-type lightgbm # before Optuna
python -m src.models.train_model --model-type lightgbm --optimize True --n-trials 20 # After Optuna
```

## Basic Experiment Tracking

[Official Documentation: MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

```python
import mlflow
import mlflow.sklearn

# Setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("experiment-name")

# Track run
with mlflow.start_run(run_name="run-name"):
    # Log parameters
    mlflow.log_param("model_type", "lightgbm")
    mlflow.log_params({"n_estimators": 100, "learning_rate": 0.1})
    
    # Log metrics
    mlflow.log_metric("roc_auc", 0.85)
    mlflow.log_metrics({"accuracy": 0.78, "precision": 0.82})
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
```

## Model Registry

[Official Documentation: MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()

# Register model
model_uri = f"runs/{run_id}/model"
mlflow.register_model(model_uri, "BankSubscriptionClassifier")

# Transition stages
client.transition_model_version_stage(
    name="BankSubscriptionClassifier",
    version=1,
    stage="Production"
)

# Load models
model = mlflow.sklearn.load_model("models:/BankSubscriptionClassifier/Production")
model = mlflow.sklearn.load_model(f"runs/{run_id}/model")
```

## Hyperparameter Tuning with Optuna

[Official Documentation: MLflow with Hyperopt](https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/index.html)

```python
import optuna

def objective(trial):
    with mlflow.start_run(nested=True):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
        }
        
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        
        mlflow.log_params(params)
        mlflow.log_metric("val_auc", score)
        
        return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

## Configuration Pattern

```yaml
# config/config.yaml
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "bank-subscription-prediction"
  artifact_location: "./mlartifacts"
```

```python
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
```

## Common Queries

[Official Documentation: MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)

```python
# Find best run
experiment = client.get_experiment_by_name("experiment-name")
runs = client.search_runs(experiment.experiment_id)
best_run = max(runs, key=lambda r: r.data.metrics.get('val_auc', 0))

# List experiments
experiments = client.search_experiments()

# Get run details
run = client.get_run(run_id)
print(run.data.params)
print(run.data.metrics)
```

## Quick Experiment Function

```python
def quick_experiment(model, params, name):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        mlflow.log_metric("val_auc", score)
        mlflow.sklearn.log_model(model, "model")
        return score
```

## Troubleshooting

```bash
# Check port usage
lsof -i :5000
# for macOS
lsof -i tcp:5000
kill -9 $(lsof -ti tcp:5000)

# Use different port
mlflow server --port 5001 --backend-store-uri sqlite:///mlflow.db

# Remove all MLflow cache and data:
rm -rf ~/.mlflow
rm -rf mlflow.db
rm -rf mlartifacts/
rm -rf .mlflow

# End active run
python -c "import mlflow; mlflow.end_run() if mlflow.active_run() else print('No active run')"
# Check artifact permissions
ls -la mlartifacts/

# Create artifact directory
mkdir -p mlartifacts
```

## MLProject Entry Points

[Official Documentation: MLproject File Format](https://mlflow.org/docs/latest/projects.html#mlproject-file)

```yaml
# MLproject
name: bank-client-subscription-classifier
python_env: python_env.yaml

entry_points:
  train:
    parameters:
      model_type: {type: string, default: "lightgbm"}
      n_estimators: {type: int, default: 100}
    command: "python src/models/train_model.py --model-type {model_type}"
```

These patterns cover the most common MLFlow operations needed for machine learning experimentation and model management.