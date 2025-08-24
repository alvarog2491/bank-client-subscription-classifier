# MLFlow Tracking

## Experiment Configuration

### Tracking Setup
The project uses SQLite backend for MLFlow tracking:

```python
# Configuration in config/config.yaml
mlflow:
  experiment_name: "bank-subscription-prediction"
  tracking_uri: "sqlite:///mlflow.db"
  artifact_location: "./mlartifacts"
```

### Experiment Initialization  
```python
import mlflow

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])
```

## Tracking Implementation

### Run Context
All training happens within MLFlow run context:
```python
with mlflow.start_run(nested=True):
    # Training and logging code
    pass
```

### Parameter Logging
Automatic logging of all model hyperparameters:
```python
mlflow.log_param("model_type", model_type)
mlflow.log_param("random_state", random_state)
for param, value in hyperparams.items():
    mlflow.log_param(param, value)
```

### Metric Logging  
Standard evaluation metrics logged:
```python
metrics = model_instance.evaluate(X_val, y_val)
for metric_name, metric_value in metrics.items():
    mlflow.log_metric(metric_name, metric_value)
```

### Model Artifacts
Models automatically logged with signatures and examples:
```python
# LightGBM example
mlflow.lightgbm.log_model(
    lgb_model=self.model,
    name="lightgbm_model",
    registered_model_name=model_name,
    signature=signature,
    input_example=input_example
)
```

## Tracked Information

### Parameters
- Model type (lightgbm, xgboost, catboost)
- All hyperparameters from config.yaml
- Random seed for reproducibility
- Data split configuration

### Metrics  
- Validation accuracy
- Validation ROC AUC score
- Model-specific training metrics

### Artifacts
- Trained model objects
- Model signatures  
- Input examples for inference
- Preprocessing artifacts (label encoders)

## Model Registry Integration

### Automatic Registration
Models are automatically registered during training:
```python
model_name = f"{config['project']['name']}-{model_type}"
model_instance.log_model(X_val, model_name=model_name)
```

### Registry Naming Convention
- `bank-client-subscription-classifier-lightgbm`
- `bank-client-subscription-classifier-xgboost`
- `bank-client-subscription-classifier-catboost`

## MLFlow UI Access

Start the MLFlow server:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Access the UI at: http://localhost:5000

### UI Features Available
- Experiment comparison across runs
- Parameter and metric visualization  
- Model artifact download
- Model registry management
- Run reproduction via MLFlow projects