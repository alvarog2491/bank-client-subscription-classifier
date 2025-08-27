# Model Training with Optuna Integration

## Training Commands

Traditional mode with predefined hyperparameters:
```bash
python -m src.models.train_model --model-type lightgbm --optimize False
```

Optimization mode with Optuna trials:
```bash
python -m src.models.train_model --model-type lightgbm --optimize True --n-trials 20
```

## Implementation: Hyperparameter Optimization Function

**File**: `src/models/train_model.py`

```python
def optimize_hyperparameters(
    model_type: str, X_train, y_train, X_val, y_val, config, n_trials: int
):
    """Run hyperparameter optimization using Optuna with persistent storage."""
    
    best_model = None
    best_metrics = None

    def objective(trial):
        nonlocal best_model, best_metrics
        
        # Create model with Optuna trial suggestions
        model_instance = ModelFactory.create_model(
            model_type, config, trial=trial
        )

        # Train and evaluate model
        model_instance.train(X_train, y_train, X_val, y_val)
        metrics = model_instance.evaluate(X_val, y_val)

        # Track best performing model
        if best_model is None or metrics["auc"] > best_metrics["auc"]:
            best_model = model_instance
            best_metrics = metrics

        return metrics["auc"]

    # Create persistent study with SQLite storage
    study_name = f"{model_type}_optimization"
    storage_url = "sqlite:///optuna.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, best_model, best_metrics
```

## Enhanced Model Factory Integration

The ModelFactory now supports Optuna trial objects alongside traditional hyperparameter dictionaries:

**File**: `src/models/core/model_factory.py`

```python
# Enhanced factory method signature
def create_model(
    cls,
    model_type: str,
    config: Dict[str, Any],
    hyperparams: Union[Dict[str, Any], optuna.Trial] = None,
    trial: optuna.Trial = None,
) -> BaseModel:
    """Create model instance supporting both hyperparams and Optuna trials."""
    
    # Handle trial parameter flexibility
    effective_trial = trial if trial is not None else hyperparams if isinstance(hyperparams, optuna.Trial) else None
    effective_hyperparams = hyperparams if not isinstance(hyperparams, optuna.Trial) else None

    model_class = cls._models[model_type]
    return model_class(config, effective_hyperparams, effective_trial)
```

## Model Implementation Enhancements

Each model implementation now includes conditional optimization logic. Key enhancement in LightGBM:

**File**: `src/models/implementations/lightgbm_model.py`

```python
def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
          X_val: pd.DataFrame, y_val: pd.Series) -> None:
    """Train model with mandatory validation data."""
    
    if self.trial:
        # NEW: Optuna hyperparameter suggestions
        params = {
            "n_estimators": self.trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": self.trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": self.trial.suggest_int("max_depth", 3, 10),
            "num_leaves": self.trial.suggest_int("num_leaves", 10, 300),
            "min_child_samples": self.trial.suggest_int("min_child_samples", 5, 100),
        }
    else:
        # EXISTING: Traditional hyperparameters from config
        params = {
            "n_estimators": self.hyperparams.get("n_estimators", 100),
            "learning_rate": self.hyperparams.get("learning_rate", 0.1),
            # ... other parameters
        }
```

## Enhanced Training Pipeline

The updated `train_model()` function seamlessly integrates optimization:

**File**: `src/models/train_model.py`

```python
def train_model(model_type: str, optimize: bool = False, n_trials: int = 10) -> str:
    """Enhanced training with optional hyperparameter optimization."""
    
    # Standard MLflow setup and data preparation...
    
    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("optimize", optimize)
        
        if optimize:
            mlflow.log_param("n_trials", n_trials)
            
            # Run optimization and get best model
            best_params, model_instance, metrics = optimize_hyperparameters(
                model_type, X_train, y_train, X_val, y_val, config, n_trials
            )
            
            # Log optimized parameters
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param}", value)
        else:
            # Traditional training approach
            hyperparams = config["hyperparameters"][model_type]
            model_instance = ModelFactory.create_model(model_type, config, hyperparams)
            
            for param, value in hyperparams.items():
                mlflow.log_param(param, value)
                
            model_instance.train(X_train, y_train, X_val, y_val)
            metrics = model_instance.evaluate(X_val, y_val)

        # Log final metrics and register model
        model_instance.log_metrics(metrics)
        model_name = f"{config['project']['name']}-{model_type}"
        model_instance.log_model(X_val, model_name=model_name)

        return mlflow.active_run().info.run_id
```

## Configuration Enhancements

### Extended config.yaml Structure

**File**: `config/config.yaml`

```yaml
# New optuna section added
optuna:
  lightgbm:
    n_estimators: [100, 1000]
    learning_rate: [0.01, 0.3]
    # ... parameter ranges
  xgboost:
    # ... parameter ranges  
  catboost:
    # ... parameter ranges
```

### Enhanced MLproject Entry Points

**File**: `MLproject`

```yaml
train:
  parameters:
    model_type: { type: string, default: "lightgbm" }
    optimize: { type: bool, default: false }      # NEW
    n_trials: { type: int, default: 10 }          # NEW
```