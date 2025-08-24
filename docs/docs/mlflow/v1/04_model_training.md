# Model Training

## Current Implementation

The model training system implements three gradient boosting algorithms with hardcoded hyperparameters. All models follow a consistent interface through abstract base class inheritance.

## Architecture

### Base Model Class
Located in `src/models/core/base_model.py`

```python
class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any], hyperparams: Dict[str, Any] = None):
        self.config = config
        self.hyperparams = hyperparams or {}
        self.random_state = config.get("model", {}).get("random_state", 42)
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod  
    def predict_proba(self, X):
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, model_uri: str, config: dict):
        """Load a model from MLflow with proper predict_proba support."""
        pass
    
    def evaluate(self, X_test, y_test):
        """Complete model evaluation with multiple metrics"""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # Calculate complete metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix components
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision), 
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
```

### Model Factory
Located in `src/models/core/model_factory.py`

The ModelFactory provides both model creation and loading capabilities:

```python
class ModelFactory:
    _models = {
        "lightgbm": LightGBMModel,
        "xgboost": XGBoostModel, 
        "catboost": CatBoostModel,
    }
    
    @classmethod
    def create_model(cls, model_type, config, hyperparams=None):
        if model_type not in cls._models:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_class = cls._models[model_type]
        return model_class(config, hyperparams)
    
    @classmethod
    def load_model(cls, model_type: str, model_uri: str, config: Dict[str, Any]):
        """Load a model instance from MLflow with proper predict_proba support.
        
        Args:
            model_type: Type of model to load (e.g. 'lightgbm', 'xgboost', 'catboost')
            model_uri: MLflow model URI
            config: Configuration dictionary for the model
            
        Returns:
            Loaded model instance with native predict_proba support
        """
        if model_type not in cls._models:
            supported_models = list(cls._models.keys())
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported models: {supported_models}")
        
        model_class = cls._models[model_type]
        return model_class.load(model_uri, config)
```

## Model Implementations

### 1. LightGBM Model
Located in `src/models/implementations/lightgbm_model.py`

**Hyperparameters (from config.yaml):**
```yaml
lightgbm:
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 6
  num_leaves: 100
  min_child_samples: 50
```

**Implementation Details:**
```python
def train(self, X_train, y_train, X_val=None, y_val=None):
    self.model = lgb.LGBMClassifier(
        n_estimators=self.hyperparams.get("n_estimators", 100),
        learning_rate=self.hyperparams.get("learning_rate", 0.1),
        max_depth=self.hyperparams.get("max_depth", -1),
        num_leaves=self.hyperparams.get("num_leaves", 31),
        min_child_samples=self.hyperparams.get("min_child_samples", 20),
        random_state=self.random_state,
        verbose=-1
    )
    
    eval_set = [(X_val, y_val)] if X_val is not None else None
    
    self.model.fit(
        X_train, y_train,
        eval_set=eval_set,
        callbacks=[lgb.early_stopping(10)] if eval_set else None
    )

@classmethod
def load(cls, model_uri: str, config: dict):
    """Load a LightGBM model from MLflow with native predict_proba support."""
    try:
        # Load using LightGBM-specific flavor for better predict_proba support
        model = mlflow.lightgbm.load_model(model_uri)
        
        # Create a model instance
        instance = cls(config)
        instance.model = model
        instance.is_trained = True
        
        return instance
    except Exception as e:
        raise RuntimeError(f"Failed to load LightGBM model from {model_uri}: {e}")
```

### 2. XGBoost Model  
Located in `src/models/implementations/xgboost_model.py`

**Hyperparameters (from config.yaml):**
```yaml
xgboost:
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 6
  min_child_weight: 3
```

**Implementation Details:**
```python
def train(self, X_train, y_train, X_val=None, y_val=None):
    self.model = xgb.XGBClassifier(
        n_estimators=self.hyperparams.get("n_estimators", 100),
        learning_rate=self.hyperparams.get("learning_rate", 0.3),
        max_depth=self.hyperparams.get("max_depth", 6),
        min_child_weight=self.hyperparams.get("min_child_weight", 1),
        random_state=self.random_state,
        eval_metric="logloss"
    )
    
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))
    
    self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

@classmethod
def load(cls, model_uri: str, config: dict):
    """Load an XGBoost model from MLflow with native predict_proba support."""
    try:
        # Load using XGBoost-specific flavor for better predict_proba support
        model = mlflow.xgboost.load_model(model_uri)
        
        # Create a model instance
        instance = cls(config)
        instance.model = model
        instance.is_trained = True
        
        return instance
    except Exception as e:
        raise RuntimeError(f"Failed to load XGBoost model from {model_uri}: {e}")
```

### 3. CatBoost Model
Located in `src/models/implementations/catboost_model.py`

**Hyperparameters (from config.yaml):**
```yaml
catboost:
  iterations: 200
  learning_rate: 0.1
  depth: 6
  l2_leaf_reg: 5
```

**Implementation Details:**
```python
def train(self, X_train, y_train, X_val=None, y_val=None):
    self.model = CatBoostClassifier(
        iterations=self.hyperparams.get("iterations", 1000),
        learning_rate=self.hyperparams.get("learning_rate", 0.03),
        depth=self.hyperparams.get("depth", 6),
        l2_leaf_reg=self.hyperparams.get("l2_leaf_reg", 3),
        random_state=self.random_state,
        verbose=False
    )
    
    eval_set = (X_val, y_val) if X_val is not None else None
    
    self.model.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=10 if eval_set else None,
        verbose=False
    )

@classmethod
def load(cls, model_uri: str, config: dict):
    """Load a CatBoost model from MLflow with native predict_proba support."""
    try:
        # Load using CatBoost-specific flavor for better predict_proba support
        model = mlflow.catboost.load_model(model_uri)
        
        # Create a model instance
        instance = cls(config)
        instance.model = model
        instance.is_trained = True
        
        return instance
    except Exception as e:
        raise RuntimeError(f"Failed to load CatBoost model from {model_uri}: {e}")
```

## Training Process
Located in `src/models/train_model.py`

```python
def train_model(model_type):
    config = load_config()
    
    # MLFlow setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    # Load processed data
    train_data = pd.read_csv("data/processed/train_processed.csv")
    X = train_data.drop([config["data"]["target_column"]], axis=1)
    y = train_data[config["data"]["target_column"]]
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y
    )
    
    with mlflow.start_run(nested=True):
        # Create model via factory
        model_instance = ModelFactory.create_model(
            model_type, config, hyperparams
        )
        
        # Train model
        model_instance.train(X_train, y_train, X_val, y_val)
        
        # Evaluate and log
        metrics = model_instance.evaluate(X_val, y_val)
        model_instance.log_metrics(metrics)
        
        # Register model
        model_name = f"{config['project']['name']}-{model_type}"
        model_instance.log_model(X_val, model_name=model_name)
```
!!! warning "Unbalanced dataset"
    The dataset is highly unbalanced, so it's crucial to stratify the train-test split by the target variable (Y).

## MLFlow Integration

### Parameter Logging
All hyperparameters are automatically logged:
```python
mlflow.log_param("model_type", model_type)
mlflow.log_param("random_state", random_state)
for param, value in hyperparams.items():
    mlflow.log_param(param, value)
```

### Metric Logging  
Complete evaluation metrics logged to MLflow:
```python
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision), 
    "recall": float(recall),
    "f1_score": float(f1),
    "auc": float(auc),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp)
}

# Only numerical metrics are logged to MLflow
for metric_name, metric_value in metrics.items():
    if isinstance(metric_value, (int, float)) and metric_value is not None:
        mlflow.log_metric(metric_name, metric_value)
```

### Model Registration
Each trained model is registered in MLFlow model registry using algorithm-specific flavors:
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

## Configuration

Training controlled by `config/config.yaml`:

```yaml
model:
  random_state: 42
  test_size: 0.2
  cv_folds: 5

hyperparameters:
  lightgbm: [hardcoded parameters]
  xgboost: [hardcoded parameters] 
  catboost: [hardcoded parameters]
```

## Execution

### Train Individual Models
```bash
# MLFlow 
mlflow run . -e train -P model_type=lightgbm
mlflow run . -e train -P model_type=xgboost  
mlflow run . -e train -P model_type=catboost

# Python
python -m src.models.train_model --model-type lightgbm
python -m src.models.train_model --model-type xgboost
python -m src.models.train_model --model-type catboost
```

### Train All Models
```bash  
mlflow run . -e main  # Runs complete pipeline including training
```
