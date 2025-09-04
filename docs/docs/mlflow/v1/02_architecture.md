# Project Architecture

## UML Class Diagram

!!! info "Extensible Architecture"
    The system uses the **Factory Pattern** combined with **configuration-driven design** to make adding new machine learning models effortless. The `ModelFactory` maintains a registry of available models, while hyperparameters are centralized in `config.yaml`. To add a new model (e.g., RandomForest), you simply:
    
    1. Create a new class inheriting from `BaseModel` 
    2. Add it to the factory's `_models` dictionary
    3. Define its hyperparameters in the config file
    4. The model immediately becomes available through MLflow entry points
    
    This design separates algorithm implementation from pipeline orchestration, enabling rapid experimentation without modifying core training logic.

```mermaid

classDiagram
    class BaseModel {
        <<abstract>>
        -config: Dict
        -hyperparams: Dict
        -model: Any
        -is_trained: bool
        +train(X_train, y_train, X_val, y_val)*
        +predict(X)*
        +predict_proba(X)*
        +evaluate(X_test, y_test): Dict
        +log_metrics(metrics): void
        +log_model(X_val, model_name)*
        +model_type*: str
    }
    
    class LightGBMModel {
        +train(X_train, y_train, X_val, y_val)
        +predict(X)
        +predict_proba(X)
        +log_model(X_val, model_name)
        +model_type: str
    }
    
    class XGBoostModel {
        +train(X_train, y_train, X_val, y_val)
        +predict(X)
        +predict_proba(X)
        +log_model(X_val, model_name)
        +model_type: str
    }
    
    class CatBoostModel {
        +train(X_train, y_train, X_val, y_val)
        +predict(X)
        +predict_proba(X)
        +log_model(X_val, model_name)
        +model_type: str
    }
    
    class ModelFactory {
        <<static>>
        -_models: Dict[str, BaseModel]
        +create_model(model_type, config, hyperparams): BaseModel
        +get_supported_models(): List[str]
    }
    
    class DataLoader {
        <<utility>>
        +load_train_data(): DataFrame
        +load_test_data(): DataFrame
        +load_processed_data(): Tuple[DataFrame, DataFrame]
        +load_label_encoders(): Dict[str, LabelEncoder]
    }
    
    class DataPreprocessor {
        <<utility>>
        +preprocess_data(): Tuple[DataFrame, DataFrame]
    }
    
    class TrainModel {
        <<utility>>
        +train_model(model_type): str
    }
    
    BaseModel <|-- LightGBMModel
    BaseModel <|-- XGBoostModel
    BaseModel <|-- CatBoostModel
    ModelFactory --> BaseModel : creates
    TrainModel --> ModelFactory : uses
    TrainModel --> DataLoader : uses
    DataPreprocessor --> DataLoader : uses
```



## UML Sequence Diagram - Training Pipeline

```mermaid
sequenceDiagram
    participant MLProject as MLproject
    participant TrainFunc as train_model()
    participant Config as config_loader
    participant DataLoader as Data Functions
    participant Factory as ModelFactory
    participant Model as BaseModel Instance
    participant MLFlow as MLFlow Server
    
    MLProject->>TrainFunc: mlflow run . -e train
    TrainFunc->>Config: load_config()
    Config-->>TrainFunc: config dict
    
    rect rgb(240, 248, 255)
        Note over MLProject,DataLoader: DATA LOADING PHASE
        TrainFunc->>DataLoader: load_processed_data()
        DataLoader-->>TrainFunc: train_processed.csv
        TrainFunc->>TrainFunc: train_test_split(X, y)
    end
    
    rect rgb(245, 255, 245)
        Note over TrainFunc,MLFlow: MODEL TRAINING PHASE
        TrainFunc->>MLFlow: set_experiment()
        TrainFunc->>MLFlow: start_run()
        TrainFunc->>MLFlow: log_param(model_type, hyperparams...)
        
        TrainFunc->>Factory: create_model(model_type, config, hyperparams)
        Factory->>Model: __init__(config, hyperparams)
        Factory-->>TrainFunc: model_instance
        
        TrainFunc->>Model: train(X_train, y_train, X_val, y_val)
        Model-->>TrainFunc: trained model
    end
    
    rect rgb(255, 248, 240)
        Note over TrainFunc,MLFlow: EVALUATION & LOGGING PHASE
        TrainFunc->>Model: evaluate(X_val, y_val)
        Model->>Model: calculate metrics
        Model-->>TrainFunc: metrics dict
        
        TrainFunc->>Model: log_metrics(metrics)
        Model->>MLFlow: log_metric() for each metric
        
        TrainFunc->>Model: log_model(X_val, model_name)
        Model->>MLFlow: log_model() & register_model()
    end
    
    TrainFunc-->>MLProject: run_id
```

## Module Structure

```
src/
├── data/
│   ├── __init__.py
│   ├── load_data.py         # Data loading functions
│   └── preprocess.py        # Data preprocessing functions  
├── models/
│   ├── __init__.py
│   ├── core/
│   │   ├── base_model.py    # BaseModel abstract class
│   │   └── model_factory.py # ModelFactory static class
│   ├── implementations/
│   │   ├── catboost_model.py    # CatBoostModel class
│   │   ├── lightgbm_model.py    # LightGBMModel class
│   │   └── xgboost_model.py     # XGBoostModel class
│   ├── predict_model.py     # Prediction functions
│   └── train_model.py       # Training functions
│
config/
├── config.yaml              # Main configuration file
└── config_loader.py         # Configuration loading functions

MLproject                   # MLFlow project definition
main.py                     # Main pipeline runner
```

## Design Patterns Used

### Factory Pattern
- **ModelFactory**: Creates model instances based on string identifiers
- Static method approach with dictionary mapping
- Supports easy addition of new model types
- Encapsulates model instantiation logic

### Template Method Pattern  
- **BaseModel**: Abstract class defining training/prediction workflow
- Concrete implementations provide algorithm-specific details
- Ensures consistent interface across all models
- Common evaluation logic in base class

### Strategy Pattern
- **Model Implementations**: Interchangeable algorithms (LightGBM, XGBoost, CatBoost)
- Runtime selection of training strategy based on configuration
- Consistent evaluation metrics across strategies

