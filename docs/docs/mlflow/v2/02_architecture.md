# Architecture Changes - V2.0

!!! info "Optuna Integration Architecture"
    V2.0 extends the model hierarchy with `trial` parameter support, enabling the replacement of hardcoded values with automated parameter suggestions.
    
## UML Class Diagram - New Components

```mermaid
classDiagram
    class BaseModel {
        <<enhanced>>
        -trial Optional~optuna.Trial~
        +__init__(config, hyperparams, trial)
    }
    
    class ModelFactory {
        <<enhanced>>
        +create_model(model_type, config, hyperparams, trial) BaseModel
    }
    
    class LightGBMModel {
        <<enhanced>>
        +train() uses trial.suggest_int/float() for hyperparams
    }
    
    class XGBoostModel {
        <<enhanced>>
        +train() uses trial.suggest_int/float() for hyperparams
    }
    
    class CatBoostModel {
        <<enhanced>>
        +train() uses trial.suggest_int/float() for hyperparams
    }
    
    class OptunaOptimizer {
        <<new>>
        +optimize_hyperparameters(model_type, X_train, y_train, X_val, y_val, config, n_trials)
        +objective(trial) float
        -best_model BaseModel
        -best_metrics Dict
    }
    
    class OptunaStudy {
        <<external>>
        +create_study(name, storage, direction)
        +optimize(objective, n_trials)
        +best_params Dict
    }
    
    BaseModel <|-- LightGBMModel
    BaseModel <|-- XGBoostModel
    BaseModel <|-- CatBoostModel
    ModelFactory --> BaseModel : enhanced creation
    OptunaOptimizer --> ModelFactory : uses for trials
    OptunaOptimizer --> OptunaStudy : manages optimization
    OptunaStudy --> BaseModel : parameter suggestions
```

## UML Sequence Diagram - Hyperparameter Optimization Pipeline

```mermaid
sequenceDiagram
    participant CLI as Python CLI
    participant TrainFunc as train_model()
    participant Optimizer as optimize_hyperparameters()
    participant Study as Optuna Study
    participant Factory as ModelFactory
    participant Model as BaseModel Instance
    participant MLFlow as MLFlow Server
    participant DB as optuna.db
    
    CLI->>TrainFunc: --optimize True --n-trials 10
    TrainFunc->>MLFlow: start_run() [parent]
    
    rect rgb(255, 248, 240)
        Note over TrainFunc,DB: OPTIMIZATION PHASE
        TrainFunc->>Optimizer: optimize_hyperparameters()
        Optimizer->>DB: create_study(model_type + "_optimization")
        DB-->>Optimizer: persistent study
        
        loop for n_trials
            Optimizer->>Study: objective(trial)
            Study->>Factory: create_model(trial=trial)
            Factory->>Model: __init__(config, hyperparams=None, trial)
            Factory-->>Study: model_instance
            
            Study->>Model: train(X_train, y_train, X_val, y_val)
            Note over Model: uses trial.suggest_int/float()
            Model-->>Study: trained model
            
            Study->>Model: evaluate(X_val, y_val)
            Model-->>Study: metrics["auc"]
            Study->>Optimizer: store if best_model
        end
        
        Optimizer-->>TrainFunc: best_params, best_model, best_metrics
    end
    
    rect rgb(245, 255, 245)
        Note over TrainFunc,MLFlow: FINAL LOGGING PHASE
        TrainFunc->>MLFlow: log_param("best_" + param) for each
        TrainFunc->>Model: log_metrics(best_metrics)
        TrainFunc->>Model: log_model(X_val, model_name)
        Model->>MLFlow: register_model()
    end
    
    TrainFunc-->>CLI: run_id
```

## New Components Added

V2.0 extends the existing architecture with two new classes:

- **OptunaOptimizer**: Manages the hyperparameter search process
- **OptunaStudy**: External Optuna study for persistent optimization history

## Storage Architecture

### Persistent Optimization Studies
- **Database**: `optuna.db` (SQLite) for study persistence
- **Study Names**: `{model_type}_optimization` (e.g., `lightgbm_optimization`)
- **Knowledge Accumulation**: Trials build upon previous optimization history
- **Concurrent Safe**: Multiple training sessions can add trials to existing studies