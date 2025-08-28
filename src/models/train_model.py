import argparse
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import optuna
import os
from .core.model_factory import ModelFactory
from .core.base_model import BaseModel
from config.config_loader import load_config


def optimize_hyperparameters(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Dict[str, Any],
    n_trials: int,
) -> Tuple[Dict[str, Any], BaseModel, Dict[str, float]]:
    """Run hyperparameter optimization using Optuna with MLFlow child runs."""

    best_model = None
    best_metrics = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_model, best_metrics

        # Create model with trial suggestions - delegate to model factory
        model_instance = ModelFactory.create_model(
            model_type, config, trial=trial
        )

        # Train and evaluate
        model_instance.train(X_train, y_train, X_val, y_val)
        metrics = model_instance.evaluate(X_val, y_val)

        # Store best model and metrics
        if best_model is None or metrics["auc"] > best_metrics["auc"]:
            best_model = model_instance
            best_metrics = metrics

        return metrics["auc"]

    # Create or load persistent study using SQLite storage
    study_name = f"{model_type}_optimization"
    storage_url = "sqlite:///optuna.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, best_model, best_metrics


def train_model(
    model_type: str, optimize: bool = False, n_trials: int = 10
) -> str:
    """Train model with full MLflow pipeline.

    Performs setup, data splitting, training, evaluation,
    and model registration with optional hyperparameter optimization.
    """
    config = load_config()

    # Set MLFlow experiment
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    random_state = config["model"]["random_state"]

    # Load processed data
    processed_path = config["data"]["processed_data_path"]
    train_data = pd.read_csv(
        os.path.join(processed_path, "train_processed.csv")
    )

    # Separate features and target
    target_col = config["data"]["target_column"]
    X = train_data.drop([target_col], axis=1)
    y = train_data[target_col]

    # Split data
    test_size = config["model"]["test_size"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    with mlflow.start_run(nested=True):
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("optimize", optimize)
        mlflow.log_param("random_state", random_state)

        if optimize:
            mlflow.log_param("n_trials", n_trials)

            # Run hyperparameter optimization and get best model
            best_params, model_instance, metrics = optimize_hyperparameters(
                model_type, X_train, y_train, X_val, y_val, config, n_trials
            )

            # Log best parameters to parent run
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param}", value)
        else:
            # Use default hyperparameters from config
            hyperparams = config["hyperparameters"][model_type]
            model_instance = ModelFactory.create_model(
                model_type, config, hyperparams
            )

            # Log default parameters
            for param, value in hyperparams.items():
                mlflow.log_param(param, value)

            # Train and evaluate model
            model_instance.train(X_train, y_train, X_val, y_val)
            metrics = model_instance.evaluate(X_val, y_val)

        # Log individual evaluation metrics (including confusion matrix)
        model_instance.log_metrics(metrics)
        
        # Perform K-fold cross-validation for robust evaluation
        cv_folds = config["model"]["cv_folds"]
        cv_results = model_instance.cross_validate(X, y, cv_folds)

        # Log cross-validation metrics
        model_instance.log_metrics(cv_results)

        # Log and Register model
        model_name = f"{config['project']['name']}-{model_type}"
        model_instance.log_model(X_val, model_name=model_name)

        return mlflow.active_run().info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="lightgbm")
    parser.add_argument("--optimize", type=bool, default=False)
    parser.add_argument("--n-trials", type=int, default=10)

    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        optimize=args.optimize,
        n_trials=args.n_trials,
    )
