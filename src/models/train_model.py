import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import os
from .core.model_factory import ModelFactory
from config.config_loader import load_config


def train_model(model_type):
    config = load_config()

    # Set MLFlow experiment
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Get hyperparameters from config
    hyperparams = config["hyperparameters"][model_type]
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
        mlflow.log_param("random_state", random_state)
        for param, value in hyperparams.items():
            mlflow.log_param(param, value)

        # Create model using factory pattern
        model_instance = ModelFactory.create_model(
            model_type, config, hyperparams
        )

        # Train model
        model_instance.train(X_train, y_train, X_val, y_val)

        # Evaluate model
        metrics = model_instance.evaluate(X_val, y_val)

        # Log metrics
        model_instance.log_metrics(metrics)

        # Log and Register model
        model_name = f"{config['project']['name']}-{model_type}"
        model_instance.log_model(X_val, model_name=model_name)

        return mlflow.active_run().info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="lightgbm")

    args = parser.parse_args()

    train_model(model_type=args.model_type)
