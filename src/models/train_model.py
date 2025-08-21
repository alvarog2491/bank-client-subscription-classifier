import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.catboost
import yaml
import os


def load_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "config.yaml"
    )
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def train_model(
    model_type,
    n_estimators,
    learning_rate,
    max_depth,
    random_state,
):
    config = load_config()

    # Set MLFlow experiment
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

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

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        # Train model based on type
        if model_type == "lightgbm":
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                verbose=-1,
            )
            model.fit(X_train, y_train)
            mlflow.lightgbm.log_model(model, "model")

        elif model_type == "xgboost":
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train)
            mlflow.xgboost.log_model(model, "model")

        elif model_type == "catboost":
            model = CatBoostClassifier(
                iterations=n_estimators,
                learning_rate=learning_rate,
                depth=max_depth,
                random_state=random_state,
                verbose=False,
            )
            model.fit(X_train, y_train)
            mlflow.catboost.log_model(model, "model")

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)

        # Log feature importance if available
        if hasattr(model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            os.remove("feature_importance.csv")

        print("Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")

        # Register model
        model_name = f"{config['project']['name']}-{model_type}"
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model", model_name
        )

        return mlflow.active_run().info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="lightgbm")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
