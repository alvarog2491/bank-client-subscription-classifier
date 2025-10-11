import xgboost as xgb
import mlflow.xgboost
import pandas as pd
import numpy as np
from .base_model import BaseModel
from typing import Dict, Any, Optional
from mlflow.models import infer_signature


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""

    @property
    def model_type(self) -> str:
        """Model type identifier."""
        return "xgboost"

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Train model with training and validation evaluation sets."""
        if self.trial:
            # Optuna hyperparameter suggestion
            params = {
                "n_estimators": self.trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": self.trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": self.trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": self.trial.suggest_int("min_child_weight", 1, 10),
                "subsample": self.trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": self.trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": self.trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": self.trial.suggest_float("reg_lambda", 0.0, 10.0),
                "scale_pos_weight": self.trial.suggest_float("scale_pos_weight", 1.0, 15.0),
            }
        else:
            # Use provided hyperparameters or defaults
            params = {
                "n_estimators": self.hyperparams.get("n_estimators", 100),
                "learning_rate": self.hyperparams.get("learning_rate", 0.3),
                "max_depth": self.hyperparams.get("max_depth", 6),
                "min_child_weight": self.hyperparams.get("min_child_weight", 1),
            }
            
        self.model = xgb.XGBClassifier(
            **params,
            random_state=self.random_state,
            eval_metric="logloss",
        )

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)

    def log_model(
        self, X_val: pd.DataFrame = None, model_name: str = None
    ) -> None:
        """Log model to MLflow with signature inference and type conversion."""

        if not self.is_trained:
            raise ValueError("Model must be trained before logging")

        X_val_safe = X_val.copy()
        int_cols = X_val_safe.select_dtypes(include="int").columns
        X_val_safe[int_cols] = X_val_safe[int_cols].astype("float64")

        signature = infer_signature(X_val_safe, self.model.predict(X_val_safe))
        input_example = X_val_safe.head(5)

        mlflow.xgboost.log_model(
            xgb_model=self.model,
            name="xgboost_model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
        )

    @classmethod
    def load(cls, model_uri: str, config: Dict[str, Any]):
        """Load model from MLflow using XGBoost-specific flavor for
        predict_proba support."""
        try:
            # Load using XGBoost-specific flavor for better predict_proba
            model = mlflow.xgboost.load_model(model_uri)

            # Create a model instance
            instance = cls(config)
            instance.model = model
            instance.is_trained = True

            return instance
        except Exception as e:
            raise RuntimeError(
                f"Failed to load XGBoost model from {model_uri}: {e}"
            )
