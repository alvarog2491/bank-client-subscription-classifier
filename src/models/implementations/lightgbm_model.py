import lightgbm as lgb
import mlflow.lightgbm
import pandas as pd
import numpy as np
from ..core.base_model import BaseModel
from typing import Dict, Any
from mlflow.models import infer_signature


class LightGBMModel(BaseModel):
    """LightGBM model implementation."""

    @property
    def model_type(self) -> str:
        """Model type identifier."""
        return "lightgbm"

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> None:
        """Train model with optional early stopping on validation data."""
        self.model = lgb.LGBMClassifier(
            n_estimators=self.hyperparams.get("n_estimators", 100),
            learning_rate=self.hyperparams.get("learning_rate", 0.1),
            max_depth=self.hyperparams.get("max_depth", -1),
            num_leaves=self.hyperparams.get("num_leaves", 31),
            min_child_samples=self.hyperparams.get("min_child_samples", 20),
            random_state=self.random_state,
            verbose=-1,
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(10)] if eval_set else None,
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
        """Log model to MLflow with signature inference and type conversion.

        Converts int columns to float64 for MLflow compatibility.
        """

        if not self.is_trained:
            raise ValueError("Model must be trained before logging")

        X_val_safe = X_val.copy()
        int_cols = X_val_safe.select_dtypes(include="int").columns
        X_val_safe[int_cols] = X_val_safe[int_cols].astype("float64")

        signature = infer_signature(X_val_safe, self.model.predict(X_val_safe))
        input_example = X_val_safe.head(5)

        mlflow.lightgbm.log_model(
            lgb_model=self.model,
            name="lightgbm_model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
        )

    @classmethod
    def load(cls, model_uri: str, config: Dict[str, Any]):
        """Load model from MLflow using LightGBM-specific flavor
        for predict_proba support."""
        try:
            # Load using LightGBM-specific flavor for better predict_proba
            model = mlflow.lightgbm.load_model(model_uri)

            # Create a model instance
            instance = cls(config)
            instance.model = model
            instance.is_trained = True

            return instance
        except Exception as e:
            raise RuntimeError(
                f"Failed to load LightGBM model from {model_uri}: {e}"
            )
