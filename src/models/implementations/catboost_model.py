from catboost import CatBoostClassifier
import mlflow.catboost
import pandas as pd
import numpy as np
from ..core.base_model import BaseModel
from mlflow.models import infer_signature


class CatBoostModel(BaseModel):
    """CatBoost model implementation."""

    @property
    def model_type(self) -> str:
        """Return the model type name."""
        return "catboost"

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> None:
        """Train the CatBoost model."""
        self.model = CatBoostClassifier(
            iterations=self.hyperparams.get("iterations", 1000),
            learning_rate=self.hyperparams.get("learning_rate", 0.03),
            depth=self.hyperparams.get("depth", 6),
            l2_leaf_reg=self.hyperparams.get("l2_leaf_reg", 3),
            random_state=self.random_state,
            verbose=False,
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=10 if eval_set else None,
            verbose=False,
        )
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the CatBoost model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities with the CatBoost model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)

    def log_model(self, X_val: pd.DataFrame = None, model_name: str = None):
        """Log CatBoost model to MLflow."""

        if not self.is_trained:
            raise ValueError("Model must be trained before logging")

        X_val_safe = X_val.copy()
        int_cols = X_val_safe.select_dtypes(include="int").columns
        X_val_safe[int_cols] = X_val_safe[int_cols].astype("float64")

        signature = infer_signature(X_val_safe, self.model.predict(X_val_safe))
        input_example = X_val_safe.head(5)

        mlflow.catboost.log_model(
            cb_model=self.model,
            name="catboost_model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
        )

    @classmethod
    def load(cls, model_uri: str, config: dict):
        """Load a CatBoost model from MLflow with proper predict_proba support."""
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
