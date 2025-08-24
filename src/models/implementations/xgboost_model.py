import xgboost as xgb
import mlflow.xgboost
import pandas as pd
import numpy as np
from ..core.base_model import BaseModel
from mlflow.models import infer_signature


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""

    @property
    def model_type(self) -> str:
        """Return the model type name."""
        return "xgboost"

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> None:
        """Train the XGBoost model."""
        self.model = xgb.XGBClassifier(
            n_estimators=self.hyperparams.get("n_estimators", 100),
            learning_rate=self.hyperparams.get("learning_rate", 0.3),
            max_depth=self.hyperparams.get("max_depth", 6),
            min_child_weight=self.hyperparams.get("min_child_weight", 1),
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
        """Make predictions with the XGBoost model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities with the XGBoost model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)

    def log_model(self, X_val: pd.DataFrame = None, model_name: str = None):
        """Log XGBoost model to MLflow."""

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
