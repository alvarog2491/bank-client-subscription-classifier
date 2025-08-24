from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow


class BaseModel(ABC):
    """Abstract base class for all machine learning models."""

    def __init__(
        self, config: Dict[str, Any], hyperparams: Dict[str, Any] = None
    ):
        self.config = config
        self.hyperparams = hyperparams or {}
        self.random_state = config.get("model", {}).get("random_state", 42)
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> None:
        """Train the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on the provided data."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the provided data."""
        pass

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba),
        }

        print(metrics)

        return metrics

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow."""
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type name."""
        pass

    @abstractmethod
    def log_model(
        self,
        X_val: pd.DataFrame = None,
    ):
        """Log model to MLflow."""
        pass
