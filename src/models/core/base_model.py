from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import mlflow


class BaseModel(ABC):
    """Abstract base class for all machine learning models."""

    def __init__(
        self, 
        config: Dict[str, Any], 
        hyperparams: Dict[str, Any] = None,
        trial: Optional[optuna.Trial] = None
    ):
        self.config = config
        self.hyperparams = hyperparams or {}
        self.trial = trial
        self.random_state = config.get("model", {}).get("random_state", 42)
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        pass

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model with comprehensive metrics and print results.

        Computes accuracy, precision, recall, F1, AUC (if possible),
        and confusion matrix. Handles cases where predict_proba
          is not available.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Make predictions
        y_pred = self.predict(X_test)

        # Get prediction probabilities if available
        try:
            y_pred_proba = self.predict_proba(X_test)[:, 1]
        except (AttributeError, IndexError):
            y_pred_proba = None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")

        if y_pred_proba is not None:
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = None

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc) if auc is not None else None,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "classification_report": class_report,
        }

        # Print results
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        if auc is not None:
            print(f"AUC:       {auc:.4f}")
        print("\nConfusion Matrix:")
        print(f"TN: {tn:4d} | FP: {fp:4d}")
        print(f"FN: {fn:4d} | TP: {tp:4d}")
        print("=" * 50)

        return metrics

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log numerical metrics to MLflow (excludes complex objects)."""
        for metric_name, metric_value in metrics.items():
            if (
                isinstance(metric_value, (int, float))
                and metric_value is not None
            ):
                mlflow.log_metric(metric_name, metric_value)

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Model type identifier."""
        pass

    @abstractmethod
    def log_model(
        self,
        X_val: pd.DataFrame = None,
    ) -> None:
        """Log model to MLflow with signature inference."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_uri: str, config: Dict[str, Any]):
        """Load model from MLflow with predict_proba support."""
        pass
