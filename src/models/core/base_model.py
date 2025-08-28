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
from sklearn.model_selection import StratifiedKFold
import mlflow


class BaseModel(ABC):
    """Abstract base class for all machine learning models."""

    def __init__(
        self,
        config: Dict[str, Any],
        hyperparams: Dict[str, Any] = None,
        trial: Optional[optuna.Trial] = None,
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

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
    ) -> Dict[str, float]:
        """Perform K-fold cross-validation and return performance metrics."""
        print(f"Performing {cv_folds}-fold cross-validation...")

        # Create stratified K-fold to maintain class distribution
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Initialize lists to store metrics for each fold
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_aucs = []

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"  Fold {fold}/{cv_folds}...")

            # Split data for this fold
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # Train model on fold
            self.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

            # Evaluate on validation set
            fold_metrics = self.evaluate(X_val_fold, y_val_fold)

            # Store metrics
            fold_accuracies.append(fold_metrics["accuracy"])
            fold_precisions.append(fold_metrics["precision"])
            fold_recalls.append(fold_metrics["recall"])
            fold_f1s.append(fold_metrics["f1_score"])
            if "auc" in fold_metrics:
                fold_aucs.append(fold_metrics["auc"])

            fold_results.append(
                {
                    "fold": fold,
                    "accuracy": fold_metrics["accuracy"],
                    "precision": fold_metrics["precision"],
                    "recall": fold_metrics["recall"],
                    "auc": fold_metrics.get("auc", None),
                }
            )

        # Calculate mean and std for each metric
        cv_results = {
            "cv_accuracy_mean": np.mean(fold_accuracies),
            "cv_accuracy_std": np.std(fold_accuracies),
            "cv_precision_mean": np.mean(fold_precisions),
            "cv_precision_std": np.std(fold_precisions),
            "cv_recall_mean": np.mean(fold_recalls),
            "cv_recall_std": np.std(fold_recalls),
            "cv_f1_mean": np.mean(fold_f1s),
            "cv_f1_std": np.std(fold_f1s),
        }

        if fold_aucs:
            cv_results.update(
                {
                    "cv_auc_mean": np.mean(fold_aucs),
                    "cv_auc_std": np.std(fold_aucs),
                }
            )

        # Print results summary
        print("\nCROSS-VALIDATION RESULTS")
        print("=" * 60)
        print(
            f"Accuracy:  {cv_results['cv_accuracy_mean']:.4f} ± "
            f"{cv_results['cv_accuracy_std']:.4f}"
        )
        print(
            f"Precision: {cv_results['cv_precision_mean']:.4f} ± "
            f"{cv_results['cv_precision_std']:.4f}"
        )
        print(
            f"Recall:    {cv_results['cv_recall_mean']:.4f} ± "
            f"{cv_results['cv_recall_std']:.4f}"
        )
        print(
            f"F1 Score:  {cv_results['cv_f1_mean']:.4f} ± "
            f"{cv_results['cv_f1_std']:.4f}"
        )
        if "cv_auc_mean" in cv_results:
            print(
                f"AUC:       {cv_results['cv_auc_mean']:.4f} ± "
                f"{cv_results['cv_auc_std']:.4f}"
            )
        print("=" * 60)

        return cv_results

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
