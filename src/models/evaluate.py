import argparse
import pandas as pd
import numpy as np
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
import mlflow.pyfunc
import os
import json
from config.config_loader import load_config


def evaluate_model(model_uri, test_data_path):
    config = load_config()

    # Set MLFlow experiment
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Load the model
    print(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # Load test data
    print(f"Loading test data from: {test_data_path}")
    test_data = pd.read_csv(test_data_path)

    # Separate features and target
    target_col = config["data"]["target_column"]
    if target_col not in test_data.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in test data"
        )

    X_test = test_data.drop([target_col], axis=1)
    y_test = test_data[target_col]

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Get prediction probabilities if available
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_pred_proba = None
        print("Warning: Could not get prediction probabilities")

    # Calculate metrics
    print("Calculating evaluation metrics...")

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

    # Log evaluation results
    with mlflow.start_run():
        mlflow.log_param("model_uri", model_uri)
        mlflow.log_param("test_data_path", test_data_path)
        mlflow.log_param("test_samples", len(y_test))

        # Log metrics
        mlflow.log_metric("eval_accuracy", accuracy)
        mlflow.log_metric("eval_precision", precision)
        mlflow.log_metric("eval_recall", recall)
        mlflow.log_metric("eval_f1_score", f1)

        if auc is not None:
            mlflow.log_metric("eval_auc", auc)

        # Log confusion matrix components
        mlflow.log_metric("true_negatives", int(tn))
        mlflow.log_metric("false_positives", int(fp))
        mlflow.log_metric("false_negatives", int(fn))
        mlflow.log_metric("true_positives", int(tp))

        # Save and log detailed results
        evaluation_results = {
            "model_uri": model_uri,
            "test_data_path": test_data_path,
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "auc": float(auc) if auc is not None else None,
            },
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "classification_report": class_report,
        }

        # Save results to file
        results_file = "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        # Log as artifact
        mlflow.log_artifact(results_file)
        os.remove(results_file)

        # Save confusion matrix as CSV
        cm_df = pd.DataFrame(
            cm,
            index=["Actual_0", "Actual_1"],
            columns=["Predicted_0", "Predicted_1"],
        )
        cm_file = "confusion_matrix.csv"
        cm_df.to_csv(cm_file)
        mlflow.log_artifact(cm_file)
        os.remove(cm_file)

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

    return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-uri",
        type=str,
        required=True,
        help="MLFlow model URI (e.g., 'models:/model-name/1' or 'runs:/run-id/model')",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        required=True,
        help="Path to test data CSV file with target column",
    )

    args = parser.parse_args()

    evaluate_model(
        model_uri=args.model_uri, test_data_path=args.test_data_path
    )
