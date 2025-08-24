import argparse
import os
import pandas as pd
import mlflow
import mlflow.pyfunc
from config.config_loader import load_config
from src.models.core.model_factory import ModelFactory


def _setup_mlflow(config):
    """Setup MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def _validate_model_uri(model_uri):
    """Validate that model_uri is provided."""
    if not model_uri:
        raise ValueError("model_uri is required")

    print(f"Using model URI: {model_uri}")
    return model_uri


def _validate_paths(input_path, output_path):
    """Validate that required paths are provided and create output directory."""
    if input_path is None:
        raise ValueError("input_path is required")

    if output_path is None:
        raise ValueError("output_path is required")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    return input_path, output_path


def _load_model(model_uri, model_type, config):
    """Load model from MLflow with proper predict_proba support.

    Args:
        model_uri: MLflow model URI
        model_type: Model type (required) - one of: lightgbm, xgboost, catboost
        config: Configuration dictionary

    Returns:
        Loaded model instance

    Raises:
        ValueError: If model_type is not provided or not supported
    """
    print(f"Loading model from: {model_uri}")

    if not model_type:
        raise ValueError(
            "model_type is required. "
            "Supported types: lightgbm, xgboost, catboost"
        )

    supported_models = ModelFactory.get_supported_models()
    if model_type not in supported_models:
        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            f"Supported types: {supported_models}"
        )

    try:
        print(
            f"Loading as {model_type} model with native predict_proba support"
        )
        return ModelFactory.load_model(model_type, model_uri, config)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {model_type} model from {model_uri}: {e}"
        )


def _prepare_prediction_data(input_data, config, model):
    """Prepare data for prediction based on model's expected input schema."""
    id_col = config["data"]["id_column"]

    # Get model's expected input schema if available
    try:
        model_schema = model.metadata.get_input_schema()
        expected_columns = (
            [col.name for col in model_schema.inputs] if model_schema else None
        )
    except:
        expected_columns = None

    if id_col in input_data.columns:
        ids = input_data[id_col].copy()

        # If model expects id column, keep it; otherwise remove it
        if expected_columns and id_col in expected_columns:
            print(
                f"Model expects '{id_col}' column - keeping it in prediction data"
            )
            prediction_data = input_data.copy()
            # Convert all numeric columns to float64 to match model schema
            numeric_columns = prediction_data.select_dtypes(
                include=["int64"]
            ).columns
            for col in numeric_columns:
                prediction_data[col] = prediction_data[col].astype("float64")
        else:
            print(
                f"Model doesn't expect '{id_col}' column - removing it from prediction data"
            )
            prediction_data = input_data.drop([id_col], axis=1)
            # Still need to convert numeric types for remaining columns
            numeric_columns = prediction_data.select_dtypes(
                include=["int64"]
            ).columns
            for col in numeric_columns:
                prediction_data[col] = prediction_data[col].astype("float64")

        return prediction_data, ids, id_col
    else:
        return input_data.copy(), None, id_col


def _create_predictions_dataframe(predictions, ids, id_col):
    """Create predictions DataFrame with proper column structure for Kaggle submission."""
    if ids is not None:
        return pd.DataFrame({id_col: ids, "y": predictions})
    else:
        return pd.DataFrame({"y": predictions})


def _save_predictions(predictions_df, output_path):
    """Save predictions to CSV file."""
    print(f"Saving predictions to: {output_path}")
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions completed! Results saved to {output_path}")
    print(f"Number of predictions: {len(predictions_df)}")


def predict_model(
    model_uri=None, model_type=None, input_path=None, output_path=None
):
    """Generate predictions using MLflow model.

    Args:
        model_uri: MLflow model URI (required) - e.g., 'models:/model-name/1' or 'runs:/run-id/model'
        model_type: Model type (required) - one of: lightgbm, xgboost, catboost
        input_path: Path to input CSV file (required)
        output_path: Path to save predictions CSV file (required)

    Returns:
        pd.DataFrame: Predictions dataframe

    Raises:
        ValueError: If required parameters are not provided
    """
    config = load_config()

    # Setup MLflow
    _setup_mlflow(config)

    # Validate model URI and paths
    model_uri = _validate_model_uri(model_uri)
    input_path, output_path = _validate_paths(input_path, output_path)

    # Load model with explicit model type
    model = _load_model(model_uri, model_type, config)

    print(f"Loading data from: {input_path}")
    input_data = pd.read_csv(input_path)

    # Prepare data for prediction
    prediction_data, ids, id_col = _prepare_prediction_data(
        input_data, config, model
    )

    # Make probability predictions
    print("Making probability predictions...")
    try:
        # Get probabilities for the positive class (index 1)
        probabilities = model.predict_proba(prediction_data)[:, 1]
        # Round to 3 decimal places
        predictions = probabilities.round(3)
        print("Generated probabilities for positive class (y=1)")
    except AttributeError:
        # Fallback to regular predictions if predict_proba is not available
        print(
            "Model doesn't support predict_proba, using regular predictions..."
        )
        predictions = model.predict(prediction_data)

    # Create and save predictions DataFrame
    predictions_df = _create_predictions_dataframe(predictions, ids, id_col)
    _save_predictions(predictions_df, output_path)

    return predictions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate predictions using MLflow model"
    )

    parser.add_argument(
        "--model-uri",
        type=str,
        required=True,
        help="MLFlow model URI (e.g., 'models:/model-name/1' or 'runs:/run-id/model')",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["lightgbm", "xgboost", "catboost"],
        help="Model type (required)",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save predictions CSV file",
    )

    args = parser.parse_args()

    predict_model(
        model_uri=args.model_uri,
        model_type=args.model_type,
        input_path=args.input_path,
        output_path=args.output_path,
    )
