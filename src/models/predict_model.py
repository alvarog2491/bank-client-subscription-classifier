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


def _resolve_model_uri(model_uri, model_name):
    """Resolve model URI from model_name or validate existing model_uri."""
    if model_name and not model_uri:
        model_uri = f"models:/{model_name}/latest"
        print(f"Using model from registry: {model_name} (latest version)")
        return model_uri
    elif model_uri:
        print(f"Using provided model URI: {model_uri}")
        return model_uri
    else:
        raise ValueError("Either model_uri or model_name must be provided")


def _resolve_paths(config, input_path, output_path):
    """Resolve input and output paths, using defaults if not provided."""
    if input_path is None:
        input_path = (
            f"{config['data']['processed_data_path']}test_processed.csv"
        )
        print(f"Using default input path: {input_path}")

    if output_path is None:
        output_path = "data/predictions/test_predictions.csv"
        print(f"Using default output path: {output_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    return input_path, output_path


def _load_model(model_uri, model_type, config):
    """Load model from MLflow with proper predict_proba support."""
    print(f"Loading model from: {model_uri}")

    # Check if model type is in supported models list from ModelFactory
    if model_type and model_type in ModelFactory.get_supported_models():
        try:
            print(
                f"Loading as {model_type} model with native predict_proba support"
            )
            return ModelFactory.load_model(model_type, model_uri, config)
        except Exception as e:
            print(f"Failed to load with {model_type} loader: {e}")

    # Fallback to pyfunc
    print("Loading with pyfunc interface")
    return mlflow.pyfunc.load_model(model_uri)


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
    model_uri=None, model_name=None, input_path=None, output_path=None
):
    """Generate predictions using MLflow model.

    Args:
        model_uri: MLflow model URI (e.g., 'runs:/run-id/model')
        model_name: Registered model name (uses latest version)
        input_path: Path to input CSV file
        output_path: Path to save predictions CSV file

    Returns:
        pd.DataFrame: Predictions dataframe
    """
    config = load_config()

    # Setup MLflow
    _setup_mlflow(config)

    # Resolve model URI and paths
    model_uri = _resolve_model_uri(model_uri, model_name)
    input_path, output_path = _resolve_paths(config, input_path, output_path)

    # Determine model type from model_name or model_uri
    model_type = None
    if model_name:
        model_name_lower = model_name.lower()
        supported_models = ModelFactory.get_supported_models()
        model_type = next(
            (m for m in supported_models if m in model_name_lower), None
        )
    elif model_uri:
        model_uri_lower = model_uri.lower()
        supported_models = ModelFactory.get_supported_models()
        model_type = next(
            (m for m in supported_models if m in model_uri_lower), None
        )

    # Load model and data
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

    # Model specification (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-uri",
        type=str,
        help="MLFlow model URI (e.g., 'models:/model-name/1' or 'runs:/run-id/model')",
    )
    model_group.add_argument(
        "--model-name",
        type=str,
        help="MLFlow registered model name (uses latest version)",
    )

    parser.add_argument(
        "--input-path",
        type=str,
        help="Path to input CSV file (default: data/processed/test_processed.csv)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to save predictions CSV file (default: data/predictions/test_predictions.csv)",
    )

    args = parser.parse_args()

    predict_model(
        model_uri=args.model_uri,
        model_name=args.model_name,
        input_path=args.input_path,
        output_path=args.output_path,
    )
