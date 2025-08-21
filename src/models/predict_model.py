import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import yaml
import os

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def predict_model(model_uri, input_path, output_path):
    config = load_config()
    
    # Set MLFlow experiment
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Load the model
    print(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Load input data
    print(f"Loading data from: {input_path}")
    input_data = pd.read_csv(input_path)
    
    # Store ID column if it exists
    id_col = config['data']['id_column']
    if id_col in input_data.columns:
        ids = input_data[id_col].copy()
        # Remove ID column for prediction
        prediction_data = input_data.drop([id_col], axis=1)
    else:
        ids = None
        prediction_data = input_data.copy()
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(prediction_data)
    
    # Create predictions DataFrame
    if ids is not None:
        predictions_df = pd.DataFrame({
            id_col: ids,
            'prediction': predictions
        })
    else:
        predictions_df = pd.DataFrame({
            'prediction': predictions
        })
    
    # Save predictions
    print(f"Saving predictions to: {output_path}")
    predictions_df.to_csv(output_path, index=False)
    
    print(f"Predictions completed! Results saved to {output_path}")
    print(f"Number of predictions: {len(predictions)}")
    
    # Log prediction summary
    with mlflow.start_run():
        mlflow.log_param("model_uri", model_uri)
        mlflow.log_param("input_path", input_path)
        mlflow.log_param("output_path", output_path)
        mlflow.log_metric("num_predictions", len(predictions))
        
        # Log prediction distribution
        unique_preds, counts = np.unique(predictions, return_counts=True)
        for pred, count in zip(unique_preds, counts):
            mlflow.log_metric(f"prediction_class_{pred}_count", count)
        
        # Log the predictions file as artifact
        mlflow.log_artifact(output_path)
    
    return predictions_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", type=str, required=True, 
                       help="MLFlow model URI (e.g., 'models:/model-name/1' or 'runs:/run-id/model')")
    parser.add_argument("--input-path", type=str, required=True,
                       help="Path to input CSV file for prediction")
    parser.add_argument("--output-path", type=str, default="predictions.csv",
                       help="Path to save predictions CSV file")
    
    args = parser.parse_args()
    
    predict_model(
        model_uri=args.model_uri,
        input_path=args.input_path,
        output_path=args.output_path
    )