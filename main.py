import mlflow


def main():
    """Run the complete pipeline"""

    print("Running data preprocessing...")
    mlflow.run(".", entry_point="data_preprocessing")

    print("Running feature engineering...")
    mlflow.run(".", entry_point="feature_engineering")

    print("Training model...")
    mlflow.run(".", entry_point="train")

    print("Evaluating model...")
    mlflow.run(".", entry_point="evaluate")

    print("Pipeline complete")


if __name__ == "__main__":
    main()
