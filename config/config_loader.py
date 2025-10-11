import os
import yaml
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """
    Load project configuration from model_config.yaml and hyperparameters.yaml.

    Returns:
        Dict containing merged configuration from both files.
    """
    config_dir = os.path.dirname(__file__)

    # Load model configuration
    model_config_path = os.path.join(config_dir, "model_config.yaml")
    with open(model_config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load hyperparameters
    hyperparams_path = os.path.join(config_dir, "hyperparameters.yaml")
    with open(hyperparams_path, "r") as file:
        hyperparams = yaml.safe_load(file)

    # Merge hyperparameters into config
    config.update(hyperparams)

    return config


def load_model_config() -> Dict[str, Any]:
    """Load model configuration from model_config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "model_config.yaml")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_hyperparameters() -> Dict[str, Any]:
    """Load hyperparameters from hyperparameters.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "hyperparameters.yaml")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)