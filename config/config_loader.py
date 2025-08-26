import os
import yaml
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """Load project configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)