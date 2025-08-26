from typing import Dict, Any
from .base_model import BaseModel
from ..implementations.lightgbm_model import LightGBMModel
from ..implementations.xgboost_model import XGBoostModel
from ..implementations.catboost_model import CatBoostModel


class ModelFactory:
    """Factory class for creating different types of ML models."""

    _models = {
        "lightgbm": LightGBMModel,
        "xgboost": XGBoostModel,
        "catboost": CatBoostModel,
    }

    @classmethod
    def create_model(
        cls,
        model_type: str,
        config: Dict[str, Any],
        hyperparams: Dict[str, Any] = None,
    ) -> BaseModel:
        """Create model instance using factory pattern.

        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls._models:
            supported_models = list(cls._models.keys())
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported models: {supported_models}"
            )

        model_class = cls._models[model_type]
        return model_class(config, hyperparams)

    @classmethod
    def get_supported_models(cls) -> list:
        """Return list of supported model types."""
        return list(cls._models.keys())

    @classmethod
    def load_model(
        cls, model_type: str, model_uri: str, config: Dict[str, Any]
    ) -> BaseModel:
        """Load model from MLflow using algorithm-specific flavor
        for predict_proba support.

        Validates model_type and delegates to appropriate model class loader.
        """
        if model_type not in cls._models:
            supported_models = list(cls._models.keys())
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported models: {supported_models}"
            )

        model_class = cls._models[model_type]
        return model_class.load(model_uri, config)
