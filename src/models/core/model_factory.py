from typing import Dict, Any, Union, Optional
import optuna
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
        hyperparams: Optional[Union[Dict[str, Any], optuna.Trial]] = None,
        trial: Optional[optuna.Trial] = None,
    ) -> BaseModel:
        """Create model instance using factory pattern.

        Args:
            model_type: Type of model to create
            config: Configuration dictionary
            hyperparams: Either hyperparameters dict or Optuna trial for optimization
            trial: Optuna trial (alternative way to pass trial)

        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls._models:
            supported_models = list(cls._models.keys())
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported models: {supported_models}"
            )

        # Handle trial parameter - support both ways for flexibility
        effective_trial = trial if trial is not None else hyperparams if isinstance(hyperparams, optuna.Trial) else None
        effective_hyperparams = hyperparams if not isinstance(hyperparams, optuna.Trial) else None

        model_class = cls._models[model_type]
        return model_class(config, effective_hyperparams, effective_trial)

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
