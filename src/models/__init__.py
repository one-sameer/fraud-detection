"""
src/models/__init__.py
──────────────────────
Central registry for all base models.

Usage:
    from src.models import get_all_models, load_model

    models = get_all_models(input_dim=31)
    # → { "logistic_regression": LRModel, "random_forest": RFModel, ... }
"""

from __future__ import annotations

from src.models.classical import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
)
from src.models.mlp import MLPModel
from src.models.autoencoder import AutoencoderModel


MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "mlp",
    "autoencoder",
]

# Models used in the stacking ensemble.
# The autoencoder is excluded: its AUPRC is too low to contribute positively
# to the meta-learner. It remains trained and saved for standalone analysis
# but is not included in the ensemble feature matrix.
ENSEMBLE_MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "mlp",
]


def get_all_models(input_dim: int = 31) -> dict:
    """Instantiates all six base models with default hyperparameters."""
    return {
        "logistic_regression": LogisticRegressionModel(),
        "random_forest":       RandomForestModel(),
        "xgboost":             XGBoostModel(),
        "lightgbm":            LightGBMModel(),
        "mlp":                 MLPModel(input_dim=input_dim),
        "autoencoder":         AutoencoderModel(input_dim=input_dim),
    }


_LOADERS = {
    "logistic_regression": LogisticRegressionModel.load,
    "random_forest":       RandomForestModel.load,
    "xgboost":             XGBoostModel.load,
    "lightgbm":            LightGBMModel.load,
    "mlp":                 MLPModel.load,
    "autoencoder":         AutoencoderModel.load,
}


def load_model(name: str, path: str):
    """Loads a saved model by name from path."""
    if name not in _LOADERS:
        raise ValueError(f"Unknown model '{name}'. Valid: {list(_LOADERS)}")
    return _LOADERS[name](path)
