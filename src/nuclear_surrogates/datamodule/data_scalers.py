"""Scaler registry.

Fitting, persistence and column ordering live in `preprocessor.py`; this module
only maps a config name to a fresh estimator.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

SCALERS = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
    "normalizer": Normalizer,
    "quantile": QuantileTransformer,
    "power": PowerTransformer,
}


class NoOpScaler(BaseEstimator, TransformerMixin):
    """Passthrough, for a column that should not be scaled at all."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X


def get_scaler(scaler_name):
    """Build the scaler *scaler_name* names.

    Unknown names raise: a typo used to fall back to `NoOpScaler`, which trained
    the model on unscaled data and reported the result as if nothing were wrong.
    """
    key = scaler_name.lower()
    if key == "none":
        return NoOpScaler()
    if key not in SCALERS:
        raise ValueError(
            f"Unknown scaler {scaler_name!r}. Choose one of "
            f"{sorted([*SCALERS, 'none'])}."
        )
    return SCALERS[key]()


def create_scaler_dict(config_dict):
    """Map each configured column name to a fresh, unfitted scaler."""
    return {column: get_scaler(name) for column, name in config_dict.items()}
