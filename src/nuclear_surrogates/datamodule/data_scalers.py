from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from loguru import logger


# No-operation scaler that returns data unchanged
class NoOpScaler(BaseEstimator, TransformerMixin):
    """A scaler that does nothing - returns data unchanged."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X


# Returns a scaler object based on user's input
def get_scaler(scaler_name):
    scaler_map = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "maxabs": MaxAbsScaler(),
        "normalizer": Normalizer(),
        "quantile": QuantileTransformer(),
        "power": PowerTransformer(),
        "none": NoOpScaler(),
    }

    scaler_name_lower = scaler_name.lower()
    if scaler_name_lower not in scaler_map:
        logger.error(f"Unknown scaler: {scaler_name}. Using NoOpScaler.")
        return NoOpScaler()

    return scaler_map[scaler_name_lower]


def create_scaler_dict(config_dict):
    scaler_dict = {}

    for key, value in config_dict.items():
        scaler_dict[key] = get_scaler(value)

    return scaler_dict


def print_transformer_summary(column_transformer, col_index_map):
    """Print a summary of which scaler is applied to which columns."""
    logger.info("ColumnTransformer Summary:")

    # Create reverse mapping: index -> column name
    idx_to_col = {idx: col for col, idx in col_index_map.items()}

    for name, transformer, columns in column_transformer.transformers:
        if transformer == "passthrough":
            logger.error(f"Col {name} is being passed through")
            scaler_name = "Passthrough (no scaling)"
        else:
            scaler_name = transformer.__class__.__name__

        col_names = [idx_to_col.get(col, f"index_{col}") for col in columns]
        logger.info(f"  {scaler_name}: {col_names}")


def create_column_transformer(scaler_dict, col_index_map):
    # If only one column, return the scaler directly (like old single-target code)
    if len(scaler_dict) == 1:
        col_name = list(scaler_dict.keys())[0]
        scaler = scaler_dict[col_name]
        logger.info(
            f"Using single scaler for '{col_name}': {scaler.__class__.__name__}"
        )
        return scaler

    # Multiple columns - use ColumnTransformer
    transformers = []

    # Sort columns by their index to maintain original order
    sorted_cols = sorted(col_index_map.items(), key=lambda x: x[1])

    for col_name, col_idx in sorted_cols:
        if col_name in scaler_dict:
            scaler = scaler_dict[col_name]
            transformers.append((f"{col_name}", scaler, [col_idx]))
        else:
            raise ValueError(
                f"Column '{col_name}' with index {col_idx} is not in the scaler dictionary.\n"
                f"Available scalers: {list(scaler_dict.keys())}"
            )

    # Create ColumnTransformer
    column_transformer = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0,  # Return dense array
        verbose_feature_names_out=False,
    )

    logger.info(f"Created ColumnTransformer with {len(transformers)} transformers")
    print_transformer_summary(column_transformer, col_index_map)

    return column_transformer


def inverse_transformer(column_transformer, X):
    # Handle single scaler (not a ColumnTransformer)
    if not isinstance(column_transformer, ColumnTransformer):
        return column_transformer.inverse_transform(X)

    # Handle ColumnTransformer
    X_original = X.copy()

    for name, transformer, columns in column_transformer.transformers_:
        if isinstance(columns, list):
            col_indices = columns
        else:
            col_indices = [columns]

        # Extract data for these columns
        col_data = X[:, col_indices]

        # Inverse transform
        if hasattr(transformer, "inverse_transform"):
            col_data_original = transformer.inverse_transform(col_data)
            X_original[:, col_indices] = col_data_original  # Put back in array
        else:
            raise ValueError(
                f"Could not compute inverse transform for col {columns} with {name} scaler"
            )

    return X_original
