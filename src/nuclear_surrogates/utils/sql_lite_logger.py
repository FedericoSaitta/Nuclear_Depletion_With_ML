"""One SQLite row per experiment: config in when it starts, results in when it ends."""

import json
import os
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime

from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    status TEXT DEFAULT 'running',

    -- Model architecture
    model_name TEXT,
    layers TEXT,
    activation TEXT,
    output_activation TEXT,
    residual_connections BOOLEAN,
    dropout_prob REAL,
    n_inputs INTEGER,
    n_outputs INTEGER,

    -- Input/output features, as JSON mapping column -> scaler
    input_features TEXT,
    target_features TEXT,

    -- Training config
    learning_rate REAL,
    weight_decay REAL,
    batch_size INTEGER,
    epochs INTEGER,
    loss_function TEXT,
    lr_scheduler_patience INTEGER,

    -- Dataset config
    dataset_path TEXT,
    fraction_of_data REAL,
    delta_conc BOOLEAN,

    -- Final training metrics
    final_train_loss REAL,
    final_val_loss REAL,
    final_val_r2 REAL,
    final_val_mae REAL,
    min_train_loss REAL,
    min_val_loss REAL,
    max_val_r2 REAL,
    min_val_mae REAL,

    -- Overall test metrics
    mae_avg REAL,
    rmse_avg REAL,
    r2_avg REAL,

    -- Per-target metrics, as JSON
    target_metrics TEXT,

    -- Metadata
    duration_seconds REAL,
    completed_at TEXT
)
"""

# Database column -> dotted config path. Anything absent from the config is
# stored as NULL rather than failing the run.
CONFIG_COLUMNS = {
    "model_name": "model.name",
    "activation": "model.activation",
    "output_activation": "model.output_activation",
    "residual_connections": "model.residual_connections",
    "dropout_prob": "model.dropout_probability",
    "learning_rate": "train.learning_rate",
    "weight_decay": "train.weight_decay",
    "batch_size": "dataset.train.batch_size",
    "epochs": "train.num_epochs",
    "loss_function": "train.loss",
    "lr_scheduler_patience": "train.lr_scheduler_patience",
    "dataset_path": "dataset.path_to_data",
    "fraction_of_data": "dataset.fraction_of_data",
    "delta_conc": "dataset.target_delta_conc",
}


def _dig(config, dotted_path):
    """Follow a dotted config path, returning None if any step is missing."""
    node = config
    for key in dotted_path.split("."):
        if node is None or not hasattr(node, key):
            return None
        node = getattr(node, key)
    return node


def _feature_columns(config, side):
    """(JSON of column -> scaler, count) for `dataset.inputs` or `dataset.targets`."""
    features = _dig(config, f"dataset.{side}")
    if features is None:
        return None, None
    # A mapping carries its scalers; a bare list is the older, scaler-less form.
    as_dict = dict(features) if hasattr(features, "items") else list(features)
    return json.dumps(as_dict), len(as_dict)


class SQLiteLogger(Logger):
    def __init__(
        self, db_path="results/experiments.db", name="experiment", config=None
    ):
        super().__init__()
        self.db_path = db_path
        self._name = name
        self._experiment_id = None
        self._config = config
        self._start_time = datetime.now()
        self._init_db()
        self._create_experiment()

    @contextmanager
    def _connect(self):
        """Open, commit and close — a leaked handle keeps the database locked."""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            yield conn

    def _init_db(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        with self._connect() as conn:
            conn.execute(SCHEMA)

    def _row_from_config(self):
        """Flatten the config into the experiment table's columns."""
        config = self._config
        if config is None:
            logger.warning("No config provided to SQLiteLogger")
            return {}

        row = {column: _dig(config, path) for column, path in CONFIG_COLUMNS.items()}

        layers = _dig(config, "model.layers")
        row["layers"] = None if layers is None else str(list(layers))

        row["input_features"], row["n_inputs"] = _feature_columns(config, "inputs")
        row["target_features"], row["n_outputs"] = _feature_columns(config, "targets")

        if row["fraction_of_data"] is None:
            row["fraction_of_data"] = 1.0
        if row["delta_conc"] is None:
            row["delta_conc"] = False
        return row

    def _create_experiment(self):
        try:
            row = self._row_from_config()
        except Exception as exc:  # noqa: BLE001 - a logging gap must not kill a run
            logger.error(f"Error extracting config values: {exc}")
            row = {}

        row["name"] = self._name
        row["timestamp"] = self._start_time.isoformat()

        columns = ", ".join(row)
        placeholders = ", ".join("?" * len(row))
        with self._connect() as conn:
            cursor = conn.execute(
                f"INSERT INTO experiments ({columns}) VALUES ({placeholders})",
                tuple(row.values()),
            )
            self._experiment_id = cursor.lastrowid

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._experiment_id

    @property
    def experiment_id(self):
        return self._experiment_id

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        """Per-epoch metrics are not stored — only the final row matters."""

    @rank_zero_only
    def log_hyperparams(self, params):
        """Already handled in `_create_experiment`."""

    @rank_zero_only
    def update_final_results(
        self,
        train_losses,
        val_losses,
        test_metrics,
        val_r2_scores=None,
        val_mae_scores=None,
    ):
        """Fill in the run's results.

        *test_metrics* holds `mae_avg`, `rmse_avg`, `r2_avg` and a `per_target`
        list of {name, mae, rmse, r2, mare_tf, mare_ar}.
        """
        updates = {
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "final_val_r2": val_r2_scores[-1] if val_r2_scores else None,
            "final_val_mae": val_mae_scores[-1] if val_mae_scores else None,
            "min_train_loss": min(train_losses) if train_losses else None,
            "min_val_loss": min(val_losses) if val_losses else None,
            "max_val_r2": max(val_r2_scores) if val_r2_scores else None,
            "min_val_mae": min(val_mae_scores) if val_mae_scores else None,
            "mae_avg": test_metrics.get("mae_avg"),
            "rmse_avg": test_metrics.get("rmse_avg"),
            "r2_avg": test_metrics.get("r2_avg"),
            "target_metrics": json.dumps(test_metrics.get("per_target", [])),
            "duration_seconds": (datetime.now() - self._start_time).total_seconds(),
            "completed_at": datetime.now().isoformat(),
            "status": "completed",
        }

        assignments = ", ".join(f"{column} = ?" for column in updates)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE experiments SET {assignments} WHERE id = ?",
                (*updates.values(), self._experiment_id),
            )
        logger.info(f"Updated final results for experiment {self._experiment_id}")

    def save(self):
        pass

    def finalize(self, status):
        with self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET status = ?, completed_at = ? WHERE id = ?",
                (status, datetime.now().isoformat(), self._experiment_id),
            )
