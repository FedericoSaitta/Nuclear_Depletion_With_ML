"""One SQLite row per experiment: an index of runs and their results.

The row is deliberately *not* a copy of the config. Everything describing how a
run was set up — resolved config, fitted scalers, split indices, git SHA,
dataset hash, library versions — lives in that run's bundle, and `bundle_path`
points at it. Duplicating a subset of those fields into columns is how the two
drift apart, and the schema only ever knew about the DNN's hyperparameters
anyway: it silently recorded nothing for `matrix_ode`, the matrix constraints,
the solver or its tolerances.

To read a run's configuration:
`results/<name>/model-bundle/config.resolved.yaml`.
"""

import json
import os
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime

from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger

# Column -> SQL type. Also drives the migration of databases created by an
# earlier schema, so a new column can be added without a manual ALTER.
COLUMNS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "name": "TEXT NOT NULL",
    "timestamp": "TEXT NOT NULL",
    "status": "TEXT DEFAULT 'running'",
    # The run's full record. Everything not stored here is stored there.
    "bundle_path": "TEXT",
    # Training curves, reduced to their endpoints and extrema.
    "final_train_loss": "REAL",
    "final_val_loss": "REAL",
    "final_val_r2": "REAL",
    "final_val_mae": "REAL",
    "min_train_loss": "REAL",
    "min_val_loss": "REAL",
    "max_val_r2": "REAL",
    "min_val_mae": "REAL",
    # Test metrics, averaged over targets and then per target as JSON.
    "mae_avg": "REAL",
    "rmse_avg": "REAL",
    "r2_avg": "REAL",
    "target_metrics": "TEXT",
    "duration_seconds": "REAL",
    "completed_at": "TEXT",
}

SCHEMA = "CREATE TABLE IF NOT EXISTS experiments (\n  {}\n)".format(
    ",\n  ".join(f"{column} {sql_type}" for column, sql_type in COLUMNS.items())
)


class SQLiteLogger(Logger):
    def __init__(self, db_path="results/experiments.db", name="experiment"):
        super().__init__()
        self.db_path = db_path
        self._name = name
        self._experiment_id = None
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
            self._migrate(conn)

    @staticmethod
    def _migrate(conn):
        """Add any column an older database is missing.

        SQLite cannot ALTER in a primary key or a NOT NULL column without a
        default, but neither is ever new — those columns exist in every schema
        this project has had. Columns dropped since are left in place: they
        still hold the values old runs recorded.
        """
        existing = {row[1] for row in conn.execute("PRAGMA table_info(experiments)")}
        for column, sql_type in COLUMNS.items():
            if (
                column in existing
                or "PRIMARY KEY" in sql_type
                or "NOT NULL" in sql_type
            ):
                continue
            conn.execute(f"ALTER TABLE experiments ADD COLUMN {column} {sql_type}")
            logger.info(f"Added column {column!r} to the experiments table")

    def _create_experiment(self):
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO experiments (name, timestamp) VALUES (?, ?)",
                (self._name, self._start_time.isoformat()),
            )
            self._experiment_id = cursor.lastrowid

    def _update(self, **values):
        assignments = ", ".join(f"{column} = ?" for column in values)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE experiments SET {assignments} WHERE id = ?",
                (*values.values(), self._experiment_id),
            )

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
        """Hyperparameters live in the bundle's resolved config, not here."""

    @rank_zero_only
    def set_bundle_path(self, bundle_path):
        """Point this row at the bundle holding the run's full record."""
        if bundle_path is None:
            logger.warning(
                f"Experiment {self._experiment_id} has no bundle — its config and "
                f"provenance were not recorded anywhere."
            )
            return
        self._update(bundle_path=str(bundle_path))

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
        self._update(
            final_train_loss=train_losses[-1] if train_losses else None,
            final_val_loss=val_losses[-1] if val_losses else None,
            final_val_r2=val_r2_scores[-1] if val_r2_scores else None,
            final_val_mae=val_mae_scores[-1] if val_mae_scores else None,
            min_train_loss=min(train_losses) if train_losses else None,
            min_val_loss=min(val_losses) if val_losses else None,
            max_val_r2=max(val_r2_scores) if val_r2_scores else None,
            min_val_mae=min(val_mae_scores) if val_mae_scores else None,
            mae_avg=test_metrics.get("mae_avg"),
            rmse_avg=test_metrics.get("rmse_avg"),
            r2_avg=test_metrics.get("r2_avg"),
            target_metrics=json.dumps(test_metrics.get("per_target", [])),
            duration_seconds=(datetime.now() - self._start_time).total_seconds(),
            completed_at=datetime.now().isoformat(),
            status="completed",
        )
        logger.info(f"Updated final results for experiment {self._experiment_id}")

    def save(self):
        pass

    def finalize(self, status):
        """Record how the run ended.

        Lightning calls this with "success" after `update_final_results` has
        already written "completed", which used to leave the column holding two
        synonyms depending on which path wrote last. A finished run keeps
        "completed"; anything else is recorded verbatim.
        """
        if status == "success":
            status = "completed"
        self._update(status=status, completed_at=datetime.now().isoformat())
