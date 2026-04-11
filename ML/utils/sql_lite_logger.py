import sqlite3
from datetime import datetime
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from loguru import logger
import json
from omegaconf import OmegaConf

class SQLiteLogger(Logger):
  def __init__(self, db_path='results/experiments.db', name='experiment', config=None):
    super().__init__()
    self.db_path = db_path
    self._name = name
    self._experiment_id = None
    self._config = config
    self._start_time = datetime.now()
    self._init_db()
    self._create_experiment()
  
  def _init_db(self):
    """Initialize the SQLite database with a single experiments table"""
    import os
    db_dir = os.path.dirname(self.db_path)
    if db_dir:  # Only create if there's actually a directory path
        os.makedirs(db_dir, exist_ok=True)
    
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    # Single table with all info
    cursor.execute('''
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
            
            -- Input/Output features (JSON with scalers)
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
            
            -- Per-target metrics (stored as JSON)
            target_metrics TEXT,
            
            -- Metadata
            duration_seconds REAL,
            completed_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
  
  def _create_experiment(self):
    """Create a new experiment entry with hyperparameters"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    if self._config:
        try:
            # Model config
            model_name = getattr(self._config.model, 'name', None)
            layers = str(list(self._config.model.layers)) if hasattr(self._config.model, 'layers') else None
            activation = getattr(self._config.model, 'activation', None)
            output_activation = getattr(self._config.model, 'output_activation', None)
            residual_connections = getattr(self._config.model, 'residual_connections', None)
            dropout_prob = getattr(self._config.model, 'dropout_probability', None)
            
            # ============================================================
            # FIXED: Properly extract input/output features with scalers
            # ============================================================
            
            # Convert OmegaConf to Python dict to preserve key-value pairs
            if hasattr(self._config.dataset, 'inputs'):
                inputs_raw = self._config.dataset.inputs
                # Convert OmegaConf DictConfig to regular Python dict
                if hasattr(inputs_raw, 'items'):
                    inputs_dict = dict(inputs_raw)
                    input_features = json.dumps(inputs_dict)  # {"power_W_g": "MinMax", ...}
                    n_inputs = len(inputs_dict)
                else:
                    # Fallback for old format
                    input_features = json.dumps(list(inputs_raw))
                    n_inputs = len(list(inputs_raw))
            else:
                input_features = None
                n_inputs = None
            
            if hasattr(self._config.dataset, 'targets'):
                targets_raw = self._config.dataset.targets
                # Convert OmegaConf DictConfig to regular Python dict
                if hasattr(targets_raw, 'items'):
                    targets_dict = dict(targets_raw)
                    target_features = json.dumps(targets_dict)  # {"U238": "robust", ...}
                    n_outputs = len(targets_dict)
                else:
                    # Fallback for old format
                    target_features = json.dumps(list(targets_raw))
                    n_outputs = len(list(targets_raw))
            else:
                target_features = None
                n_outputs = None
            
            # Training config
            learning_rate = getattr(self._config.train, 'learning_rate', None)
            weight_decay = getattr(self._config.train, 'weight_decay', None)
            
            # Batch size is nested under dataset.train.batch_size
            if hasattr(self._config.dataset, 'train') and hasattr(self._config.dataset.train, 'batch_size'):
                batch_size = self._config.dataset.train.batch_size
            else:
                batch_size = None
            
            # It's num_epochs, not epochs
            epochs = getattr(self._config.train, 'num_epochs', None)
            loss_function = getattr(self._config.train, 'loss', None)
            lr_scheduler_patience = getattr(self._config.train, 'lr_scheduler_patience', None)
            
            # Dataset config
            dataset_path = getattr(self._config.dataset, 'path_to_data', None)
            fraction_of_data = getattr(self._config.dataset, 'fraction_of_data', 1.0)
            delta_conc = getattr(self._config.dataset, 'target_delta_conc', False)
            
            cursor.execute('''
                INSERT INTO experiments (
                    name, timestamp, model_name, layers, activation, output_activation,
                    residual_connections, dropout_prob, n_inputs, n_outputs,
                    input_features, target_features,
                    learning_rate, weight_decay, batch_size, epochs, loss_function,
                    lr_scheduler_patience, dataset_path, fraction_of_data, delta_conc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self._name,
                self._start_time.isoformat(),
                model_name,
                layers,
                activation,
                output_activation,
                residual_connections,
                dropout_prob,
                n_inputs,
                n_outputs,
                input_features,
                target_features,
                learning_rate,
                weight_decay,
                batch_size,
                epochs,
                loss_function,
                lr_scheduler_patience,
                dataset_path,
                fraction_of_data,
                delta_conc
            ))
            
        except Exception as e:
            logger.error(f"Error extracting config values: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to minimal insert
            cursor.execute('''
                INSERT INTO experiments (name, timestamp) VALUES (?, ?)
            ''', (self._name, self._start_time.isoformat()))
    else:
        logger.warning("No config provided to SQLiteLogger")
        cursor.execute('''
            INSERT INTO experiments (name, timestamp) VALUES (?, ?)
        ''', (self._name, self._start_time.isoformat()))
    
    self._experiment_id = cursor.lastrowid
    conn.commit()
    conn.close()
  
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
    """Don't log per-epoch metrics - only final results matter"""
    pass
  
  @rank_zero_only
  def log_hyperparams(self, params):
    """Already handled in _create_experiment"""
    pass
  
  @rank_zero_only
  def update_final_results(self, train_losses, val_losses, test_metrics, 
                         val_r2_scores=None, val_mae_scores=None):
    """Update the single row with final results after training/testing completes
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        test_metrics: Dict containing:
            - mae_avg, rmse_avg, r2_avg: Overall metrics
            - per_target: List of dicts with {name, mae, rmse, r2, mare_tf, mare_ar}
        val_r2_scores: List of validation R² per epoch
        val_mae_scores: List of validation MAE per epoch
    """
    import json
    
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    # Calculate duration
    duration = (datetime.now() - self._start_time).total_seconds()
    
    # Format per-target metrics as JSON
    target_metrics_json = json.dumps(test_metrics.get('per_target', []))
    
    cursor.execute('''
        UPDATE experiments SET
            final_train_loss = ?,
            final_val_loss = ?,
            final_val_r2 = ?,
            final_val_mae = ?,
            min_train_loss = ?,
            min_val_loss = ?,
            max_val_r2 = ?,
            min_val_mae = ?,
            mae_avg = ?,
            rmse_avg = ?,
            r2_avg = ?,
            target_metrics = ?,
            duration_seconds = ?,
            completed_at = ?,
            status = 'completed'
        WHERE id = ?
    ''', (
        train_losses[-1] if train_losses else None,
        val_losses[-1] if val_losses else None,
        val_r2_scores[-1] if val_r2_scores else None,
        val_mae_scores[-1] if val_mae_scores else None,
        min(train_losses) if train_losses else None,
        min(val_losses) if val_losses else None,
        max(val_r2_scores) if val_r2_scores else None,
        min(val_mae_scores) if val_mae_scores else None,
        test_metrics.get('mae_avg'),
        test_metrics.get('rmse_avg'),
        test_metrics.get('r2_avg'),
        target_metrics_json,
        duration,
        datetime.now().isoformat(),
        self._experiment_id
    ))
    
    conn.commit()
    conn.close()
    logger.info(f"✓ Updated final results for experiment {self._experiment_id}")
  
  def save(self):
    pass
  
  def finalize(self, status):
    """Mark experiment as complete/failed"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE experiments SET status = ?, completed_at = ?
        WHERE id = ?
    ''', (status, datetime.now().isoformat(), self._experiment_id))
    
    conn.commit()
    conn.close()