import os

import lightning as L
import torch
import torch.multiprocessing as mp
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger

from ML.utils.sql_lite_logger import SQLiteLogger


# ── Shared helpers ───────────────────────────────────────────────────────────

def _build_callbacks(cfg, result_dir):
  model_name = cfg.model.name
  callbacks = []

  checkpoint_cb = ModelCheckpoint(
    dirpath=result_dir,
    filename=f"best-{model_name}-{{epoch:02d}}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_last=False,
    verbose=False,
  )
  callbacks.append(checkpoint_cb)

  patience = getattr(cfg.train, "early_stopping_patience", None)
  if patience is not None:
    callbacks.append(
      EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=False,
      )
    )

  return callbacks, checkpoint_cb


def _build_trainer(cfg, callbacks, pl_logger, **extra_kwargs):
  mp.set_sharing_strategy("file_system")

  return L.Trainer(
    max_epochs=cfg.train.num_epochs,
    accelerator=cfg.runtime.device,
    devices="auto",
    callbacks=callbacks,
    logger=pl_logger,
    gradient_clip_val=cfg.train.grad_clip,
    gradient_clip_algorithm="norm",
    **extra_kwargs,
  )


def _result_dir(model_name):
  path = f"results/{model_name}/"
  os.makedirs(path, exist_ok=True)
  return path


# ── Checkpoint utilities ─────────────────────────────────────────────────────

def _fix_state_dict_keys(model_keys, ckpt_keys, state_dict):
  """Reconcile a 'model.' prefix mismatch between model and checkpoint."""
  if model_keys == ckpt_keys:
    return state_dict

  model_has_prefix = any(k.startswith("model.") for k in model_keys)
  ckpt_has_prefix = any(k.startswith("model.") for k in ckpt_keys)

  if model_has_prefix and not ckpt_has_prefix:
    return {f"model.{k}": v for k, v in state_dict.items()}
  if ckpt_has_prefix and not model_has_prefix:
    return {k.replace("model.", "", 1): v for k, v in state_dict.items()}

  # Keys differ for another reason — return as-is and let strict loading surface the error.
  return state_dict


def load_checkpoint_into_model(model, ckpt_path, save_fixed=True):
  """Load a checkpoint into *model*, fixing a leading 'model.' prefix if needed."""
  checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
  state_dict = (
    checkpoint.get("state_dict", checkpoint)
    if isinstance(checkpoint, dict)
    else checkpoint
  )

  new_state_dict = _fix_state_dict_keys(
    set(model.state_dict().keys()), set(state_dict.keys()), state_dict,
  )

  if save_fixed:
    fixed = checkpoint.copy() if isinstance(checkpoint, dict) else {}
    fixed["state_dict"] = new_state_dict
    torch.save(fixed, f"{ckpt_path}.fixed")

  model.load_state_dict(new_state_dict, strict=True)
  return model


# ── Public API ───────────────────────────────────────────────────────────────

def train_and_test(datamodule, model_class, cfg):
  """Instantiate a model, train it, and test using the best checkpoint."""
  result_dir = _result_dir(cfg.model.name)
  callbacks, checkpoint_cb = _build_callbacks(cfg, result_dir)

  pl_logger = SQLiteLogger( 
    db_path=cfg.runtime.model_database, name=cfg.model.name, config=cfg,
  )

  model = model_class(config_object=cfg)
  trainer = _build_trainer(cfg, callbacks, pl_logger)

  trainer.fit(model=model, datamodule=datamodule)

  # Capture validation metrics before test() overwrites callback_metrics.
  val_metrics = {
    "val_r2": float(trainer.callback_metrics.get("val_r2", -float("inf"))),
    "val_loss": float(trainer.callback_metrics.get("val_loss", float("inf"))),
    "val_mae": float(trainer.callback_metrics.get("val_mae", float("inf"))),
  }

  best_path = checkpoint_cb.best_model_path
  logger.info(f"Best model saved at: {best_path}")

  trainer.test(model=model, datamodule=datamodule, ckpt_path=best_path)
  return val_metrics


def train_from_checkpoint_and_test(datamodule, model_class, cfg):
  """Resume training from a checkpoint, then test using the best checkpoint."""
  result_dir = _result_dir(cfg.model.name)
  callbacks, checkpoint_cb = _build_callbacks(cfg, result_dir)

  pl_logger = SQLiteLogger( 
    db_path=cfg.runtime.model_database, name=cfg.model.name, config=cfg,
  )

  logger.info(f"Loading model from checkpoint: {cfg.runtime.ckp_path}")

  model = model_class(cfg)
  datamodule.setup(stage="fit")
  model = load_checkpoint_into_model(model, cfg.runtime.ckp_path)

  logger.info("Successfully loaded checkpoint — starting training")

  trainer = _build_trainer(cfg, callbacks, pl_logger)
  trainer.fit(model=model, datamodule=datamodule)

  best_path = checkpoint_cb.best_model_path
  logger.info(f"Best model saved at: {best_path}")

  trainer.test(model=model, datamodule=datamodule, ckpt_path=best_path)


def inference(datamodule, model_class, cfg):
  """Load a checkpoint, fit scalers on training data, test on inference data."""
  logger.info(f"Inference mode — loading checkpoint: {cfg.runtime.ckp_path}")

  model = model_class(cfg)
  model = load_checkpoint_into_model(model, cfg.runtime.ckp_path, save_fixed=False)

  datamodule.inference_mode = True

  trainer = L.Trainer(
    accelerator=cfg.runtime.device,
    devices="auto",
    logger=False,
  )

  trainer.test(model=model, datamodule=datamodule)
