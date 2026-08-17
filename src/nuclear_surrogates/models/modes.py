import json
import os

import lightning as L
import torch
import torch.multiprocessing as mp
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger

from nuclear_surrogates.bundle import BUNDLE_DIRNAME, SPLIT_NAME, write_bundle
from nuclear_surrogates.utils.paths import result_dir

# ── Shared helpers ───────────────────────────────────────────────────────────


def _build_callbacks(cfg, result_dir_path):
    model_name = cfg.model.name
    callbacks = []

    checkpoint_cb = ModelCheckpoint(
        dirpath=result_dir_path,
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


def _build_trainer(cfg, callbacks, **extra_kwargs):
    # Only meaningful with dataloader workers: shipping tensors between
    # processes exhausts the default file-descriptor sharing on Linux. With
    # num_workers=0 there are no worker processes to share with.
    if cfg.runtime.get("num_workers", 0):
        mp.set_sharing_strategy("file_system")

    return L.Trainer(
        max_epochs=cfg.train.num_epochs,
        accelerator=cfg.runtime.device,
        devices="auto",
        callbacks=callbacks,
        # A run's record is its bundle and its `test_metrics.json`, both written
        # to the result directory. There is no experiment tracker to log to.
        logger=False,
        gradient_clip_val=cfg.train.grad_clip,
        gradient_clip_algorithm="norm",
        **extra_kwargs,
    )


# ── Checkpoint utilities ─────────────────────────────────────────────────────


def load_checkpoint_into_model(model, ckpt_path):
    """Load a checkpoint into *model*.

    Strict, and it can afford to be: every checkpoint now arrives inside a
    bundle written by the same model class, so the keys match by construction —
    `func.*` for a NODE, `model.*` for a DNN. A mismatch means the config no
    longer describes the architecture the weights came from, which is exactly
    the thing that should fail loudly rather than be patched up.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    model.load_state_dict(state_dict, strict=True)
    return model


# ── Public API ───────────────────────────────────────────────────────────────


def train(datamodule, model_class, cfg):
    """Instantiate a model, train it, and test using the best checkpoint."""
    result_dir_path = result_dir(cfg)
    callbacks, checkpoint_cb = _build_callbacks(cfg, result_dir_path)

    model = model_class(config_object=cfg)
    trainer = _build_trainer(cfg, callbacks)

    trainer.fit(model=model, datamodule=datamodule)

    # Capture validation metrics before test() overwrites callback_metrics.
    val_metrics = {
        "val_r2": float(trainer.callback_metrics.get("val_r2", -float("inf"))),
        "val_loss": float(trainer.callback_metrics.get("val_loss", float("inf"))),
        "val_mae": float(trainer.callback_metrics.get("val_mae", float("inf"))),
    }

    best_path = checkpoint_cb.best_model_path
    logger.info(f"Best model saved at: {best_path}")

    _write_run_bundle(datamodule, cfg, result_dir_path, best_path, val_metrics)

    # After the bundle, not before: the test metrics do not exist yet, which is
    # why the model writes them to `test_metrics.json` itself rather than the
    # bundle carrying them.
    trainer.test(model=model, datamodule=datamodule, ckpt_path=best_path)
    return val_metrics


def _write_run_bundle(datamodule, cfg, result_dir_path, ckpt_path, metrics=None):
    """Package weights + fitted scalers + provenance beside the run's outputs.

    This is the run's record. Best-effort — a bundle failure must not throw away
    a finished training run, so it returns None rather than raising.
    """
    preprocessor = getattr(datamodule, "preprocessor", None)
    if preprocessor is None:
        logger.warning("Datamodule exposes no preprocessor — skipping bundle")
        return None
    try:
        return write_bundle(
            out_dir=os.path.join(result_dir_path, BUNDLE_DIRNAME),
            cfg=cfg,
            preprocessor=preprocessor,
            ckpt_path=ckpt_path,
            split_info=getattr(datamodule, "split_info", None),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001 - never lose a trained model to this
        logger.error(f"Failed to write model bundle: {exc}")
        return None


def finetune(datamodule, model_class, cfg, bundle):
    """Warm-start a new training run from a bundle's weights.

    Only the weights are restored. The optimizer, the LR scheduler and the
    epoch counter all start fresh, so this is a fine-tune rather than a resume
    — a run started here will not reproduce the tail of the run it came from.
    """
    result_dir_path = result_dir(cfg)
    callbacks, checkpoint_cb = _build_callbacks(cfg, result_dir_path)

    logger.info(f"Loading weights from bundle: {bundle.path}")

    model = model_class(cfg)
    datamodule.setup(stage="fit")
    model = load_checkpoint_into_model(model, bundle.weights)

    logger.info("Weights loaded — starting training with a fresh optimizer")

    trainer = _build_trainer(cfg, callbacks)
    trainer.fit(model=model, datamodule=datamodule)

    best_path = checkpoint_cb.best_model_path
    logger.info(f"Best model saved at: {best_path}")

    _write_run_bundle(datamodule, cfg, result_dir_path, best_path)

    trainer.test(model=model, datamodule=datamodule, ckpt_path=best_path)


def infer(datamodule, model_class, cfg, bundle):
    """Evaluate a frozen bundle on `dataset.path_to_inference_data`.

    The scalers come from the bundle; the training dataset is never opened.
    """
    logger.info(f"Inference from bundle: {bundle.path}")

    model = model_class(cfg)
    model = load_checkpoint_into_model(model, bundle.weights)

    datamodule.inference_mode = True

    trainer = L.Trainer(
        accelerator=cfg.runtime.device,
        devices="auto",
        logger=False,
    )

    trainer.test(model=model, datamodule=datamodule)


def _verify_split_matches_bundle(datamodule, bundle):
    """Check the reproduced split against the one the bundle recorded.

    `plots` rebuilds the split from `runtime.seed` and `dataset.split` rather
    than replaying indices, so the figures it produces are only the original
    run's if that rebuild lands on the same runs. A different dataset, a
    different `fraction_of_data`, or a change to the splitting code all move
    it, and every one of them would otherwise produce plausible-looking figures
    labelled as the published run's.

    So it is checked, not assumed. `split_indices.json` is the recorded truth.
    """
    reproduced = getattr(datamodule, "split_info", None)
    if reproduced is None:
        logger.warning(
            "The datamodule recorded no split — cannot verify it against "
            "split_indices.json. These figures may not be the original run's."
        )
        return

    split_file = os.path.join(bundle.path, SPLIT_NAME)
    if not os.path.isfile(split_file):
        logger.warning(
            f"{split_file} is missing — the bundle predates split recording, "
            f"so the reproduced split cannot be verified against it."
        )
        return

    with open(split_file) as f:
        recorded = json.load(f)

    for key in ("train", "val", "test"):
        if list(recorded.get(key, [])) != list(reproduced.get(key, [])):
            raise SystemExit(
                f"The reproduced {key} split does not match the bundle's "
                f"split_indices.json ({len(recorded.get(key, []))} runs "
                f"recorded, {len(reproduced.get(key, []))} reproduced).\n\n"
                f"These figures would not be the run the bundle describes. "
                f"Check that --data is the dataset it was trained on "
                f"(metadata.json records its sha256), and that --seed and "
                f"dataset.fraction_of_data still match config.resolved.yaml."
            )
    logger.info("Reproduced split matches the bundle's split_indices.json")


def plots(datamodule, model_class, cfg, bundle):
    """Redraw a finished run's figures from its bundle, changing no numbers.

    Rebuilds the original train/val/test split over `dataset.path_to_data`,
    loads the bundle's scalers rather than fitting fresh ones, and runs the
    evaluation epoch on the test split alone — the same runs, the same weights
    and the same scalers the published figures came from. Nothing is trained
    and no checkpoint is written.

    Unlike `infer`, this needs the training dataset: the split is recorded as
    indices into that file, and the test portion is a share of it rather than a
    separate file.
    """
    logger.info(f"Regenerating plots from bundle: {bundle.path}")

    dataset = cfg.dataset.get("path_to_data")
    if not dataset or not os.path.isfile(dataset):
        raise SystemExit(
            f"`nucml plots` needs the dataset the run was trained on, to carve "
            f"out the same test split; --data is {dataset!r}."
        )

    # stage="fit" is the split-building path — the test split it produces is
    # the run's own, not a fresh one.
    datamodule.setup(stage="fit")
    _verify_split_matches_bundle(datamodule, bundle)

    model = model_class(cfg)
    model = load_checkpoint_into_model(model, bundle.weights)

    trainer = L.Trainer(
        accelerator=cfg.runtime.device,
        devices="auto",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.test(model=model, datamodule=datamodule)
    logger.info(f"Figures written to {result_dir(cfg)}")
