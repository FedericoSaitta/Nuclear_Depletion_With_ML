"""Entry point: train / resume / evaluate the depletion surrogates."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

import ML.datamodule.dnn_datamodule as dnn_datamodule
import ML.datamodule.neural_ode_datamodule as node_datamodule
import ML.models.modes as modes
from ML.models.dnn_model import DNN_Model
from ML.models.neural_ode import NODE_Model

# main_config.yaml sits next to this file, so it is found whether you run
# `nucml`, `python -m ML.main`, or `python ML/main.py` — from any directory.
DEFAULT_CONFIG = Path(__file__).with_name("main_config.yaml")

MODELS = {
    "DNN": (DNN_Model, dnn_datamodule.DNN_Datamodule),
    "NODE": (NODE_Model, node_datamodule.NODE_Datamodule),
}
MODES = {
    "train": modes.train_and_test,
    "train_from_ckp": modes.train_from_checkpoint_and_test,
    "inference": modes.inference,
}

# Dataset paths in the YAML are written relative to the config file, not to CWD.
_PATH_KEYS = ("path_to_data", "path_to_inference_data")


def _resolve_dataset_paths(cfg, config_path: Path) -> None:
    """Rewrite dataset paths to absolute, anchored at the config's directory."""
    for key in _PATH_KEYS:
        value = cfg.dataset.get(key)
        if value:
            cfg.dataset[key] = str((config_path.parent / value).resolve())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate DNN and Neural-ODE depletion surrogates"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"path to the run config (default: {DEFAULT_CONFIG})")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf dotlist overrides, e.g. train.num_epochs=5")
    args = parser.parse_args(argv)

    cfg = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_dotlist(args.overrides),
    )
    _resolve_dataset_paths(cfg, args.config.resolve())

    torch.set_float32_matmul_precision("high")   # allow tensor cores

    model_cls, datamodule_cls = MODELS[cfg.runtime.model]
    MODES[cfg.runtime.mode](datamodule_cls(cfg), model_cls, cfg)


if __name__ == "__main__":
    main()
