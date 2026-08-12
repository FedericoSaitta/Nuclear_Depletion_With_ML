"""Entry point: train / resume / evaluate the depletion surrogates."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from loguru import logger
from omegaconf import OmegaConf

import nuclear_surrogates.datamodule.dnn_datamodule as dnn_datamodule
import nuclear_surrogates.datamodule.neural_ode_datamodule as node_datamodule
import nuclear_surrogates.models.modes as modes
from nuclear_surrogates.models.dnn_model import DNN_Model
from nuclear_surrogates.models.neural_ode import NODE_Model
from nuclear_surrogates.utils.paths import resolve_config_paths

MODELS = {
    "DNN": (DNN_Model, dnn_datamodule.DNN_Datamodule),
    "NODE": (NODE_Model, node_datamodule.NODE_Datamodule),
}
MODES = {
    "train": modes.train_and_test,
    "train_from_ckp": modes.train_from_checkpoint_and_test,
    "inference": modes.inference,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate DNN and Neural-ODE depletion surrogates"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to the run config, e.g. configs/main_config.yaml",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. train.num_epochs=5",
    )
    args = parser.parse_args(argv)

    cfg = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_dotlist(args.overrides),
    )
    # Paths in the YAML are relative to the config file, so the run is
    # independent of the directory it was launched from.
    resolve_config_paths(cfg, args.config)

    # Seed before anything constructs a model or a datamodule: this covers torch
    # weight init and the DataLoader shuffle. The run-splitting permutations take
    # an explicit generator instead, so a datamodule built outside this entry
    # point is deterministic too.
    seed = cfg.runtime.get("seed")
    if seed is None:
        seed = 42
        logger.warning("runtime.seed is absent from the config — defaulting to 42")
    L.seed_everything(seed, workers=True)

    torch.set_float32_matmul_precision("high")  # allow tensor cores

    model_cls, datamodule_cls = MODELS[cfg.runtime.model]
    MODES[cfg.runtime.mode](datamodule_cls(cfg), model_cls, cfg)


if __name__ == "__main__":
    main()
