"""Entry point: train / resume / evaluate the depletion surrogates."""

from __future__ import annotations

import argparse
from pathlib import Path

from nuclear_surrogates.utils.quiet import silence_import_noise

# Must precede the lightning import below: the warnings it suppresses are
# emitted while lightning is being imported. This is why E402 is scoped off
# for this file in pyproject.toml.
silence_import_noise()

import lightning as L
import torch
from loguru import logger
from omegaconf import OmegaConf

import nuclear_surrogates.datamodule.dnn_datamodule as dnn_datamodule
import nuclear_surrogates.datamodule.neural_ode_datamodule as node_datamodule
import nuclear_surrogates.models.modes as modes
from nuclear_surrogates.bundle import read_bundle
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
    "regenerate_plots": modes.regenerate_plots,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate DNN and Neural-ODE depletion surrogates"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        type=Path,
        help="path to the run config, e.g. configs/main_config.yaml",
    )
    source.add_argument(
        "--bundle",
        type=Path,
        help=(
            "a model-bundle directory to run inference from — it already holds "
            "the weights, the fitted scalers and the config that built them"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "the file to evaluate (with --bundle) — a new dataset for "
            "inference, or the original training dataset for --regenerate-plots"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where figures and metrics land, sets runtime.output_dir (with --bundle)",
    )
    parser.add_argument(
        "--regenerate-plots",
        action="store_true",
        help=(
            "redraw a finished run's figures from its bundle: the run's own "
            "test split, weights and scalers, no training, no new numbers"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. train.num_epochs=5",
    )
    return parser


def build_config(args):
    """Turn parsed arguments into the config the run will use."""
    if args.bundle is None:
        if args.data is not None or args.out is not None or args.regenerate_plots:
            raise SystemExit(
                "--data, --out and --regenerate-plots apply to --bundle. With "
                "--config, set runtime.mode, dataset.path_to_inference_data "
                "and runtime.output_dir in the YAML or as dotlist overrides."
            )
        cfg = OmegaConf.merge(
            OmegaConf.load(args.config),
            OmegaConf.from_dotlist(args.overrides),
        )
        # Paths in the YAML are relative to the config file, so the run is
        # independent of the directory it was launched from.
        resolve_config_paths(cfg, args.config)
        return cfg

    if args.regenerate_plots:
        # --data is the dataset the run was trained on: the split is recorded
        # as indices into it, and the test share is carved back out of it.
        cfg, _metadata = read_bundle(
            args.bundle,
            mode="regenerate_plots",
            training_data=args.data,
            output_dir=args.out,
            overrides=args.overrides,
        )
        if not cfg.dataset.get("path_to_data"):
            raise SystemExit(
                "--regenerate-plots needs the dataset the run was trained on, "
                "to carve out the same test split. Pass --data <training.h5>."
            )
        return cfg

    cfg, _metadata = read_bundle(
        args.bundle,
        inference_data=args.data,
        output_dir=args.out,
        overrides=args.overrides,
    )
    if not cfg.dataset.get("path_to_inference_data"):
        raise SystemExit(
            "--bundle needs something to evaluate. Pass --data <file>, or set "
            "dataset.path_to_inference_data=<file> as an override."
        )
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    cfg = build_config(args)

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
