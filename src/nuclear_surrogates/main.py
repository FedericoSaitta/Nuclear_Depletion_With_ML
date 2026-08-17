"""Entry point: train / finetune / evaluate the depletion surrogates.

Four verbs, and the verb is the whole decision:

    nucml train    --config configs/node.yaml --data datasets/runs.h5
    nucml finetune --bundle results/<name>/model-bundle --data datasets/runs.h5
    nucml infer    --bundle results/<name>/model-bundle --data datasets/new.h5
    nucml plots    --bundle results/<name>/model-bundle --data datasets/runs.h5

The split of responsibility is deliberate: **the YAML describes the model, the
command line describes the run.** A config carries `dataset` / `model` / `train`
and nothing else — no paths, no device, no mode — so the same file trains on a
laptop and on the cluster. Everything that varies between those two is a flag
with a default, and the `runtime` block the rest of the code reads is
synthesised here from those flags rather than loaded from disk.

Three of the four verbs take a bundle rather than a config, because a bundle is
the only thing that carries a model's weights *and* the scalers they were
fitted with. There is no checkpoint path anywhere in this interface.
"""

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
from omegaconf import OmegaConf

import nuclear_surrogates.datamodule.dnn_datamodule as dnn_datamodule
import nuclear_surrogates.datamodule.neural_ode_datamodule as node_datamodule
import nuclear_surrogates.models.modes as modes
from nuclear_surrogates.bundle import read_bundle
from nuclear_surrogates.models.dnn_model import DNN_Model
from nuclear_surrogates.models.neural_ode import NODE_Model

MODELS = {
    "DNN": (DNN_Model, dnn_datamodule.DNN_Datamodule),
    "NODE": (NODE_Model, node_datamodule.NODE_Datamodule),
}

DEFAULT_SEED = 42

# `auto` lets one config serve both the laptop and the cluster, which is what
# makes a machine-independent config file possible at all.
DEVICES = ("auto", "cpu", "cuda", "gpu", "mps")


def _machine_flags() -> argparse.ArgumentParser:
    """The flags that describe the machine, shared by every verb.

    These are exactly the keys that used to have to be copied between config
    files without anyone meaning to change them.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--out",
        type=Path,
        default=Path("results"),
        help="where figures, checkpoints and the bundle land (default: results)",
    )
    parent.add_argument(
        "--device",
        default="auto",
        choices=DEVICES,
        help="accelerator to run on (default: auto)",
    )
    parent.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "dataloader worker processes (default: 0). For the NODE, 0 is "
            "usually faster — the dataset is already tensors in RAM"
        ),
    )
    parent.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            f"seeds weights, shuffling and the split (default: {DEFAULT_SEED}, "
            f"or the seed the bundle recorded)"
        ),
    )
    parent.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="skip the data-distribution figures",
    )
    parent.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. train.num_epochs=5",
    )
    return parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nucml",
        description="Train and evaluate DNN and Neural-ODE depletion surrogates",
    )
    sub = parser.add_subparsers(
        dest="command", required=True, metavar="{train,finetune,infer,plots}"
    )

    train = sub.add_parser(
        "train",
        parents=[_machine_flags()],
        help="train a model from scratch",
        description=(
            "Fit a model described by a config on a dataset. Writes a model "
            "bundle beside the run's outputs — that bundle is what the other "
            "three verbs take."
        ),
    )
    train.add_argument(
        "--config",
        type=Path,
        required=True,
        help="the model config, e.g. configs/node.yaml",
    )
    train.add_argument(
        "--data",
        type=Path,
        required=True,
        help="training HDF5 — the scalers are fitted on its training split",
    )
    train.set_defaults(func=modes.train)

    finetune = sub.add_parser(
        "finetune",
        parents=[_machine_flags()],
        help="continue training from a bundle's weights",
        description=(
            "Warm-start a new training run from a bundle's weights. Only the "
            "weights are restored — the optimizer, the LR scheduler and the "
            "epoch counter all start fresh, so this is a fine-tune rather than "
            "a resume."
        ),
    )
    finetune.add_argument(
        "--bundle", type=Path, required=True, help="the model bundle to start from"
    )
    finetune.add_argument(
        "--data", type=Path, required=True, help="training HDF5 to continue on"
    )
    finetune.set_defaults(func=modes.finetune)

    infer = sub.add_parser(
        "infer",
        parents=[_machine_flags()],
        help="evaluate a bundle on new data",
        description=(
            "Evaluate a frozen model on a dataset it has never seen. The "
            "scalers come from the bundle and the training file is never "
            "opened, so a bundle copied off the cluster runs unmodified."
        ),
    )
    infer.add_argument(
        "--bundle", type=Path, required=True, help="the model bundle to evaluate"
    )
    infer.add_argument(
        "--data", type=Path, required=True, help="the HDF5 to evaluate it on"
    )
    infer.set_defaults(func=modes.infer)

    plots = sub.add_parser(
        "plots",
        parents=[_machine_flags()],
        help="redraw a finished run's figures",
        description=(
            "Replay a finished run over its own recorded test split: the same "
            "runs, weights and scalers the published figures came from. "
            "Nothing is trained and no checkpoint is written. Unlike `infer` "
            "this needs the dataset the run was TRAINED on, because the split "
            "is recorded as indices into that file."
        ),
    )
    plots.add_argument(
        "--bundle", type=Path, required=True, help="the model bundle to replay"
    )
    plots.add_argument(
        "--data",
        type=Path,
        required=True,
        help="the dataset the run was trained on, to carve out the same split",
    )
    plots.set_defaults(func=modes.plots)

    return parser


def _runtime_section(args, bundle):
    """Build the `runtime` block from the command line.

    Nothing here comes from the YAML. A bundle contributes exactly one value —
    the seed — because `plots` has to rebuild the split the recorded run used,
    and the split is a function of the seed.
    """
    seed = args.seed
    if seed is None:
        seed = DEFAULT_SEED if bundle is None else bundle.seed

    return {
        "device": args.device,
        "num_workers": args.workers,
        "seed": seed,
        "output_dir": str(args.out.resolve()),
        "plots": args.plots,
    }


def build_config(args):
    """Turn parsed arguments into `(cfg, bundle)`. *bundle* is None for `train`."""
    if args.command == "train":
        cfg, bundle = OmegaConf.load(args.config), None
    else:
        cfg, bundle = read_bundle(args.bundle)

    # `infer` is the one verb that must never open the training file; every
    # other verb's --data IS the training file. `read_bundle` has already
    # cleared `path_to_data`, so an infer run has nothing to fall back to.
    if args.command == "infer":
        cfg.dataset.path_to_inference_data = str(args.data.resolve())
    else:
        cfg.dataset.path_to_data = str(args.data.resolve())

    cfg.runtime = _runtime_section(args, bundle)

    # Last, so a dotlist override beats everything above — including,
    # deliberately, the paths and the runtime block just set.
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    return cfg, bundle


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    cfg, bundle = build_config(args)

    # Seed before anything constructs a model or a datamodule: this covers torch
    # weight init and the DataLoader shuffle. The run-splitting permutations take
    # an explicit generator instead, so a datamodule built outside this entry
    # point is deterministic too.
    L.seed_everything(cfg.runtime.seed, workers=True)

    torch.set_float32_matmul_precision("high")  # allow tensor cores

    model_cls, datamodule_cls = MODELS[cfg.model.kind]
    if bundle is None:
        args.func(datamodule_cls(cfg), model_cls, cfg)
    else:
        args.func(datamodule_cls(cfg), model_cls, cfg, bundle)


if __name__ == "__main__":
    main()
