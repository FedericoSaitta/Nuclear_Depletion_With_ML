"""Turn an existing checkpoint plus its config into a self-contained bundle.

Runs trained before bundles existed left only a `.ckpt`. Their fitted scalers
were thrown away with the process, so the weights alone cannot be served or
evaluated. This tool reconstructs the missing half from the config that produced
them and the dataset they were trained on:

    uv run nucml-package \
        --ckpt   results/legacy_ML/Best_Chain_Result/best-Best_Chain_Result-epoch=176.ckpt \
        --config configs/BEST_7_Isotope_DNN.yaml \
        --data   datasets/casl_3305_runs_inter.h5 \
        --out    bundles/dnn_best_chain

Run it once, on a machine that still has the datasets. The resulting bundle is a
few hundred kB and needs neither.

The caveat, which the bundle records rather than hides: these scalers are
*refitted*, not the originals. Splits used to be unseeded, so for a checkpoint
predating that fix the exact training subset — and the scalers fitted to it — is
unrecoverable. `metadata.json` carries `scalers_refit` and
`original_split_recoverable` so a backfilled bundle is never mistaken for a
faithful record: it is a valid regression anchor, not a basis for a new number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nuclear_surrogates.utils.quiet import silence_import_noise

# Must precede the lightning import below — see main.py.
silence_import_noise()

import lightning as L
from loguru import logger
from omegaconf import OmegaConf

from nuclear_surrogates.bundle import write_bundle
from nuclear_surrogates.utils.paths import resolve_config_paths


def _load_config(config_path, data_path=None, overrides=None):
    cfg = OmegaConf.merge(
        OmegaConf.load(config_path),
        OmegaConf.from_dotlist(list(overrides or [])),
    )
    resolve_config_paths(cfg, config_path)
    if data_path:
        cfg.dataset.path_to_data = str(Path(data_path).resolve())
    # Packaging is a CPU-side, single-process job regardless of what the config
    # says the training machine had.
    cfg.runtime.update(device="cpu", num_workers=0, plots=False)
    return cfg


def package(ckpt_path, config_path, out_dir, data_path=None, overrides=None):
    """Build a bundle from *ckpt_path* + *config_path*. Returns the bundle path."""
    import tempfile

    from nuclear_surrogates.main import MODELS

    cfg = _load_config(config_path, data_path, overrides)

    seed = cfg.runtime.get("seed", 42)
    L.seed_everything(seed, workers=True)

    # Building a datamodule has side effects (distribution plots). Send them to
    # a scratch directory: packaging must not write into results/.
    scratch = tempfile.TemporaryDirectory(prefix="nucml-package-")
    cfg.runtime.output_dir = scratch.name

    model_kind = cfg.runtime.get("model")
    if model_kind not in MODELS:
        raise SystemExit(
            f"config {config_path} has runtime.model={model_kind!r}; "
            f"expected one of {sorted(MODELS)}"
        )
    model_cls, datamodule_cls = MODELS[model_kind]

    dataset = cfg.dataset.get("path_to_data")
    if not dataset or not Path(dataset).is_file():
        raise SystemExit(
            f"Cannot refit scalers: dataset not found at {dataset!r}. "
            f"Pass --data pointing at the file this model was trained on."
        )

    logger.info(f"Refitting scalers from {dataset} (seed {seed})")
    datamodule = datamodule_cls(cfg)
    datamodule.setup(stage="fit")

    logger.info(f"Loading weights from {ckpt_path}")
    model = model_cls(cfg)
    try:
        from nuclear_surrogates.models.modes import load_checkpoint_into_model

        load_checkpoint_into_model(model, ckpt_path)
    except RuntimeError as exc:
        raise SystemExit(
            f"Checkpoint does not match the architecture this config builds.\n\n{exc}\n\n"
            f"The config and the checkpoint have to be the pair that were used "
            f"together — check model.layers, model.matrix_ode, "
            f"model.matrix_zero_entries and the input/target lists."
        ) from exc

    bundle = write_bundle(
        out_dir=out_dir,
        cfg=cfg,
        preprocessor=datamodule.preprocessor,
        ckpt_path=ckpt_path,
        split_info=getattr(datamodule, "split_info", None),
        extra_metadata={
            "scalers_refit": True,
            "original_split_recoverable": False,
            "provenance_note": (
                "Backfilled by nucml-package. The scalers were refitted from the "
                "dataset under the config's seed; the checkpoint's ORIGINAL split "
                "was unseeded and is unrecoverable. Valid as a regression anchor, "
                "not as a reproduction of the original run."
            ),
        },
    )
    scratch.cleanup()
    return bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Package a checkpoint + config into a self-contained model bundle",
    )
    parser.add_argument(
        "--ckpt", required=True, type=Path, help="checkpoint to package"
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="the config that produced it"
    )
    parser.add_argument("--out", required=True, type=Path, help="bundle directory")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="dataset to refit scalers from (overrides dataset.path_to_data)",
    )
    parser.add_argument(
        "overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. runtime.seed=1"
    )
    args = parser.parse_args(argv)

    if not args.ckpt.is_file():
        raise SystemExit(f"No checkpoint at {args.ckpt}")
    if not args.config.is_file():
        raise SystemExit(f"No config at {args.config}")

    out = package(
        ckpt_path=str(args.ckpt),
        config_path=str(args.config),
        out_dir=str(args.out),
        data_path=args.data,
        overrides=args.overrides,
    )
    print(f"\nBundle written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
