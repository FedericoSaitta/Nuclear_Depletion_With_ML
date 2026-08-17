"""A model bundle: everything needed to reuse a trained model, in one directory.

A training run writes one beside its own outputs, as
`<--out>/<model_name>/model-bundle/`. It is the only thing `nucml finetune`,
`nucml infer` and `nucml plots` accept — there is deliberately no way to point
them at a bare checkpoint.

    model-bundle/
    ├── weights.ckpt          Lightning checkpoint
    ├── preprocessor.json     fitted scalers, PLAIN TEXT (source of truth)
    ├── preprocessor.joblib   convenience copy
    ├── config.resolved.yaml  the fully-merged config, post-override
    ├── split_indices.json    which runs were train / val / test
    └── metadata.json         git sha, seed, dataset sha256, library versions

`metadata.json` is what lets a number be traced back to the weights that
produced it six months later.

`write_bundle` and `read_bundle` are the two halves. Weights alone cannot be
served: both models build their architecture from the config, `load_state_dict`
is strict, and for a NODE the solver and its tolerances are part of the model's
definition rather than of its training. `config.resolved.yaml` is what closes
that gap, which is why the bundle carries it and `read_bundle` reads it back.

`config.resolved.yaml` records the run's `runtime` block — device, workers,
seed, output directory — even though a hand-written config in `configs/` has no
such section. That is the point of the word *resolved*: it is the machine's
record of what actually ran, not a file anyone edits. `read_bundle` strips it
back off, because the machine that reads a bundle is rarely the one that wrote
it; only the seed survives, and only so `plots` can rebuild the same split.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

# Where a training run puts its bundle, relative to that run's result dir.
BUNDLE_DIRNAME = "model-bundle"

# Named once and used by both halves, so a rename cannot make the reader and the
# writer disagree about what a bundle is.
WEIGHTS_NAME = "weights.ckpt"
PREPROCESSOR_NAME = "preprocessor.json"
CONFIG_NAME = "config.resolved.yaml"
SPLIT_NAME = "split_indices.json"
METADATA_NAME = "metadata.json"

# Without all three a directory is not a bundle: no weights means no model, no
# preprocessor means the wrong scalers, no config means no architecture to load
# the weights into.
REQUIRED_NAMES = (WEIGHTS_NAME, PREPROCESSOR_NAME, CONFIG_NAME)


@dataclass(frozen=True)
class Bundle:
    """A bundle's contents, addressed by name rather than by config key.

    This is what replaced `runtime.ckp_path`. A checkpoint path used to travel
    inside the config, which meant every mode had to trust that some earlier
    step had put the right one there — and that a user could set it to a
    checkpoint whose scalers lived somewhere else entirely. Passing the bundle
    itself makes the weights and the scalers arrive together or not at all.
    """

    path: Path
    weights: Path
    preprocessor: Path
    metadata: dict
    seed: int | None


def _git(*args) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _library_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in ("torch", "lightning", "sklearn", "numpy", "torchdiffeq"):
        try:
            versions[module] = __import__(module).__version__
        except (ImportError, AttributeError):
            versions[module] = None
    return versions


def _finite(metrics: dict) -> dict:
    """Replace non-finite metric values with None.

    `modes.train` defaults a metric the run never logged to +/-inf, and
    `json.dump` writes that as a bare `Infinity` — which Python reads back but
    `jq`, JS and Rust all reject. A metric that was never recorded is honestly
    null, so write it that way.
    """
    return {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in metrics.items()
    }


def sha256_file(path, chunk_size=8 << 20) -> str | None:
    """Streamed SHA-256 — the training files are hundreds of MB."""
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bundle(
    out_dir,
    cfg,
    preprocessor,
    ckpt_path=None,
    split_info=None,
    metrics=None,
    extra_metadata=None,
    hash_dataset=True,
):
    """Assemble a bundle directory. Returns its path.

    *hash_dataset* can be turned off in tests, where hashing the source file is
    pure overhead.
    """
    os.makedirs(out_dir, exist_ok=True)

    preprocessor.save(out_dir)
    OmegaConf.save(cfg, os.path.join(out_dir, CONFIG_NAME))

    if ckpt_path and os.path.isfile(ckpt_path):
        shutil.copy2(ckpt_path, os.path.join(out_dir, WEIGHTS_NAME))
    else:
        logger.warning(f"No checkpoint at {ckpt_path!r} — bundle has no weights")

    if split_info is not None:
        with open(os.path.join(out_dir, SPLIT_NAME), "w") as f:
            json.dump({"seed": cfg.runtime.get("seed"), **split_info}, f, indent=2)

    dirty = _git("status", "--porcelain")
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "model_name": cfg.model.get("name"),
        "model_kind": cfg.model.get("kind"),
        "seed": cfg.runtime.get("seed"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "source_checkpoint": ckpt_path,
        "dataset": {
            "path": cfg.dataset.get("path_to_data"),
            "sha256": (
                sha256_file(cfg.dataset.get("path_to_data")) if hash_dataset else None
            ),
            "fraction_of_data": cfg.dataset.get("fraction_of_data"),
        },
        "libraries": _library_versions(),
    }
    if metrics is not None:
        metadata["metrics"] = _finite(metrics)
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(os.path.join(out_dir, METADATA_NAME), "w") as f:
        # allow_nan=False so a non-finite value that slipped past _finite fails
        # here rather than producing a file only Python can read.
        json.dump(metadata, f, indent=2, allow_nan=False)

    logger.info(f"Wrote model bundle to {out_dir}")
    return out_dir


def read_bundle(bundle_dir):
    """Load a bundle. Returns `(cfg, Bundle)`.

    The config comes back describing the *model* only: the `runtime` block the
    training machine recorded is dropped, and `dataset.path_to_data` is cleared
    so `infer` has no way to reopen the training file even by accident. The
    caller supplies a fresh runtime from the command line, and the file to work
    on as `--data`.

    `dataset.preprocessor_path` is the one path this does set, because the
    datamodules read it directly and the whole point of a bundle is that the
    scalers arrive with the weights.
    """
    bundle = Path(bundle_dir).resolve()
    if not bundle.is_dir():
        raise SystemExit(f"No bundle directory at {bundle}")

    missing = [name for name in REQUIRED_NAMES if not (bundle / name).is_file()]
    if missing:
        raise SystemExit(
            f"{bundle} is not a model bundle — missing {', '.join(missing)}.\n"
            f"A bundle is written by a training run to "
            f"<--out>/<model_name>/{BUNDLE_DIRNAME}/."
        )

    metadata = {}
    metadata_path = bundle / METADATA_NAME
    if metadata_path.is_file():
        with open(metadata_path) as f:
            metadata = json.load(f)

    cfg = OmegaConf.load(bundle / CONFIG_NAME)

    # The seed is the one runtime value worth carrying forward: `plots` rebuilds
    # the recorded run's split from it, and a different seed silently produces
    # different figures under the published run's name.
    seed = cfg.runtime.get("seed") if "runtime" in cfg else None
    if seed is None:
        seed = metadata.get("seed")
    cfg.pop("runtime", None)

    cfg.dataset.preprocessor_path = str(bundle / PREPROCESSOR_NAME)
    cfg.dataset.path_to_data = ""

    logger.info(
        f"Loaded bundle {bundle} "
        f"(model {metadata.get('model_name')!r}, kind {metadata.get('model_kind')!r}, "
        f"git {str(metadata.get('git_sha'))[:8]})"
    )
    return cfg, Bundle(
        path=bundle,
        weights=bundle / WEIGHTS_NAME,
        preprocessor=bundle / PREPROCESSOR_NAME,
        metadata=metadata,
        seed=seed,
    )
