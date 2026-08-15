"""A model bundle: everything needed to reuse a trained model, in one directory.

A training run writes it beside its own outputs, as
`results/<model_name>/model-bundle/`; `nucml-package` writes one wherever
`--out` points.

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
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

BUNDLE_VERSION = 1

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

    `train_and_test` defaults a metric the run never logged to +/-inf, and
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
        "bundle_version": BUNDLE_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "model_name": cfg.model.get("name"),
        "model_kind": cfg.runtime.get("model"),
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


def read_bundle(
    bundle_dir,
    mode="inference",
    inference_data=None,
    training_data=None,
    output_dir=None,
    overrides=None,
):
    """Rebuild the config a bundle describes. Returns (cfg, metadata).

    A bundle is self-anchoring: the config it carries is rewritten to point at
    the bundle's *own* weights and scalers, so a bundle copied off the cluster
    stops referring to the cluster's filesystem. `dataset.path_to_data` is
    cleared for the same reason — the training file is not part of a bundle,
    and `inference` must never reopen it.

    *training_data* puts it back, for `regenerate_plots`, which replays a
    finished run over its own recorded test split and therefore does need the
    file that run was trained on.

    The result is a plain config, so everything downstream (`MODES`, the
    datamodules, dotlist overrides) works exactly as it does for `--config`.
    """
    bundle = Path(bundle_dir).resolve()
    if not bundle.is_dir():
        raise SystemExit(f"No bundle directory at {bundle}")

    missing = [name for name in REQUIRED_NAMES if not (bundle / name).is_file()]
    if missing:
        raise SystemExit(
            f"{bundle} is not a model bundle — missing {', '.join(missing)}.\n"
            f"A bundle is written by a training run to "
            f"results/<model_name>/{BUNDLE_DIRNAME}/, or by nucml-package."
        )

    metadata = {}
    metadata_path = bundle / METADATA_NAME
    if metadata_path.is_file():
        with open(metadata_path) as f:
            metadata = json.load(f)

    version = metadata.get("bundle_version", BUNDLE_VERSION)
    if version > BUNDLE_VERSION:
        raise SystemExit(
            f"{bundle} is bundle_version {version}; this build reads up to "
            f"{BUNDLE_VERSION}. Upgrade nuclear_surrogates to read it."
        )

    cfg = OmegaConf.load(bundle / CONFIG_NAME)

    # No resolve_config_paths here: config.resolved.yaml already holds absolute
    # paths, and re-anchoring them at the bundle directory would corrupt them.
    # Everything set below is absolute already.
    cfg.runtime.mode = mode
    cfg.runtime.ckp_path = str(bundle / WEIGHTS_NAME)
    # Recorded so a mode can find the rest of the bundle — regenerate_plots
    # reads split_indices.json back out of it.
    cfg.runtime.bundle_path = str(bundle)
    cfg.dataset.preprocessor_path = str(bundle / PREPROCESSOR_NAME)
    cfg.dataset.path_to_data = ""

    if inference_data is not None:
        cfg.dataset.path_to_inference_data = str(Path(inference_data).resolve())
    if training_data is not None:
        cfg.dataset.path_to_data = str(Path(training_data).resolve())
    if output_dir is not None:
        cfg.runtime.output_dir = str(Path(output_dir).resolve())

    # Last, so the user can override anything above — including, deliberately,
    # the paths this function just set.
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    logger.info(
        f"Loaded bundle {bundle} "
        f"(model {metadata.get('model_name')!r}, kind {metadata.get('model_kind')!r}, "
        f"git {str(metadata.get('git_sha'))[:8]})"
    )
    return cfg, metadata
