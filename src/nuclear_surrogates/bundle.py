"""A model bundle: everything needed to reuse a trained model, in one directory.

A `.ckpt` is half a model — the other half is the fitted scalers and the config
that shaped them. Shipping only weights is what forced every reload to re-derive
the scalers from a 542 MB HDF5, and what made the golden regression tests
un-runnable anywhere but the author's machine.

    bundles/<run_id>/
    ├── weights.ckpt          Lightning checkpoint
    ├── preprocessor.json     fitted scalers, PLAIN TEXT (source of truth)
    ├── preprocessor.joblib   convenience copy
    ├── config.resolved.yaml  the fully-merged config, post-override
    ├── split_indices.json    which runs were train / val / test
    └── metadata.json         git sha, seed, dataset sha256, library versions

`metadata.json` is what lets a number be traced back to the weights that
produced it six months later (AUDIT.md §6.1).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from loguru import logger
from omegaconf import OmegaConf

BUNDLE_VERSION = 1


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
    OmegaConf.save(cfg, os.path.join(out_dir, "config.resolved.yaml"))

    if ckpt_path and os.path.isfile(ckpt_path):
        shutil.copy2(ckpt_path, os.path.join(out_dir, "weights.ckpt"))
    else:
        logger.warning(f"No checkpoint at {ckpt_path!r} — bundle has no weights")

    if split_info is not None:
        with open(os.path.join(out_dir, "split_indices.json"), "w") as f:
            json.dump({"seed": cfg.runtime.get("seed"), **split_info}, f, indent=2)

    dirty = _git("status", "--porcelain")
    metadata = {
        "bundle_version": BUNDLE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
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
        metadata["metrics"] = metrics
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Wrote model bundle to {out_dir}")
    return out_dir
