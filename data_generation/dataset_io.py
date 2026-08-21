"""Writing generated depletion data in the layout the ML pipeline reads.

The generators used to write CSV, which then had to be merged and converted
before training could touch it. They write this layout directly instead, so
there is one format from the simulation to the model:

    numeric_data      (rows, n_numeric) float64, gzip-compressed
    numeric_columns   names of the numeric columns, in order
    string_<col>      one dataset per non-numeric column
    string_columns    names of the non-numeric columns
    all_columns       every column name, in generation order

`src/nuclear_surrogates/datamodule/dataset_helper.py::read_h5_file` is the
consumer. `read_run_h5` below is a second, minimal reader that exists only
because the consumer imports torch and the simulation environment has none;
`tests/test_datagen_io.py` asserts the two agree, so the duplication cannot
drift silently.

No OpenMC import anywhere in this module — that is what lets the format contract
be tested in CI, which has neither OpenMC nor the nuclear-data library.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime

import h5py
import numpy as np
from loguru import logger

MANIFEST_NAME = "run_manifest.json"

# run_label identifies the generating worker. It is the only non-numeric column
# the generators produce; the ML side drops it and detects run boundaries from
# the time axis resetting instead.
STRING_COLUMNS = ("run_label",)


def _string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def write_run_h5(data, path):
    """Write one run's columns to *path*. Returns the path.

    *data* maps column name to a sequence, all of the same length. Insertion
    order is preserved as `all_columns`, so the file reads back in the order the
    generator produced.
    """
    columns = list(data)
    if not columns:
        raise ValueError("refusing to write a dataset with no columns")

    lengths = {len(data[c]) for c in columns}
    if len(lengths) != 1:
        raise ValueError(
            f"columns have differing lengths {sorted(lengths)} — every column "
            f"must have one value per time point"
        )

    string_cols = [c for c in columns if c in STRING_COLUMNS]
    numeric_cols = [c for c in columns if c not in STRING_COLUMNS]

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    with h5py.File(path, "w") as f:
        if numeric_cols:
            block = np.column_stack(
                [np.asarray(data[c], dtype=np.float64) for c in numeric_cols]
            )
            f.create_dataset(
                "numeric_data", data=block, compression="gzip", compression_opts=9
            )
            f.create_dataset(
                "numeric_columns", data=numeric_cols, dtype=_string_dtype()
            )

        if string_cols:
            for col in string_cols:
                values = [("" if v is None else str(v)) for v in data[col]]
                f.create_dataset(f"string_{col}", data=values, dtype=_string_dtype())
            f.create_dataset("string_columns", data=string_cols, dtype=_string_dtype())

        f.create_dataset("all_columns", data=columns, dtype=_string_dtype())

    rows = lengths.pop()
    logger.info(
        f"Wrote {path}: {rows} rows, {len(numeric_cols)} numeric + "
        f"{len(string_cols)} string columns"
    )
    return path


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_run_h5(path):
    """Read a file written by `write_run_h5` into `{column: list}`.

    Minimal on purpose. The ML side's `read_h5_file` is the real reader and
    returns a polars DataFrame; this one exists so the simulation environment,
    which has no torch, can merge its own output.
    """
    with h5py.File(path, "r") as f:
        columns = [_decode(c) for c in f["all_columns"][:]]
        out = {}

        if "numeric_data" in f:
            numeric_cols = [_decode(c) for c in f["numeric_columns"][:]]
            block = f["numeric_data"][:]
            for i, col in enumerate(numeric_cols):
                out[col] = block[:, i]

        if "string_columns" in f:
            for col in [_decode(c) for c in f["string_columns"][:]]:
                out[col] = [_decode(v) for v in f[f"string_{col}"][:]]

        return {c: out[c] for c in columns if c in out}


def merge_frames(paths):
    """Concatenate per-worker files into one column dict, rows in *paths* order.

    Workers can disagree on their column set — a chain change, or a tally that
    failed to score, leaves a column absent from one file and present in
    another. Rather than silently dropping the difference, every column any file
    has appears in the result, padded with NaN (or "" for text) wherever a file
    did not have it. A NaN is visible downstream; a dropped column is not.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("no input files to merge")

    frames = [read_run_h5(p) for p in paths]

    # Union of columns, first-seen order — so the common case (every worker
    # agreeing) preserves the generation order exactly.
    columns = []
    for frame in frames:
        for col in frame:
            if col not in columns:
                columns.append(col)

    merged = {col: [] for col in columns}
    for path, frame in zip(paths, frames, strict=True):
        n_rows = len(next(iter(frame.values())))
        missing = [c for c in columns if c not in frame]
        if missing:
            logger.warning(f"{path} is missing {len(missing)} columns: {missing[:5]}")
        for col in columns:
            if col in frame:
                merged[col].extend(list(frame[col]))
            elif col in STRING_COLUMNS:
                merged[col].extend([""] * n_rows)
            else:
                merged[col].extend([float("nan")] * n_rows)

    logger.info(f"Merged {len(paths)} files: {len(merged[columns[0]])} rows")
    return merged


def merge_h5(paths, output_path):
    """`merge_frames` straight to disk."""
    return write_run_h5(merge_frames(paths), output_path)


def drop_empty_columns(data):
    """Drop numeric columns whose every value is zero. Returns a new dict.

    The depletion chain carries a couple of hundred nuclides and a pin-cell run
    produces exactly zero of most of them, so this is usually most of the file.
    """
    kept = {}
    dropped = []
    for col, values in data.items():
        if col not in STRING_COLUMNS and not np.any(np.asarray(values, dtype=float)):
            dropped.append(col)
            continue
        kept[col] = values
    if dropped:
        logger.info(f"Dropped {len(dropped)} all-zero columns: {dropped[:10]}")
    return kept


# ── Provenance ───────────────────────────────────────────────────────────────
#
# `sha256_file` and `_git` duplicate `src/nuclear_surrogates/bundle.py`. That is
# deliberate, not an oversight: `bundle.py` imports omegaconf at module level and
# omegaconf is in the `ml` extra only, so it cannot be imported from the
# simulation environment at all.


def sha256_file(path, chunk_size=8 << 20):
    """Streamed SHA-256 — depletion chains run to tens of MB."""
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args):
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


def write_manifest(output_dir, config, worker_configs, chain_file, extra=None):
    """Record what produced a dataset, beside the dataset.

    The datagen counterpart of a model bundle's `metadata.json`. Without it a
    published `.h5` is a table of numbers that nobody can regenerate: the chain
    file, the cross-section library and the per-worker seeds are all outside the
    data itself.
    """
    try:
        import openmc

        openmc_version = openmc.__version__
    except ImportError:
        openmc_version = None

    dirty = _git("status", "--porcelain")
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "workers": [
            {"worker_id": w.get("worker_id"), "seed": w.get("seed")}
            for w in worker_configs
        ],
        "chain_file": {
            "path": str(chain_file),
            "name": os.path.basename(str(chain_file)),
            "sha256": sha256_file(chain_file),
        },
        "cross_sections": os.environ.get("OPENMC_CROSS_SECTIONS"),
        "openmc_version": openmc_version,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if extra:
        manifest.update(extra)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, MANIFEST_NAME)
    with open(path, "w") as f:
        # allow_nan=False so a non-finite value fails here rather than producing
        # a file only Python can read back.
        json.dump(manifest, f, indent=2, allow_nan=False)

    logger.info(f"Wrote provenance manifest to {path}")
    return path
