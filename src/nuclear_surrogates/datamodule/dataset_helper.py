# Reshape and mold the csv/h5 dataset to be more approachable for ML
import re

import h5py
import numpy as np
import polars as pl
import torch
from loguru import logger
from torch.utils.data import TensorDataset

# ── I/O helpers ──────────────────────────────────────────────────────────────


def _decode(value):
    """Decode bytes to str, pass-through if already a string."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_h5_file(file_path, columns=None):
    """Read the HDF5 written by the datagen pipeline into a polars DataFrame.

    With *columns*, only those are pulled out of `numeric_data`. The files hold
    every nuclide in the depletion chain — 238 columns for the CASL set — while a
    run configures a handful, so reading the lot costs seconds and hundreds of MB
    that are then discarded.
    """
    with h5py.File(file_path, "r") as f:
        all_columns = [_decode(c) for c in f["all_columns"][:]]
        wanted = None if columns is None else set(columns)
        data_dict = {}

        if "numeric_data" in f:
            numeric_cols = [_decode(c) for c in f["numeric_columns"][:]]
            if wanted is None:
                numeric_data = f["numeric_data"][:]
                keep = list(enumerate(numeric_cols))
            else:
                # h5py needs the indices in increasing order; the DataFrame is
                # reordered by `all_columns` below regardless.
                keep_idx = sorted(
                    i for i, col in enumerate(numeric_cols) if col in wanted
                )
                numeric_data = f["numeric_data"][:, keep_idx]
                keep = [(pos, numeric_cols[i]) for pos, i in enumerate(keep_idx)]

            for i, col in keep:
                data_dict[col] = numeric_data[:, i]

        if "string_columns" in f:
            for col in [_decode(c) for c in f["string_columns"][:]]:
                if wanted is not None and col not in wanted:
                    continue
                raw = f[f"string_{col}"][:]
                data_dict[col] = (
                    [_decode(s) for s in raw] if raw.dtype.kind in ("S", "O") else raw
                )

        ordered = [c for c in all_columns if c in data_dict]
        return pl.DataFrame(data_dict).select(ordered)


def read_data(file_path, fraction_of_data, drop_run_label=True, columns=None):
    """Read a run's data, keeping only *columns* (plus what this function needs).

    Passing the columns a run actually configures avoids materialising the whole
    file; `time_days` is always included because the run-length detection and the
    NODE's time axis both need it.
    """
    logger.info(f"Reading data from: {file_path}")

    if columns is not None:
        columns = [*dict.fromkeys([*columns, "time_days"])]

    if file_path.endswith(".csv"):
        df = pl.read_csv(file_path)
        if columns is not None:
            df = df.select([c for c in df.columns if c in set(columns)])
    elif file_path.endswith(".h5"):
        df = read_h5_file(file_path, columns=columns)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    df = remove_empty_columns(df)

    if drop_run_label and "run_label" in df.columns:
        df = df.drop("run_label")

    if df.is_duplicated().any():
        logger.warning("DataFrame contains duplicate rows.")

    run_length = detect_run_length(df)
    logger.info(f"Detected run length: {run_length}")

    if fraction_of_data < 1.0:
        total_runs = df.shape[0] // run_length
        runs_kept = int(fraction_of_data * total_runs)
        df = df.slice(0, runs_kept * run_length)
        logger.info(f"Keeping {fraction_of_data * 100:.0f}% of data ({runs_kept} runs)")

    time_array = df["time_days"].to_numpy()
    return df, run_length, time_array


# ── DataFrame inspection / cleaning ──────────────────────────────────────────


def remove_empty_columns(df):
    """Drop columns where every value is 0 or null."""
    empty = [col for col in df.columns if ((df[col] == 0) | df[col].is_null()).all()]
    if empty:
        logger.warning(f"Removed empty columns: {empty}")
        return df.drop(empty)

    logger.info("No empty columns to remove.")
    return df


def detect_run_length(df, time_col="time_days"):
    """Return the number of timesteps in the first run (detected via time resets)."""
    time = df[time_col].to_numpy()
    resets = np.where(np.diff(time) <= 0)[0]
    return int(resets[0] + 1) if len(resets) > 0 else len(df)


def print_dataset_stats(df):
    logger.info("=== Dataset Statistics ===")
    logger.info(f"Rows (timesteps): {df.shape[0]}  |  Columns: {df.shape[1]}")

    numeric_dtypes = {pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64}
    numeric_cols = [c for c in df.columns if df[c].dtype in numeric_dtypes]
    isotope_re = re.compile(r"^[A-Za-z]+\d+(_.*)?$")
    isotope_cols = [c for c in numeric_cols if isotope_re.match(c)]

    first_row = df.row(0, named=True)
    nonzero = [c for c in isotope_cols if first_row[c] != 0]

    logger.info(
        f"Element columns: {len(isotope_cols)}  |  State columns: {len(df.columns) - len(isotope_cols)}"
    )
    logger.info(f"Isotopes with non-zero concentration at t=0: {nonzero}")


# ── Column selection / splitting ─────────────────────────────────────────────


def split_df(df, keys):
    """Select *keys* that exist in *df*, return numpy array + column index map."""
    cols = [c for c in df.columns if c in keys]
    subset = df.select(cols)
    col_map = {col: idx for idx, col in enumerate(subset.columns)}
    return subset.to_numpy(), col_map


def ordered_names(index_map):
    """Column names in array order, from a {name: column_index} map.

    This — not the config's dict order — is the order the data arrays are laid
    out in. Every place that pairs a name with an array column must use it, or
    a config that lists targets in a different order than they appear among the
    inputs would silently mislabel every per-target metric and figure.
    """
    return [name for name, _ in sorted(index_map.items(), key=lambda kv: kv[1])]


# ── Time-series target creation ──────────────────────────────────────────────


def _find_run_end_indices(time_values):
    """Return indices of the last timestep in each run."""
    resets = np.where(np.diff(time_values) <= 0)[0]
    return np.append(resets, len(time_values) - 1)


def create_timeseries_targets(
    input_data, target_data, time_values, input_col_map, target_elements, delta_conc
):
    for el in target_elements:
        if el not in input_col_map:
            logger.warning(f"Target feature '{el}' not found in inputs")

    logger.info(f"Input column map: {input_col_map}")

    run_ends = _find_run_end_indices(time_values)

    # Every index except run-ending rows is valid (we need t+1 to exist within the same run)
    valid = np.ones(len(input_data), dtype=bool)
    valid[run_ends] = False
    idx = np.where(valid)[0]

    X = input_data[idx]
    y = (
        (target_data[idx + 1] - target_data[idx])
        if delta_conc
        else target_data[idx + 1]
    )

    logger.info(f"Runs: {len(run_ends)}  |  Samples: {len(X)}")
    logger.info(f"Input shape: {X.shape}  |  Target shape: {y.shape}")
    return X, y


# ── Train / val / test splitting ─────────────────────────────────────────────


SPLIT_STRATEGIES = ("sequential_by_run", "random_by_run")


def split_fractions(cfg, default):
    """Read `dataset.split` as (train, val, test), falling back to *default*.

    Only the fractions live here; which runs land in which split is
    `split_strategy`'s business.
    """
    section = cfg.dataset.get("split") if "dataset" in cfg else None
    if section is None:
        return default

    fractions = tuple(float(section[name]) for name in ("train", "val", "test"))
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(
            f"dataset.split must sum to 1.0, got {fractions} summing to "
            f"{sum(fractions)}"
        )
    if any(f < 0 for f in fractions):
        raise ValueError(
            f"dataset.split fractions must be non-negative, got {fractions}"
        )
    return fractions


def split_strategy(cfg, default):
    """Read `dataset.split.strategy`, falling back to *default*.

    Both models support both strategies, so a head-to-head comparison can hold
    out the same runs. The default is each model's historical behaviour, so a
    config written before this key existed — including the one archived inside a
    bundle — still rebuilds the split it was trained with.
    """
    section = cfg.dataset.get("split") if "dataset" in cfg else None
    strategy = default if section is None else section.get("strategy", default)
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(
            f"Unknown dataset.split.strategy {strategy!r}. "
            f"Choose from {list(SPLIT_STRATEGIES)}"
        )
    return strategy


def run_permutation(num_runs, seed):
    """The seeded run order both models partition by under `random_by_run`.

    The DNN and the NODE have to draw the *same* permutation for their test sets
    to contain the same runs, which is the entire point of the strategy. One
    expression, called from both datamodules, so they cannot drift apart again.
    """
    return np.random.default_rng(seed).permutation(num_runs)


def _run_gather(perm, steps_per_run):
    """Row indices that reorder a flat (runs*steps, features) array by *perm*."""
    return (perm[:, None] * steps_per_run + np.arange(steps_per_run)).ravel()


def timeseries_train_val_test_split(
    X,
    Y,
    train_frac=0.8,
    val_frac=0.1,
    test_frac=0.1,
    steps_per_run=100,
    shuffle_within_train=True,
    rng=None,
    strategy="sequential_by_run",
    seed=None,
):
    """Split by whole runs — never mid-run — under one of two strategies.

    ``"sequential_by_run"`` takes contiguous slices in file order and shuffles
    the training runs afterwards; it is the historical behaviour and stays the
    default, so a caller that predates this argument is unaffected.

    ``"random_by_run"`` partitions a seeded permutation of the runs, which is
    what `NODE_Datamodule` does. Both go through `run_permutation`, so the same
    *seed* and run count put the same runs in the same splits for both models —
    the head-to-head comparison is then paired rather than merely matched.

    *rng* and *shuffle_within_train* apply only to the sequential strategy: the
    permutation already randomises run order, and the training DataLoader
    shuffles samples on top of it.

    Returns the six arrays plus a ``split_info`` dict recording which original
    run indices landed in which split, so a run's partition can be audited later.
    """
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(
            f"Unknown split strategy {strategy!r}. Choose from {list(SPLIT_STRATEGIES)}"
        )
    if rng is None:
        rng = np.random.default_rng()

    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(
            f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"
        )

    total_runs = len(X) // steps_per_run
    remainder = len(X) % steps_per_run
    if remainder:
        logger.warning(f"Discarding last {remainder} samples (not a full run)")
        X, Y = X[: total_runs * steps_per_run], Y[: total_runs * steps_per_run]

    n_train = int(total_runs * train_frac)
    n_val = int(total_runs * val_frac)
    n_test = total_runs - n_train - n_val

    logger.info(
        f"Runs — train: {n_train}, val: {n_val}, test: {n_test}  "
        f"(steps/run: {steps_per_run}, strategy: {strategy})"
    )

    # The run order the contiguous slices below cut through. `perm[i]` is the
    # original index of the run now sitting at position i, which is what makes
    # `split_info` readable against the source file either way.
    if strategy == "random_by_run":
        if seed is None:
            raise ValueError(
                "random_by_run needs a seed — the split is a function of it, and "
                "an unseeded one could not be rebuilt from a bundle"
            )
        perm = run_permutation(total_runs, seed)
        rows = _run_gather(perm, steps_per_run)
        X, Y = X[rows], Y[rows]
    else:
        perm = np.arange(total_runs)

    t1 = n_train * steps_per_run
    t2 = t1 + n_val * steps_per_run

    X_train, y_train = X[:t1], Y[:t1]
    X_val, y_val = X[t1:t2], Y[t1:t2]
    X_test, y_test = X[t2:], Y[t2:]

    # Shuffle entire runs (not individual timesteps) within the training set.
    # Redundant under random_by_run, where the permutation already did it.
    order = None
    if strategy == "sequential_by_run" and shuffle_within_train and n_train > 1:
        order = rng.permutation(n_train)
        X_train = np.concatenate(
            [X_train[i * steps_per_run : (i + 1) * steps_per_run] for i in order]
        )
        y_train = np.concatenate(
            [y_train[i * steps_per_run : (i + 1) * steps_per_run] for i in order]
        )
        logger.info("Shuffled training runs (timestep order within runs preserved)")

    logger.info(
        f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )

    split_info = {
        "strategy": strategy,
        "fractions": [train_frac, val_frac, test_frac],
        "n_runs": total_runs,
        "steps_per_run": steps_per_run,
        "train": perm[:n_train].tolist(),
        "val": perm[n_train : n_train + n_val].tolist(),
        "test": perm[n_train + n_val :].tolist(),
        "train_shuffle_order": None if order is None else order.tolist(),
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, split_info


# ── Scaling & tensor conversion ──────────────────────────────────────────────


def ensure_2d(arr):
    """Reshape to (n, 1) if 1-D, otherwise pass through."""
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _to_tensor(arr, name):
    """Convert to float32, refusing NaNs.

    A NaN reaching this point is upstream data corruption. It used to be
    silently patched to -1 — a plausible-looking scaled value that trained and
    evaluated without complaint — so now it fails instead.
    """
    t = torch.tensor(arr, dtype=torch.float32)
    if torch.isnan(t).any():
        raise ValueError(
            f"{name} contains NaNs — refusing to build a dataset from corrupt "
            f"data. Check the source file and the run-boundary handling."
        )
    return t


def create_tensor_datasets(X_train, X_val, X_test, y_train, y_val, y_test):
    return (
        TensorDataset(_to_tensor(X_train, "X_train"), _to_tensor(y_train, "y_train")),
        TensorDataset(_to_tensor(X_val, "X_val"), _to_tensor(y_val, "y_val")),
        TensorDataset(_to_tensor(X_test, "X_test"), _to_tensor(y_test, "y_test")),
    )
