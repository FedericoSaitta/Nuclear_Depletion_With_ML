"""Coarsen a depletion dataset onto a longer timestep.

The DNN is a one-step map with **no notion of elapsed time**: it is trained to
predict the concentration change over one row-to-row step, and nothing in its
input says how long that step was. A model trained at 10-day steps therefore
cannot be evaluated on 1-day data — it emits a 10-day-sized change where a
1-day-sized one belongs. This script matches the step sizes by coarsening the
finer dataset.

    uv run --extra ml python util/resample_dataset.py \\
        datasets/beavrs_cycle1_daily.h5 \\
        datasets/beavrs_cycle1_10day.h5 --factor 10

Columns are not all coarsened the same way, and getting this wrong is the whole
risk of the operation:

* **State** (`time_days`, `k_eff`, every nuclide concentration) is the value at
  an instant, so the coarse grid takes the value at the window's start.
* **Per-step rates** (`power_W_g`, `flux`, the fission/capture tallies) describe
  the step *leaving* that row, so a coarse step's value is the mean over the
  fine steps it spans. For power this is the right choice on purpose: the model
  was trained where power is constant across a step, and the energy-equivalent
  constant power for a varying window is its mean.
* **Cumulative** quantities (`burnup_MWd_kg`) take the value at the window's
  end, since they already integrate what came before.

The two classes are told apart structurally: `depletion_results_frame` pads
per-step columns with a trailing NaN because the results grid has one more point
than it has steps. This script asserts that signature rather than trusting the
column names alone.

**Known limitation.** Mean power is exact for the accumulating and depleting
species, whose change over a window depends on the integral of power. It is
*not* exact for the short-lived equilibrium nuclides (U239, Np239), whose
concentration tracks the instantaneous power at the grid point rather than the
window average. The source data has no such tension because its own steps hold
power constant; a coarsened window does. Expect the equilibrium species to
degrade slightly relative to a natively-generated coarse dataset.
"""

import argparse
import os
import re
import sys

import numpy as np
import polars as pl

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_generation"),
)

import dataset_io  # noqa: E402

# Per-step columns: one value per depletion step, describing the step that
# leaves this row. Matches what the generators put in `per_step`.
PER_STEP_EXACT = {"power_W_g", "flux", "flux_std"}
PER_STEP_PATTERN = re.compile(r"_(fission|capture)(_std|_power_frac)?$")

# Per-step but cumulative: already an integral, so take the window's end value.
CUMULATIVE = {"burnup_MWd_kg"}

TEXT_COLUMNS = {"run_label"}


def classify(name):
    if name in TEXT_COLUMNS:
        return "text"
    if name in CUMULATIVE:
        return "cumulative"
    if name in PER_STEP_EXACT or PER_STEP_PATTERN.search(name):
        return "per_step"
    return "state"


def read_dataset(path):
    if path.endswith(".h5"):
        return dataset_io.read_run_h5(path)
    df = pl.read_csv(path)
    return {c: df[c].to_numpy() for c in df.columns}


def write_dataset(data, path):
    if path.endswith(".h5"):
        return dataset_io.write_run_h5(data, path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    pl.DataFrame({k: list(v) for k, v in data.items()}).write_csv(path)
    return path


def _check_padding(data, per_step, state):
    """The trailing-NaN signature is what distinguishes the two classes.

    If a column's name says one thing and its padding says another, the schema
    has changed under this script and coarsening it would silently corrupt the
    output — so it stops.
    """
    disagree = []
    for name in per_step:
        values = np.asarray(data[name], dtype=float)
        if not np.isnan(values[-1]):
            disagree.append(f"{name}: classified per-step but its last row is not NaN")
    for name in state:
        values = np.asarray(data[name], dtype=float)
        if values.size and np.isnan(values[-1]) and not np.all(np.isnan(values)):
            disagree.append(f"{name}: classified state but its last row is NaN")
    if disagree:
        raise SystemExit(
            "Column padding does not match the expected schema:\n  "
            + "\n  ".join(disagree[:10])
            + f"\n({len(disagree)} column(s) total). Update the classification "
            f"rules in {__file__} before coarsening this file."
        )


def resample(data, factor):
    time = np.asarray(data["time_days"], dtype=float)
    n_rows = len(time)

    steps = np.diff(time)
    if not np.allclose(steps, steps[0]):
        raise SystemExit(
            f"Source timestep is not uniform (min {steps.min()}, max {steps.max()}); "
            f"coarsening assumes a uniform grid."
        )
    print(f"Source: {n_rows} rows at {steps[0]:g} day steps")

    classes = {name: classify(name) for name in data}
    per_step = [n for n, c in classes.items() if c in ("per_step", "cumulative")]
    state = [n for n, c in classes.items() if c == "state"]
    _check_padding(data, per_step, state)

    # Grid points of the coarse dataset. A coarse step leaving row `i` spans
    # fine steps i .. i+factor-1, so the last point that has a complete window
    # is the one that leaves `factor` fine steps ahead of the final NaN pad.
    starts = np.arange(0, n_rows - 1, factor)
    complete = starts + factor <= n_rows - 1

    out = {}
    for name, values in data.items():
        kind = classes[name]
        if kind in ("state", "text"):
            out[name] = [values[i] for i in starts]
        elif kind == "cumulative":
            # Value at the window's end; NaN where the window is incomplete.
            out[name] = [
                float(values[i + factor]) if ok else float("nan")
                for i, ok in zip(starts, complete, strict=True)
            ]
        else:
            arr = np.asarray(values, dtype=float)
            out[name] = [
                float(np.nanmean(arr[i : i + factor])) if ok else float("nan")
                for i, ok in zip(starts, complete, strict=True)
            ]

    dropped = n_rows - 1 - starts[-1] - (factor if complete[-1] else 0)
    print(
        f"Output: {len(starts)} rows at {steps[0] * factor:g} day steps "
        f"({int(complete.sum())} complete steps, "
        f"{len(starts) - int(complete.sum())} trailing row(s) NaN-padded)"
    )
    if dropped > 0:
        print(f"  {dropped} trailing fine row(s) dropped: incomplete final window")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Coarsen a depletion dataset onto a longer timestep"
    )
    parser.add_argument("input", help="source dataset (.csv or .h5)")
    parser.add_argument("output", help="where to write the coarsened dataset")
    parser.add_argument(
        "--factor",
        type=int,
        required=True,
        help="how many source steps make one output step (e.g. 10 for 1d -> 10d)",
    )
    args = parser.parse_args()

    if args.factor < 2:
        raise SystemExit("--factor must be at least 2")

    data = read_dataset(args.input)
    write_dataset(resample(data, args.factor), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
