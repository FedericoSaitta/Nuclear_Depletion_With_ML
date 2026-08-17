"""The OpenMC-touching layer shared by the depletion entry points.

Everything here needs OpenMC. Everything that does not lives in `config`,
`dataset_io`, `power_history` and `nuclides`, which import cleanly without it —
that split is what lets CI test any of this at all.

NOTE: importing this module imports OpenMC. Entry points must set
OMP_NUM_THREADS *before* importing it, exactly as they must before importing
openmc directly.
"""

import multiprocessing as mp
import os
import time

import numpy as np
import openmc
import openmc.deplete
from loguru import logger

import dataset_io
from config import DAY_IN_SECONDS, HOUR_IN_SECONDS, create_worker_configs

__all__ = [
    "DAY_IN_SECONDS",
    "HOUR_IN_SECONDS",
    "create_worker_configs",
    "depletion_results_frame",
    "run_parallel_simulations",
    "run_sweep",
    "save_results",
    "setup_paths",
]


def setup_paths(script_dir, worker_id, chain_filename, use_wmp=False):
    """Create the worker's results directory and point OpenMC at its data."""
    results_dir = os.path.abspath(
        os.path.join(script_dir, "results", f"worker_{worker_id}")
    )
    os.makedirs(results_dir, exist_ok=True)

    openmc.config["cross_sections"] = os.path.join(
        script_dir, "../data/cross_sections.xml"
    )
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ["OPENMC_CROSS_SECTIONS"] = str(openmc.config["cross_sections"])

    if use_wmp:
        wmp_path = os.path.join(script_dir, "../data/wmp")
        if os.path.exists(wmp_path):
            os.environ["OPENMC_MULTIPOLE_LIBRARY"] = wmp_path
            logger.info(f"Worker {worker_id} | using WMP library from {wmp_path}")
        else:
            logger.warning(
                f"Worker {worker_id} | WMP enabled but library not found at {wmp_path}"
            )

    chain_file = os.path.join(script_dir, "../data", chain_filename)
    # Parse once so a missing or malformed chain fails here rather than deep
    # inside the first depletion step. CoupledOperator wants the *path*, not the
    # parsed Chain, so that is what gets handed onward.
    openmc.deplete.Chain.from_xml(chain_file)

    return results_dir, chain_file


def run_parallel_simulations(worker_fn, configs):
    """Run *worker_fn(config)* once per config, each in its own process."""
    processes = []
    try:
        for config in configs:
            process = mp.Process(target=worker_fn, args=(config,))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    except Exception:
        logger.exception("Error during parallel data generation")
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise


def run_sweep(worker_fn, base_config, num_runs, num_workers, master_seed=None):
    """Run *num_runs* rounds of *num_workers* parallel workers.

    Returns every worker config that was used, so the caller can record the
    seeds in the run manifest.
    """
    used = []
    for i in range(num_runs):
        # Each outer iteration needs its own master seed. Passing the same one
        # every time made create_worker_configs re-seed `random` identically, so
        # every iteration regenerated the same worker seeds — the same operating
        # histories and the same MC seeds — and the dataset filled with
        # duplicates that then straddled the train/test split. `-n 1 -s S` is
        # unaffected: offset 0 reproduces exactly what it produced before.
        run_seed = None if master_seed is None else master_seed + i
        configs = create_worker_configs(base_config, num_workers, master_seed=run_seed)
        used.extend(configs)

        start = time.perf_counter()
        run_parallel_simulations(worker_fn, configs)
        elapsed = time.perf_counter() - start

        logger.info(
            f"Run {i + 1}/{num_runs} complete | {num_workers} workers | "
            f"{elapsed:.1f}s"
        )
    return used


def depletion_results_frame(results, label, per_step=None, time_offset_days=0.0):
    """Turn an `openmc.deplete.Results` into the dataset's column dict.

    *per_step* holds columns with one value per depletion **step**; the results
    grid has one extra point — the state after the final step — so they are
    padded with NaN to match. Concentrations and k-eff already have one value
    per point.

    *time_offset_days* shifts the time axis, which the zoom pipeline needs: its
    results start at zero but the window sits partway through the cycle, and the
    absolute day is what makes it comparable to the daily run.
    """
    time_seconds, k = results.get_keff()
    time_days = time_seconds / DAY_IN_SECONDS + time_offset_days
    num_points = len(time_days)

    def pad(values):
        values = list(values)
        return values + [float("nan")] * (num_points - len(values))

    data = {
        "run_label": [label] * num_points,
        "time_days": time_days,
        "k_eff": k[:, 0],
        "k_eff_std": k[:, 1],
    }
    for key, values in (per_step or {}).items():
        data[key] = pad(values)

    for nuclide in results[0].index_nuc:
        _, concentration = results.get_atoms("1", nuclide, nuc_units="atom/b-cm")
        data[nuclide] = concentration

    return data


def specific_burnup(powers_W_g, dt_seconds):
    """Cumulative energy released per unit fuel mass, in MWd/kg.

    power [W/g] x step [d] gives W*d/g, and 1 W*d/g = 1e-3 MWd/kg.
    """
    dt_days = np.asarray(dt_seconds, dtype=float) / DAY_IN_SECONDS
    return np.cumsum(np.asarray(powers_W_g, dtype=float) * dt_days) / 1000.0


def save_results(data, output_dir, worker_id):
    """Write one worker's run to `<output_dir>/worker_<id>.h5`."""
    return dataset_io.write_run_h5(
        data, os.path.join(output_dir, f"worker_{worker_id}.h5")
    )
