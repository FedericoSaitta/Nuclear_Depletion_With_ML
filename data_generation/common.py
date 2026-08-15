"""Shared infrastructure for the depletion data-generation entry points.

Extracted verbatim from datagen.py / quarter_datagen.py / zoom_datagen.py,
which had drifted into three near-identical copies. Behaviour is unchanged —
in particular `create_worker_configs` draws exactly the same worker seeds for
the same master seed as it always did, so `-n 1 -s S` datasets stay
reproducible.

NOTE: importing this module imports OpenMC. Entry points must set
OMP_NUM_THREADS *before* importing it, exactly as they must before importing
openmc directly.
"""

import multiprocessing as mp
import os
import random
import uuid

import numpy as np
import openmc
import openmc.deplete
import pandas as pd

HOUR_IN_SECONDS = 3600
DAY_IN_SECONDS = 24 * HOUR_IN_SECONDS


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
            print(f"Worker {worker_id}** Using WMP library from: {wmp_path}")
        else:
            print(
                f"Worker {worker_id}** WARNING: WMP enabled but library not found at {wmp_path}"
            )

    chain_file = os.path.join(script_dir, "../data", chain_filename)
    # Parse once so a missing or malformed chain fails here rather than deep
    # inside the first depletion step. CoupledOperator wants the *path*, not the
    # parsed Chain, so that is what gets handed onward.
    openmc.deplete.Chain.from_xml(chain_file)

    return results_dir, chain_file


def create_worker_configs(base_config, num_workers, master_seed=None):
    """Per-worker configs with unique IDs and Monte-Carlo seeds.

    Seeding is order-sensitive: worker IDs come from `uuid4` (OS entropy) and
    worker seeds from the `random` module seeded here, so for a given master
    seed the seed sequence is identical to what this code has always produced.
    """
    if master_seed is not None:
        random.seed(master_seed)
        print(f"Using master seed: {master_seed}")
    else:
        random.seed()
        print("Using random seed (no master seed specified)")

    configs = []
    for i in range(1, num_workers + 1):
        config = base_config.copy()
        config["worker_id"] = f"{i}_{uuid.uuid4().hex[:8]}"
        config["seed"] = random.randint(1, 2**31 - 1)
        configs.append(config)
    return configs


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

    except Exception as e:
        print(f"Error during parallel data generation: {e}")
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise


def save_results(data, script_dir, worker_id):
    """Append one run's rows to the worker's CSV in data_generation/data/."""
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    df = pd.DataFrame(data)
    file_path = os.path.join(data_dir, f"worker_{worker_id}_nuclide_concentrations.csv")
    file_exists = os.path.isfile(file_path)
    df.to_csv(file_path, mode="a", index=False, header=not file_exists)
    print(f"Worker {worker_id} | Results saved to {file_path}")


def parse_power_history(filepath):
    """Parse a BEAVRS Cycle 1 power-history CSV into (days, percent) arrays.

    The file format is the BEAVRS release table: a "Cycle 1" banner, a "Day,
    Percent" header, then comma-separated rows until the next non-numeric line.
    """
    filepath = os.path.abspath(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Power history file not found: {filepath}")

    days, percents = [], []
    header_found = False

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                if header_found:
                    break
                continue
            if line.lower().startswith("cycle 1"):
                continue
            if line.lower().startswith("day"):
                header_found = True
                continue
            if line.lower().startswith("cycle") or line[0].isalpha():
                break
            try:
                parts = line.split(",")
                days.append(float(parts[0]))
                percents.append(float(parts[1]))
            except (ValueError, IndexError):
                break

    return np.array(days), np.array(percents)
