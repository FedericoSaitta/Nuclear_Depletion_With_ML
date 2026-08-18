"""BEAVRS Cycle 1 depletion data generation — power schedule only, no boron.

The measured power history is interpolated onto a uniform grid and integrated in
a single `integrate()` call. Steps at exactly zero power are decay-only inside
OpenMC — no transport, no statepoint — and their tally columns are recorded as
zeros. For every transport step, flux, per-nuclide fission rates and (n,gamma)
capture rates come out of that step's statepoint file.

Which nuclides are tallied, and why, is in `nuclides.py`; extending those lists
extends the dataset's columns automatically.

    uv run --extra sim python data_generation/quarter_datagen.py \
        --config data_generation/configs/beavrs_quarterpin.yaml -c 1 -t 40

The config describes the simulation; the flags describe the machine. `--help`
works without OpenMC installed, because argparse runs before the import.
"""

from cli import sweep_parser

# Parse argv and set OMP_NUM_THREADS BEFORE importing OpenMC, which reads the
# variable at import time. This ordering is why E402 is scoped off for this
# file in pyproject.toml. `cli` is pure argparse, so importing it up here costs
# nothing and needs no OpenMC.
args = sweep_parser(
    description="BEAVRS Cycle 1 quarter-pin depletion data generation",
    config_example="data_generation/configs/beavrs_quarterpin.yaml",
    default_runs=1,
    default_cores=1,
).parse_args()

import os

os.environ["OMP_NUM_THREADS"] = str(args.threads)

import glob
from datetime import datetime

import openmc
import openmc.deplete
from loguru import logger

import config as config_module
import dataset_io
import pin_sim
import tally_io
from common import (
    DAY_IN_SECONDS,
    depletion_results_frame,
    run_sweep,
    save_results,
    setup_paths,
)
from nuclides import CAPTURE_NUCLIDES, FISSION_NUCLIDES
from power_history import resample_power

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _statepoints_by_step(results_dir):
    """Map step index -> statepoint path.

    Tolerates either naming convention OpenMC has used
    (`openmc_simulation_nN.h5` or `statepoint.N.h5`). Decay-only steps write no
    statepoint at all, so their index is simply absent.
    """
    found = {}
    for pattern in ("openmc_simulation_n*.h5", "statepoint.*.h5"):
        for path in glob.glob(os.path.join(results_dir, pattern)):
            base = os.path.basename(path)
            try:
                index = int("".join(c for c in base.split(".")[0] if c.isdigit()))
            except ValueError:
                continue
            found.setdefault(index, path)
    return found


def run_depletion_with_tallies(
    model_parts, chain_file, powers, dt_seconds, fuel_mass_g, worker_id, results_dir
):
    """Single-`integrate()` depletion, then per-step tally extraction.

    *powers* are specific powers in W/g, one per step. Zeros pass straight
    through; OpenMC treats them as decay-only internally.
    """
    _mats, materials, geometry, settings, tallies = model_parts
    num_steps = len(powers)

    model = openmc.model.Model(geometry, materials, settings, tallies)
    operator = openmc.deplete.CoupledOperator(model, chain_file)

    logger.info(
        f"Worker {worker_id} | one integrate() over {num_steps} steps "
        f"(transport: {sum(1 for p in powers if p > 0)}, "
        f"decay-only: {sum(1 for p in powers if p == 0)})"
    )

    openmc.deplete.PredictorIntegrator(
        operator,
        [dt_seconds] * num_steps,
        [p * fuel_mass_g for p in powers],
        timestep_units="s",
    ).integrate()

    step_data = tally_io.new_step_data()
    statepoints = _statepoints_by_step(results_dir)

    for i in range(num_steps):
        if powers[i] == 0.0:
            tally = tally_io.zero_tally()
        else:
            path = statepoints.get(i)
            tally = (
                tally_io.read_statepoint_tallies(path) if path else tally_io.nan_tally()
            )
        tally_io.append_step(step_data, tally)

    return step_data


def generate_data(cfg):
    """One worker: build the model, deplete with tallies, write its HDF5."""
    worker_id = cfg["worker_id"]
    results_dir, chain_file = setup_paths(SCRIPT_DIR, worker_id, cfg["chain_file"])

    logger.info(
        f"Worker {worker_id} | seed={cfg['seed']} | {len(cfg['powers'])} steps | "
        f"dt={cfg['delta_t_days']:.5f} d | particles={cfg['particles']}"
    )
    logger.info(f"Worker {worker_id} | fission tallies: {FISSION_NUCLIDES}")
    logger.info(f"Worker {worker_id} | capture tallies: {CAPTURE_NUCLIDES}")

    model_parts = pin_sim.build_and_export_model(cfg, results_dir, with_tallies=True)
    fuel = model_parts[0][0]
    os.chdir(results_dir)

    fuel_mass_g = cfg["fuel_density"] * fuel.volume
    dt_seconds = cfg["delta_t_days"] * DAY_IN_SECONDS

    step_data = run_depletion_with_tallies(
        model_parts,
        chain_file,
        cfg["powers"],
        dt_seconds,
        fuel_mass_g,
        worker_id,
        results_dir,
    )

    results = openmc.deplete.Results("depletion_results.h5")
    label = f"beavrs_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{worker_id}"
    data = depletion_results_frame(
        results, label, per_step={"power_W_g": cfg["powers"], **step_data}
    )
    save_results(data, cfg["output_dir"], worker_id)


if __name__ == "__main__":
    base_config = config_module.load(args.config)
    output_dir = args.out or os.path.join(SCRIPT_DIR, "data")

    powers, _times = resample_power(
        os.path.join(SCRIPT_DIR, base_config["power_file"]),
        rated_power_W_g=base_config["rated_power"],
        dt_days=base_config["delta_t_days"],
    )
    base_config["powers"] = powers
    base_config["output_dir"] = output_dir

    workers = run_sweep(
        generate_data, base_config, args.runs, args.cores, master_seed=args.seed
    )

    dataset_io.write_manifest(
        output_dir,
        config=config_module.load(args.config),
        worker_configs=workers,
        chain_file=os.path.join(SCRIPT_DIR, "../data", base_config["chain_file"]),
        extra={
            "pipeline": "beavrs_quarterpin",
            "config_path": args.config,
            "num_steps": len(powers),
            "runs": args.runs,
            "workers_per_run": args.cores,
            "master_seed": args.seed,
        },
    )
