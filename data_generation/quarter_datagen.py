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

import argparse

# Parse argv and set OMP_NUM_THREADS BEFORE importing OpenMC, which reads the
# variable at import time. This ordering is why E402 is scoped off for this
# file in pyproject.toml.
parser = argparse.ArgumentParser(
    description="BEAVRS Cycle 1 quarter-pin depletion data generation",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--config",
    required=True,
    help="simulation config, e.g. data_generation/configs/beavrs_quarterpin.yaml",
)
parser.add_argument(
    "-n", "--runs", type=int, default=1, help="rounds of parallel workers"
)
parser.add_argument(
    "-c", "--cores", type=int, default=1, help="parallel worker processes per round"
)
parser.add_argument(
    "-t", "--threads", type=int, default=1, help="OpenMP threads per worker"
)
parser.add_argument(
    "-s", "--seed", type=int, default=None, help="master seed (None = random)"
)
parser.add_argument(
    "--out", default=None, help="output directory (default: data_generation/data)"
)
args = parser.parse_args()

import os

os.environ["OMP_NUM_THREADS"] = str(args.threads)

import glob
from datetime import datetime

import openmc
import openmc.deplete
from loguru import logger

import config as config_module
import dataset_io
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
from quarter_sim import (
    create_quarterpin_geometry,
    create_quarterpin_materials,
    create_quarterpin_settings,
    create_quarterpin_tallies,
    set_quarterpin_volumes,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_reactor_model(cfg, results_dir):
    """Build and export the quarter-pin reactor model."""
    fuel, gap, clad, water = create_quarterpin_materials(cfg)
    set_quarterpin_volumes(
        fuel, gap, clad, water, radii=cfg["geometry_radii"], pitch=cfg["geometry_pitch"]
    )

    materials = openmc.Materials([fuel, gap, clad, water])
    geometry = create_quarterpin_geometry(
        (fuel, gap, clad, water),
        radii=cfg["geometry_radii"],
        pitch=cfg["geometry_pitch"],
    )
    settings = create_quarterpin_settings(cfg)
    tallies = create_quarterpin_tallies(fuel)

    geometry.export_to_xml(path=results_dir)
    settings.export_to_xml(path=results_dir)
    materials.export_to_xml(path=results_dir)

    return fuel, materials, geometry, settings, tallies


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
    _fuel, materials, geometry, settings, tallies = model_parts
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

    step_data = {key: [] for key in tally_io.step_keys()}
    statepoints = _statepoints_by_step(results_dir)

    for i in range(num_steps):
        if powers[i] == 0.0:
            tally = tally_io.zero_tally()
        else:
            path = statepoints.get(i)
            tally = (
                tally_io.read_statepoint_tallies(path) if path else tally_io.nan_tally()
            )

        fractions = tally_io.compute_fission_power_fractions(tally)

        step_data["flux"].append(tally["flux"])
        step_data["flux_std"].append(tally["flux_std"])
        for nuc in FISSION_NUCLIDES:
            step_data[f"{nuc}_fission"].append(tally[f"{nuc}_fission"])
            step_data[f"{nuc}_fission_std"].append(tally[f"{nuc}_fission_std"])
            step_data[f"{nuc}_fission_power_frac"].append(
                fractions[f"{nuc}_fission_power_frac"]
            )
        for nuc in CAPTURE_NUCLIDES:
            step_data[f"{nuc}_capture"].append(tally[f"{nuc}_capture"])
            step_data[f"{nuc}_capture_std"].append(tally[f"{nuc}_capture_std"])

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

    model_parts = setup_reactor_model(cfg, results_dir)
    fuel = model_parts[0]
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
