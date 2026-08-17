"""Random-history pin-cell depletion — the pipeline behind the CASL datasets.

Each worker builds one fuel pin, draws a fresh operating history (power,
temperatures, moderator density, boron) from the ranges the config gives,
depletes it step by step, and writes the resulting concentrations to its own
HDF5 file in `data_generation/data/`.

    uv run --extra sim python data_generation/datagen.py \
        --config data_generation/configs/casl_pincell.yaml -n 4 -c 16

The config describes the simulation; the flags describe the machine. `--help`
works without OpenMC installed, because argparse runs before the import.
"""

import argparse

# Parse argv and set OMP_NUM_THREADS BEFORE importing OpenMC, which reads the
# variable at import time. This ordering is why E402 is scoped off for this
# file in pyproject.toml.
parser = argparse.ArgumentParser(
    description="Randomised-history pin-cell depletion data generation",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--config",
    required=True,
    help="simulation config, e.g. data_generation/configs/casl_pincell.yaml",
)
parser.add_argument(
    "-n", "--runs", type=int, default=4, help="rounds of parallel workers"
)
parser.add_argument(
    "-c", "--cores", type=int, default=16, help="parallel worker processes per round"
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

from datetime import datetime

import numpy as np
import openmc
import openmc.deplete
from loguru import logger

import config as config_module
import dataset_io
from common import (
    DAY_IN_SECONDS,
    depletion_results_frame,
    run_sweep,
    save_results,
    setup_paths,
    specific_burnup,
)
from reactor_sim import (
    create_pincell_geometry,
    create_pincell_materials,
    create_pincell_settings,
    set_pincell_volumes,
    update_water_composition,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_reactor_model(cfg, results_dir):
    fuel, clad, water = create_pincell_materials(cfg)
    set_pincell_volumes(
        fuel, clad, water, radii=cfg["geometry_radii"], pitch=cfg["geometry_pitch"]
    )

    materials = openmc.Materials([fuel, clad, water])
    geometry = create_pincell_geometry(
        materials, radii=cfg["geometry_radii"], pitch=cfg["geometry_pitch"]
    )
    settings = create_pincell_settings(cfg)

    geometry.export_to_xml(path=results_dir)
    settings.export_to_xml(path=results_dir)

    return fuel, clad, water, materials, geometry, settings


def generate_random_conditions(cfg):
    """Draw one operating history from the config's per-step ranges."""
    num_steps = cfg["num_steps"]
    delta_t = cfg["delta_t_days"] * DAY_IN_SECONDS

    def draw(key):
        low, high = cfg[key]
        return list(np.random.uniform(low, high, num_steps))

    return {
        "time_steps": [delta_t] * num_steps,
        "power": draw("power"),
        "fuel_temps": draw("t_fuel"),
        "mod_temps": draw("t_mod"),
        "clad_temps": draw("t_clad"),
        "mod_densities": draw("rho_mod"),
        "boron_ppm": draw("boron_ppm"),
    }


def run_depletion_step(model, chain_file, time_step, power_watts, prev_results=None):
    if prev_results and os.path.exists(prev_results):
        operator = openmc.deplete.CoupledOperator(
            model, chain_file, prev_results=openmc.deplete.Results(prev_results)
        )
    else:
        operator = openmc.deplete.CoupledOperator(model, chain_file)

    openmc.deplete.PredictorIntegrator(
        operator, [time_step], [power_watts], timestep_units="s"
    ).integrate()


def run_depletion_simulation(
    model_parts, chain_file, conditions, fuel_mass_g, worker_id, results_dir
):
    """Step the history one `integrate()` call at a time.

    One call per step rather than one for the whole history: the operating state
    changes between steps, so the materials have to be re-exported and the model
    rebuilt each time.
    """
    fuel, clad, water, materials, geometry, settings = model_parts
    num_steps = len(conditions["time_steps"])

    for i in range(num_steps):
        logger.info(f"Worker {worker_id} | step {i + 1}/{num_steps}")

        fuel.temperature = conditions["fuel_temps"][i]
        water.temperature = conditions["mod_temps"][i]
        clad.temperature = conditions["clad_temps"][i]
        update_water_composition(
            water, conditions["boron_ppm"][i], conditions["mod_densities"][i]
        )
        materials.export_to_xml(path=results_dir)

        run_depletion_step(
            openmc.model.Model(geometry, materials, settings),
            chain_file,
            conditions["time_steps"][i],
            conditions["power"][i] * fuel_mass_g,
            prev_results="depletion_results.h5" if i > 0 else None,
        )


def generate_data(cfg):
    """One worker: build the pin, deplete a random history, write its HDF5."""
    worker_id = cfg["worker_id"]
    results_dir, chain_file = setup_paths(
        SCRIPT_DIR, worker_id, cfg["chain_file"], use_wmp=cfg.get("use_wmp", False)
    )

    logger.info(
        f"Worker {worker_id} | seed={cfg['seed']} | chain={cfg['chain_file']} | "
        f"{cfg['num_steps']} steps of {cfg['delta_t_days']} d"
    )
    if cfg.get("use_wmp", False):
        logger.info(f"Worker {worker_id} | WMP on-the-fly Doppler broadening")
    else:
        logger.info(
            f"Worker {worker_id} | temperature method: "
            f"{cfg.get('temp_method', 'interpolation')}"
        )

    np.random.seed(cfg["seed"])

    model_parts = setup_reactor_model(cfg, results_dir)
    fuel = model_parts[0]
    os.chdir(results_dir)

    fuel_mass_g = cfg["fuel_density"] * fuel.volume
    conditions = generate_random_conditions(cfg)

    run_depletion_simulation(
        model_parts, chain_file, conditions, fuel_mass_g, worker_id, results_dir
    )

    results = openmc.deplete.Results("depletion_results.h5")
    label = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{worker_id}"
    data = depletion_results_frame(
        results,
        label,
        per_step={
            "power_W_g": conditions["power"],
            "burnup_MWd_kg": specific_burnup(
                conditions["power"], conditions["time_steps"]
            ),
            "fuel_temp_K": conditions["fuel_temps"],
            "mod_temp_K": conditions["mod_temps"],
            "clad_temp_K": conditions["clad_temps"],
            "mod_density_g_cm3": conditions["mod_densities"],
            "boron_ppm": conditions["boron_ppm"],
        },
    )
    save_results(data, cfg["output_dir"], worker_id)


if __name__ == "__main__":
    base_config = config_module.load(args.config)
    output_dir = args.out or os.path.join(SCRIPT_DIR, "data")
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
            "pipeline": "casl_pincell",
            "config_path": args.config,
            "runs": args.runs,
            "workers_per_run": args.cores,
            "master_seed": args.seed,
        },
    )
