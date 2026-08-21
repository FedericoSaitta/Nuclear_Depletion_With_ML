"""Random-history pin-cell depletion — the pipeline behind the CASL datasets.

Each worker builds one fuel pin — `pin_sim`, the same model the BEAVRS pipelines
run — draws a fresh operating history (power, temperatures, moderator density,
boron) from the ranges the config gives, depletes it step by step, and writes the
resulting concentrations to its own HDF5 file in `data_generation/data/`.

The randomised history is what makes this pipeline different from the BEAVRS
ones, and it is why the model is stepped one `integrate()` call at a time: the
operating state lives on the materials, and it changes between steps.

    uv run --extra sim python data_generation/datagen.py \
        --config data_generation/configs/casl_pincell.yaml -n 4 -c 16

The config describes the simulation; the flags describe the machine. `--help`
works without OpenMC installed, because argparse runs before the import.
"""

from cli import sweep_parser

# Parse argv and set OMP_NUM_THREADS BEFORE importing OpenMC, which reads the
# variable at import time. This ordering is why E402 is scoped off for this
# file in pyproject.toml. `cli` is pure argparse, so importing it up here costs
# nothing and needs no OpenMC.
args = sweep_parser(
    description="Randomised-history pin-cell depletion data generation",
    config_example="data_generation/configs/casl_pincell.yaml",
    default_runs=4,
    default_cores=16,
).parse_args()

import os

os.environ["OMP_NUM_THREADS"] = str(args.threads)

from datetime import datetime

import numpy as np
import openmc
import openmc.deplete
from loguru import logger

import config as config_module
import dataset_io
import pin_sim
from common import (
    DAY_IN_SECONDS,
    deplete_one_step,
    depletion_results_frame,
    run_sweep,
    save_results,
    setup_paths,
    specific_burnup,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def run_depletion_simulation(
    model_parts, chain_file, conditions, fuel_mass_g, worker_id, results_dir
):
    """Step the history one `integrate()` call at a time.

    One call per step rather than one for the whole history: the operating state
    changes between steps, so the materials have to be re-exported and the model
    rebuilt each time.
    """
    mats, materials, geometry, settings, _tallies = model_parts
    _fuel, _gap, _clad, water = mats
    num_steps = len(conditions["time_steps"])

    for i in range(num_steps):
        logger.info(f"Worker {worker_id} | step {i + 1}/{num_steps}")

        pin_sim.set_temperatures(
            mats,
            fuel_temp=conditions["fuel_temps"][i],
            clad_temp=conditions["clad_temps"][i],
            mod_temp=conditions["mod_temps"][i],
        )
        pin_sim.update_water_composition(
            water, conditions["boron_ppm"][i], conditions["mod_densities"][i]
        )
        materials.export_to_xml(path=results_dir)

        deplete_one_step(
            openmc.model.Model(geometry, materials, settings),
            chain_file,
            conditions["time_steps"][i],
            conditions["power"][i] * fuel_mass_g,
            continue_from="depletion_results.h5" if i > 0 else None,
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

    model_parts = pin_sim.build_and_export_model(cfg, results_dir)
    fuel = model_parts[0][0]
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
