"""Random-history pin-cell depletion — the pipeline behind the CASL datasets.

Each worker builds one fuel pin, draws a fresh 100-step operating history
(power, temperatures, moderator density, boron) from the configured ranges,
depletes it step by step, and appends the resulting concentrations to its own
CSV in data_generation/data/.
"""

import argparse

# Parse argv and set OMP_NUM_THREADS BEFORE importing OpenMC, which reads the
# variable at import time. This ordering is why E402 is scoped off for this
# file in pyproject.toml.
parser = argparse.ArgumentParser(description="Run parallel depletion simulations")
parser.add_argument(
    "-n",
    "--runs",
    type=int,
    default=4,
    help="Number of data generation runs (default: 4)",
)
parser.add_argument(
    "-c",
    "--cores",
    type=int,
    default=16,
    help="Number of parallel worker processes (default: 16)",
)
parser.add_argument(
    "-f",
    "--file",
    type=str,
    default="chain_casl_pwr.xml",
    help="Depletion chain file name (default: chain_casl_pwr.xml)",
)
parser.add_argument(
    "-t",
    "--threads",
    type=int,
    default=1,
    help="Number of OpenMP threads per worker (default: 1)",
)
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=None,
    help="Master random seed for reproducibility (default: None for random)",
)
parser.add_argument(
    "-m",
    "--temp-method",
    type=str,
    default="interpolation",
    choices=["interpolation", "nearest"],
    help="Temperature interpolation method for cross sections (default: interpolation)",
)
parser.add_argument(
    "-w",
    "--use-wmp",
    action="store_true",
    help="Use Windowed Multipole for on-the-fly Doppler broadening (overrides temp-method)",
)
args = parser.parse_args()

import os

os.environ["OMP_NUM_THREADS"] = str(args.threads)

import time
from datetime import datetime

import numpy as np
import openmc
import openmc.deplete

from common import (
    DAY_IN_SECONDS,
    create_worker_configs,
    run_parallel_simulations,
    save_results,
    setup_paths,
)
from reactor_sim import (
    create_materials,
    set_material_volumes,
    create_geometry,
    create_settings,
    update_water_composition,
)


def setup_reactor_model(config, results_dir):
    fuel, clad, water = create_materials(config)
    set_material_volumes(
        fuel,
        clad,
        water,
        radii=config["geometry_radii"],
        pitch=config["geometry_pitch"],
    )

    materials = openmc.Materials([fuel, clad, water])
    geometry = create_geometry(
        materials, radii=config["geometry_radii"], pitch=config["geometry_pitch"]
    )
    settings = create_settings(config)
    settings.verbosity = 1
    settings.output = {"tallies": False}

    geometry.export_to_xml(path=results_dir)
    settings.export_to_xml(path=results_dir)

    return fuel, clad, water, materials, geometry, settings


def generate_random_conditions(config):
    num_steps = config["num_depl_steps"]
    return {
        "time_steps": [config["delta_t"]] * num_steps,
        "power": list(
            np.random.uniform(config["power"][0], config["power"][1], num_steps)
        ),
        "fuel_temps": list(
            np.random.uniform(config["t_fuel"][0], config["t_fuel"][1], num_steps)
        ),
        "mod_temps": list(
            np.random.uniform(config["t_mod"][0], config["t_mod"][1], num_steps)
        ),
        "clad_temps": list(
            np.random.uniform(config["t_clad"][0], config["t_clad"][1], num_steps)
        ),
        "mod_densities": list(
            np.random.uniform(config["rho_mod"][0], config["rho_mod"][1], num_steps)
        ),
        "boron_ppm": list(
            np.random.uniform(config["boron_ppm"][0], config["boron_ppm"][1], num_steps)
        ),
    }


def run_depletion_step(
    model, chain_file, time_step, power_watts, prev_results_file=None
):
    if prev_results_file and os.path.exists(prev_results_file):
        prev_results = openmc.deplete.Results(prev_results_file)
        operator = openmc.deplete.CoupledOperator(
            model, chain_file, prev_results=prev_results
        )
    else:
        operator = openmc.deplete.CoupledOperator(model, chain_file)

    integrator = openmc.deplete.PredictorIntegrator(
        operator, [time_step], [power_watts], timestep_units="s"
    )
    integrator.integrate()


def run_depletion_simulation(
    fuel,
    clad,
    water,
    materials,
    geometry,
    settings,
    chain_file,
    conditions,
    fuel_mass_g,
    worker_id,
    results_dir,
):
    num_steps = len(conditions["time_steps"])

    for i in range(num_steps):
        print(f"Worker {worker_id}** Step {i+1}/{num_steps}")

        fuel.temperature = conditions["fuel_temps"][i]
        water.temperature = conditions["mod_temps"][i]
        clad.temperature = conditions["clad_temps"][i]

        update_water_composition(
            water, conditions["boron_ppm"][i], conditions["mod_densities"][i]
        )

        materials.export_to_xml(path=results_dir)

        model = openmc.model.Model(geometry, materials, settings)
        power_watts = conditions["power"][i] * fuel_mass_g
        prev_results_file = "depletion_results.h5" if i > 0 else None

        run_depletion_step(
            model,
            chain_file,
            conditions["time_steps"][i],
            power_watts,
            prev_results_file,
        )


def extract_results_data(results, conditions, worker_id):
    time, k = results.get_keff()
    time /= DAY_IN_SECONDS

    # Label the run by timestamp plus the worker's unique tag. The uuid suffix
    # in worker_id is what keeps labels distinct when several workers start
    # within the same second.
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"run_{current_time}_{worker_id}"

    # Specific burnup: cumulative energy released per unit fuel mass.
    # power [W/g] x step [d] gives W*d/g, and 1 W*d/g = 1e-3 MWd/kg.
    dt_days = np.asarray(conditions["time_steps"]) / DAY_IN_SECONDS
    burnup_MWd_kg = np.cumsum(np.asarray(conditions["power"]) * dt_days) / 1000.0

    # Operating-state columns have one value per step; the results grid has one
    # extra point (the state after the final step), padded with NaN.
    nan = float("nan")
    data = {
        "run_label": [label] * len(time),
        "time_days": time,
        "k_eff": k[:, 0],
        "k_eff_std": k[:, 1],
        "power_W_g": conditions["power"] + [nan],
        "burnup_MWd_kg": list(burnup_MWd_kg) + [nan],
        "fuel_temp_K": conditions["fuel_temps"] + [nan],
        "mod_temp_K": conditions["mod_temps"] + [nan],
        "clad_temp_K": conditions["clad_temps"] + [nan],
        "mod_density_g_cm3": conditions["mod_densities"] + [nan],
        "boron_ppm": conditions["boron_ppm"] + [nan],
    }

    nuclides = results[0].index_nuc.keys()
    for nuclide in nuclides:
        _, concentration = results.get_atoms("1", nuclide, nuc_units="atom/b-cm")
        data[nuclide] = concentration

    return data, nuclides


def generate_data(config):
    worker_id = config["worker_id"]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir, chain_file = setup_paths(
        script_dir,
        worker_id,
        config["chain_file"],
        use_wmp=config.get("use_wmp", False),
    )

    print(f"Using depletion file: {config['chain_file']}")
    print(f"Worker {worker_id}** Using seed: {config['seed']}")

    if config.get("use_wmp", False):
        print(f"Worker {worker_id}** Using WMP for on-the-fly Doppler broadening")
    else:
        print(f"Worker {worker_id}** Using temperature method: {config['temp_method']}")

    np.random.seed(config["seed"])

    fuel, clad, water, materials, geometry, settings = setup_reactor_model(
        config, results_dir
    )
    os.chdir(results_dir)

    fuel_mass_g = config["fuel_density"] * fuel.volume
    conditions = generate_random_conditions(config)

    run_depletion_simulation(
        fuel,
        clad,
        water,
        materials,
        geometry,
        settings,
        chain_file,
        conditions,
        fuel_mass_g,
        worker_id,
        results_dir,
    )

    results = openmc.deplete.Results("depletion_results.h5")
    data, nuclides = extract_results_data(results, conditions, worker_id)
    save_results(data, script_dir, worker_id)


if __name__ == "__main__":
    base_config = {
        "num_depl_steps": 100,
        "delta_t": 10 * DAY_IN_SECONDS,
        "enrichment": 3.1,
        "fuel_density": 10.4,
        "particles": 5_000,
        "inactive": 10,
        "batches": 50,
        "temp_method": args.temp_method,
        "use_wmp": args.use_wmp,
        "power": [0, 41.4],
        "t_fuel": [600, 1200],
        "t_mod": [550, 600],
        "t_clad": [570, 670],
        "rho_mod": [0.74, 1.00],
        "boron_ppm": [0, 1000],
        "geometry_radii": [0.39218, 0.45720],
        "geometry_pitch": 1.25984,
        "chain_file": args.file,
    }

    DATA_GEN_RUNS = args.runs
    NUM_WORKERS = args.cores

    for i in range(DATA_GEN_RUNS):
        # Each outer iteration needs its own master seed. Passing the same one
        # every time made create_worker_configs re-seed `random` identically, so
        # every iteration regenerated the same worker seeds — the same operating
        # histories and the same MC seeds — and the dataset filled with
        # duplicates that then straddled the train/test split. `-n 1 -s S` is
        # unaffected: offset 0 reproduces exactly what it produced before.
        run_seed = None if args.seed is None else args.seed + i
        configs = create_worker_configs(base_config, NUM_WORKERS, master_seed=run_seed)

        start_time = time.perf_counter()
        run_parallel_simulations(generate_data, configs)
        elapsed = time.perf_counter() - start_time

        print(
            f"RUN: {i+1}/{DATA_GEN_RUNS}, All workers completed in {elapsed:.2f} seconds."
        )
