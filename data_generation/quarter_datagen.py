"""BEAVRS Cycle 1 depletion data generation — power schedule only, no boron.

The measured power history is interpolated onto a uniform grid and integrated
in a single `integrate()` call. Steps with exactly zero power are decay-only
inside OpenMC (no transport, no statepoint); their tally columns are recorded
as zeros. For every transport step, flux, per-nuclide fission rates and
(n,gamma) capture rates are extracted from the step's statepoint file.

Tally coverage
--------------
Capture (n,γ) tallies are scored for every isotope in the 7-isotope
breeding chain (U238, U239, Np239, Pu239, Pu240, Pu241, Pu242) so that the
Bateman matrix can be built directly from measured data. The consumer,
uncertainty_analysis.py, is NOT in this repository — see AUDIT.md P12. The
list lives in quarter_sim.CAPTURE_NUCLIDES — any changes there are picked up
automatically here (column names in the CSV follow the pattern
{nuclide}_capture and {nuclide}_capture_std).

Usage:
  python quarter_datagen.py -p power_history.csv -c 1 -t 40
"""

import argparse

# Parse argv and set OMP_NUM_THREADS BEFORE importing OpenMC, which reads the
# variable at import time. This ordering is why E402 is scoped off for this
# file in pyproject.toml.
parser = argparse.ArgumentParser(
    description="Generate depletion data using BEAVRS Cycle 1 power schedule (no boron)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# --- I/O ---
parser.add_argument(
    "-p",
    "--power-file",
    type=str,
    required=True,
    help="Path to BEAVRS power history CSV (Day, Percent Rated Power)",
)
parser.add_argument(
    "-f",
    "--chain-file",
    type=str,
    default="chain_casl_pwr.xml",
    help="Depletion chain file name",
)

# --- Parallelism ---
parser.add_argument(
    "-n",
    "--runs",
    type=int,
    default=1,
    help="Number of data generation runs (outer loop)",
)
parser.add_argument(
    "-c",
    "--cores",
    type=int,
    default=1,
    help="Number of parallel worker processes per run",
)
parser.add_argument(
    "-t", "--threads", type=int, default=1, help="OpenMP threads per worker"
)

# --- MC transport settings ---
parser.add_argument(
    "--particles", type=int, default=20_000, help="Neutron histories per batch"
)
parser.add_argument(
    "--inactive", type=int, default=20, help="Inactive batches for source convergence"
)
parser.add_argument(
    "--batches",
    type=int,
    default=80,
    help="Total batches (active = batches - inactive)",
)
parser.add_argument(
    "--temp-method",
    type=str,
    default="interpolation",
    choices=["interpolation", "nearest"],
    help="Cross section temperature treatment",
)

# --- Depletion settings ---
parser.add_argument(
    "--dt", type=float, default=1.0, help="Depletion time step size in days"
)

# --- BEAVRS nominal power ---
parser.add_argument(
    "--rated-power",
    type=float,
    default=41.7,
    help="100%% rated specific power [W/g] for converting percent to W/g",
)

# --- Fixed reactor state (no boron) ---
parser.add_argument(
    "--fuel-temp", type=float, default=900.0, help="Fuel temperature [K]"
)
parser.add_argument(
    "--mod-temp", type=float, default=580.0, help="Moderator temperature [K]"
)
parser.add_argument(
    "--clad-temp", type=float, default=620.0, help="Cladding temperature [K]"
)
parser.add_argument(
    "--mod-density", type=float, default=0.74, help="Moderator density [g/cm3]"
)
parser.add_argument(
    "--enrichment", type=float, default=3.1, help="U-235 enrichment [%%]"
)
parser.add_argument(
    "--fuel-density", type=float, default=10.4, help="UO2 fuel density [g/cm3]"
)

# --- Reproducibility ---
parser.add_argument(
    "-s", "--seed", type=int, default=None, help="Master random seed (None = random)"
)
args = parser.parse_args()

import os

os.environ["OMP_NUM_THREADS"] = str(args.threads)

import glob
import time
from datetime import datetime

import numpy as np
import openmc
import openmc.deplete

import tally_io
from common import (
    DAY_IN_SECONDS,
    create_worker_configs,
    parse_power_history,
    run_parallel_simulations,
    save_results,
    setup_paths,
)
from quarter_sim import (
    create_materials,
    set_material_volumes_quarter,
    create_quarter_geometry,
    create_settings,
    create_tallies,
    CAPTURE_NUCLIDES,
    FISSION_NUCLIDES,
)


# ---------------------------------------------------------------------------
# BEAVRS power history loading
# ---------------------------------------------------------------------------


def load_beavrs_power_history(filepath, nominal_specific_power_W_per_g, dt_days):
    """Load BEAVRS power history and interpolate onto a uniform grid."""
    days, percents = parse_power_history(filepath)
    powers_raw = percents / 100.0 * nominal_specific_power_W_per_g

    t_end = days[-1]
    num_steps = int(np.round(t_end / dt_days))
    t_uniform = np.arange(num_steps) * dt_days
    powers_interp = np.interp(t_uniform, days, powers_raw)

    print(f"Loaded BEAVRS Cycle 1 power history: {len(days)} raw points")
    print(
        f"  Interpolated to {num_steps} steps at dt = {dt_days:.5f} days "
        f"({dt_days*24:.1f} hours)"
    )
    print(f"  Duration: {t_end:.1f} days")
    print(f"  Power range: [{powers_interp.min():.2f}, {powers_interp.max():.2f}] W/g")

    # Exactly-zero steps run no transport inside OpenMC (decay only) — the same
    # criterion the tally extraction below uses.
    n_decay = int(np.sum(powers_interp == 0.0))
    print(
        f"  Decay-only steps (P = 0): {n_decay}/{num_steps} "
        f"({100*n_decay/num_steps:.1f}%)"
    )

    return list(powers_interp), t_uniform


# ---------------------------------------------------------------------------
# Depletion driver
# ---------------------------------------------------------------------------


def run_depletion_with_tallies(
    materials,
    geometry,
    settings,
    tallies,
    chain_file,
    powers,
    dt_seconds,
    fuel_mass_g,
    worker_id,
    results_dir,
):
    """Single-integrate() depletion with per-step tally extraction.

    powers: list of specific powers in W/g (one per timestep). Zero entries
    are passed straight through; OpenMC treats them as decay-only internally
    and writes no statepoint for them.
    """
    num_steps = len(powers)

    # --- Build one Model, one Operator, one Integrator, one integrate() ---
    model = openmc.model.Model(geometry, materials, settings, tallies)
    operator = openmc.deplete.CoupledOperator(model, chain_file)

    # Convert W/g -> W for each step. Zero stays zero.
    powers_W = [p * fuel_mass_g for p in powers]
    timesteps = [dt_seconds] * num_steps

    integrator = openmc.deplete.PredictorIntegrator(
        operator, timesteps, powers_W, timestep_units="s"
    )

    print(
        f"Worker {worker_id} | running integrate() once for "
        f"{num_steps} steps "
        f"(transport steps: {sum(1 for p in powers if p > 0)}, "
        f"decay-only: {sum(1 for p in powers if p == 0)})"
    )

    integrator.integrate()

    # --- Per-step tally extraction from statepoints ---
    # OpenMC writes openmc_simulation_n{step}.h5 for each step that actually
    # ran transport (i.e. source_rate > 0). Decay steps write no statepoint —
    # their tallies are recorded as zeros.
    step_data = {k: [] for k in tally_io.step_keys()}

    # Map step index -> statepoint path, tolerating either naming convention
    # (openmc_simulation_nN.h5 or statepoint.N.h5).
    sp_by_step = {}
    for pattern in ("openmc_simulation_n*.h5", "statepoint.*.h5"):
        for sp in glob.glob(os.path.join(results_dir, pattern)):
            base = os.path.basename(sp)
            try:
                num = int("".join(ch for ch in base.split(".")[0] if ch.isdigit()))
            except ValueError:
                continue
            sp_by_step.setdefault(num, sp)

    for i in range(num_steps):
        if powers[i] == 0.0:
            t = tally_io.zero_tally()
        else:
            sp = sp_by_step.get(i)
            t = tally_io.read_statepoint_tallies(sp) if sp else tally_io.nan_tally()

        fracs = tally_io.compute_fission_power_fractions(t)

        step_data["flux"].append(t["flux"])
        step_data["flux_std"].append(t["flux_std"])
        for nuc in FISSION_NUCLIDES:
            step_data[f"{nuc}_fission"].append(t[f"{nuc}_fission"])
            step_data[f"{nuc}_fission_std"].append(t[f"{nuc}_fission_std"])
            step_data[f"{nuc}_fission_power_frac"].append(
                fracs[f"{nuc}_fission_power_frac"]
            )
        for nuc in CAPTURE_NUCLIDES:
            step_data[f"{nuc}_capture"].append(t[f"{nuc}_capture"])
            step_data[f"{nuc}_capture_std"].append(t[f"{nuc}_capture_std"])

    return step_data


# ---------------------------------------------------------------------------
# Results extraction
# ---------------------------------------------------------------------------


def extract_results_data(results, powers, step_data, worker_id):
    """Combine depletion results with per-step tally data."""
    time_arr, k = results.get_keff()
    time_days = time_arr / DAY_IN_SECONDS

    # Timestamp plus the worker's unique tag, so labels stay distinct when
    # several workers start within the same second.
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"beavrs_{current_time}_{worker_id}"

    num_points = len(time_days)
    num_steps = len(powers)

    def pad(lst):
        return list(lst) + [float("nan")] * (num_points - num_steps)

    data = {
        "run_label": [label] * num_points,
        "time_days": time_days,
        "k_eff": k[:, 0],
        "k_eff_std": k[:, 1],
        "power_W_g": pad(powers),
    }

    for key, values in step_data.items():
        data[key] = pad(values)

    nuclides = results[0].index_nuc.keys()
    for nuclide in nuclides:
        _, concentration = results.get_atoms("1", nuclide, nuc_units="atom/b-cm")
        data[nuclide] = concentration

    return data, nuclides


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------


def setup_reactor_model(config, results_dir):
    """Build and export the quarter-pin reactor model."""
    fuel, gap, clad, water = create_materials(config)
    set_material_volumes_quarter(
        fuel,
        gap,
        clad,
        water,
        radii=config["geometry_radii"],
        pitch=config["geometry_pitch"],
    )

    materials = openmc.Materials([fuel, gap, clad, water])
    geometry = create_quarter_geometry(
        (fuel, gap, clad, water),
        radii=config["geometry_radii"],
        pitch=config["geometry_pitch"],
    )
    settings = create_settings(config)
    tallies = create_tallies(fuel)

    geometry.export_to_xml(path=results_dir)
    settings.export_to_xml(path=results_dir)
    materials.export_to_xml(path=results_dir)

    return fuel, gap, clad, water, materials, geometry, settings, tallies


def generate_data(config):
    """Single worker: build the model, deplete with tallies, save the CSV."""
    worker_id = config["worker_id"]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir, chain_file = setup_paths(script_dir, worker_id, config["chain_file"])

    print(
        f"Worker {worker_id} | seed={config['seed']} | "
        f"steps={len(config['powers'])} | "
        f"dt={config['dt_seconds']/DAY_IN_SECONDS:.5f} d | "
        f"particles={config['particles']}"
    )
    print(f"Worker {worker_id} | fission tallies: {FISSION_NUCLIDES}")
    print(f"Worker {worker_id} | capture tallies: {CAPTURE_NUCLIDES}")

    fuel, gap, clad, water, materials, geometry, settings, tallies = (
        setup_reactor_model(config, results_dir)
    )
    os.chdir(results_dir)

    fuel_mass_g = config["fuel_density"] * fuel.volume

    step_data = run_depletion_with_tallies(
        materials,
        geometry,
        settings,
        tallies,
        chain_file,
        config["powers"],
        config["dt_seconds"],
        fuel_mass_g,
        worker_id,
        results_dir,
    )

    results = openmc.deplete.Results("depletion_results.h5")
    data, nuclides = extract_results_data(
        results, config["powers"], step_data, worker_id
    )
    save_results(data, script_dir, worker_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load BEAVRS power history
    powers, time_days = load_beavrs_power_history(
        args.power_file,
        nominal_specific_power_W_per_g=args.rated_power,
        dt_days=args.dt,
    )
    dt_seconds = args.dt * DAY_IN_SECONDS

    base_config = {
        # --- Depletion ---
        "powers": powers,
        "dt_seconds": dt_seconds,
        # --- MC transport ---
        "particles": args.particles,
        "inactive": args.inactive,
        "batches": args.batches,
        "temp_method": args.temp_method,
        "chain_file": args.chain_file,
        # --- Fixed reactor state (no boron) ---
        "enrichment": args.enrichment,
        "fuel_density": args.fuel_density,
        "fuel_temp": args.fuel_temp,
        "mod_temp": args.mod_temp,
        "clad_temp": args.clad_temp,
        "mod_density": args.mod_density,
        # --- Geometry (BEAVRS pin cell) ---
        "geometry_radii": [0.39218, 0.40005, 0.45720],
        "geometry_pitch": 1.25984,
    }

    NUM_RUNS = args.runs
    NUM_WORKERS = args.cores

    for i in range(NUM_RUNS):
        # Each outer iteration needs its own master seed. Passing the same one
        # every time made create_worker_configs re-seed `random` identically, so
        # every iteration regenerated the same worker seeds — the same operating
        # histories and the same MC seeds — and the dataset filled with
        # duplicates that then straddled the train/test split. `-n 1 -s S` is
        # unaffected: offset 0 reproduces exactly what it produced before.
        run_seed = None if args.seed is None else args.seed + i
        configs = create_worker_configs(base_config, NUM_WORKERS, master_seed=run_seed)

        t0 = time.perf_counter()
        run_parallel_simulations(generate_data, configs)
        elapsed = time.perf_counter() - t0

        print(
            f"Run {i+1}/{NUM_RUNS} complete | {NUM_WORKERS} workers | "
            f"{elapsed:.1f}s"
        )
