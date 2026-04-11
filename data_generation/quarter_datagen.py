"""
BEAVRS Cycle 1 depletion data generation.

Generates high-fidelity depletion data using realistic BEAVRS Cycle 1 conditions:
  - Power history read from an external file (CSV or whitespace-delimited)
  - Fixed state parameters at nominal hot full power conditions
  - Quarter-pin geometry for computational efficiency
  - Parallel workers with independent MC seeds for statistical averaging

Usage:
  python generate_beavrs_data.py -p power_history.csv -n 4 -c 4
  python generate_beavrs_data.py -p power_history.csv --particles 20000 --dt 5

Power history file format (CSV with header):
  time_days,specific_power_W_per_g
  0,38.5
  5,38.3
  10,37.9
  ...

Or a simple single-column file (one power value per line, applied at each dt):
  38.5
  38.3
  37.9
  ...
"""
import sys
import argparse

parser = argparse.ArgumentParser(
  description='Generate depletion data using BEAVRS Cycle 1 conditions',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# --- I/O ---
parser.add_argument('-p', '--power-file', type=str, required=True,
          help='Path to power history file (CSV or single-column)')
parser.add_argument('-f', '--chain-file', type=str, default='chain_casl_pwr.xml',
          help='Depletion chain file name')

# --- Parallelism ---
parser.add_argument('-n', '--runs', type=int, default=1,
          help='Number of data generation runs (outer loop)')
parser.add_argument('-c', '--cores', type=int, default=10        ,
          help='Number of parallel worker processes per run')
parser.add_argument('-t', '--threads', type=int, default=1,
          help='OpenMP threads per worker')

# --- MC transport settings ---
parser.add_argument('--particles', type=int, default=20_000,
          help='Neutron histories per batch')
parser.add_argument('--inactive', type=int, default=20,
          help='Inactive batches for source convergence')
parser.add_argument('--batches', type=int, default=80,
          help='Total batches (active = batches - inactive)')
parser.add_argument('--temp-method', type=str, default='interpolation',
          choices=['interpolation', 'nearest'],
          help='Cross section temperature treatment')

# --- Depletion settings ---
parser.add_argument('--dt', type=float, default=10.0,
          help='Depletion time step size in days')

# --- Fixed reactor state (BEAVRS Cycle 1 nominal HFP) ---
parser.add_argument('--fuel-temp', type=float, default=900.0,
          help='Fuel temperature [K]')
parser.add_argument('--mod-temp', type=float, default=580.0,
          help='Moderator temperature [K]')
parser.add_argument('--clad-temp', type=float, default=620.0,
          help='Cladding temperature [K]')
parser.add_argument('--mod-density', type=float, default=0.74,
          help='Moderator density [g/cm3]')
parser.add_argument('--boron-ppm', type=float, default=975.0,
          help='Soluble boron concentration [ppm]')
parser.add_argument('--enrichment', type=float, default=3.1,
          help='U-235 enrichment [%]')
parser.add_argument('--fuel-density', type=float, default=10.4,
          help='UO2 fuel density [g/cm3]')

# --- Reproducibility ---
parser.add_argument('-s', '--seed', type=int, default=None,
          help='Master random seed (None = random)')


if __name__ == "__main__":
  args = parser.parse_args()
else:
  args = None  # Will not run as module

import os
os.environ["OMP_NUM_THREADS"] = str(args.threads if args else 1)

import openmc
import openmc.deplete
import time
import uuid
import numpy as np
import pandas as pd
import random
import multiprocessing as mp
from datetime import datetime

from quarter_sim import (
  create_materials,
  set_material_volumes_quarter,
  create_quarter_geometry,
  create_settings,
)

HOUR_IN_SECONDS = 3600
DAY_IN_SECONDS = 24 * HOUR_IN_SECONDS


# ---------------------------------------------------------------------------
# Power history loading
# ---------------------------------------------------------------------------

def load_power_history(filepath, dt_days=None):
  """Load a power history from file.
  
  Supports two formats:
  
  1) CSV with columns 'time_days' and 'specific_power_W_per_g':
     Returns the power values directly. If time steps in the file
     don't match dt_days, the history is interpolated onto a uniform
     grid with spacing dt_days.
  
  2) Single-column file (one power value per line, comments with #):
     Each value is the specific power [W/g] for one depletion step
     of duration dt_days.
  
  Returns:
    powers: list of specific power values [W/g], one per depletion step
    dt:   time step size in seconds
  """
  filepath = os.path.abspath(filepath)
  if not os.path.exists(filepath):
    raise FileNotFoundError(f"Power history file not found: {filepath}")

  # Try CSV with header first
  try:
    df = pd.read_csv(filepath, comment='#')
    cols_lower = {c.strip().lower(): c for c in df.columns}

    if 'time_days' in cols_lower and 'specific_power_w_per_g' in cols_lower:
      times = df[cols_lower['time_days']].values
      powers_raw = df[cols_lower['specific_power_w_per_g']].values

      if dt_days is not None:
        # Interpolate onto uniform grid
        t_end = times[-1]
        num_steps = int(np.round(t_end / dt_days))
        t_uniform = np.linspace(0, t_end, num_steps + 1)
        # Power at each step is the average over the interval;
        # approximate by the value at the start of each interval
        powers_interp = np.interp(t_uniform[:-1], times, powers_raw)
        return list(powers_interp), dt_days * DAY_IN_SECONDS

      else:
        # Use the time steps as given in the file
        dt_values = np.diff(times)
        if not np.allclose(dt_values, dt_values[0], rtol=1e-3):
          raise ValueError(
            "Non-uniform time steps detected in power file. "
            "Specify --dt to interpolate onto a uniform grid."
          )
        return list(powers_raw[:-1]), dt_values[0] * DAY_IN_SECONDS

  except (pd.errors.ParserError, KeyError, ValueError):
    pass

  # Fallback: single-column file
  powers = []
  with open(filepath) as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith('#'):
        continue
      powers.append(float(line))

  if dt_days is None:
    raise ValueError(
      "Single-column power file requires --dt to set the step size."
    )
  return powers, dt_days * DAY_IN_SECONDS


# ---------------------------------------------------------------------------
# Path and model setup
# ---------------------------------------------------------------------------

def setup_paths(script_dir, worker_id, chain_filename):
  """Create results directory and load depletion chain."""
  results_dir = os.path.abspath(
    os.path.join(script_dir, "results", f"worker_{worker_id}")
  )
  os.makedirs(results_dir, exist_ok=True)

  openmc.config['cross_sections'] = os.path.join(
    script_dir, "../data/cross_sections.xml"
  )
  os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
  os.environ['OPENMC_CROSS_SECTIONS'] = str(openmc.config['cross_sections'])

  chain_file = os.path.join(script_dir, "../data", chain_filename)
  chain = openmc.deplete.Chain.from_xml(chain_file)

  return results_dir, chain


def setup_reactor_model(config, results_dir):
    """Build and export the quarter-pin reactor model."""
    fuel, gap, clad, water = create_materials(config)
    set_material_volumes_quarter(
        fuel, gap, clad, water,
        radii=config['geometry_radii'],
        pitch=config['geometry_pitch']
    )

    materials = openmc.Materials([fuel, gap, clad, water])
    geometry = create_quarter_geometry(
        (fuel, gap, clad, water),
        radii=config['geometry_radii'],
        pitch=config['geometry_pitch']
    )
    settings = create_settings(config)

    geometry.export_to_xml(path=results_dir)
    settings.export_to_xml(path=results_dir)
    materials.export_to_xml(path=results_dir)

    return fuel, gap, clad, water, materials, geometry, settings



# ---------------------------------------------------------------------------
# Depletion driver
# ---------------------------------------------------------------------------

def run_depletion_step(model, chain, time_step_s, power_watts, prev_results_file=None):
  """Run a single coupled depletion step."""
  if prev_results_file and os.path.exists(prev_results_file):
    prev_results = openmc.deplete.Results(prev_results_file)
    operator = openmc.deplete.CoupledOperator(model, chain, prev_results=prev_results)
  else:
    operator = openmc.deplete.CoupledOperator(model, chain)

  integrator = openmc.deplete.PredictorIntegrator(
    operator, [time_step_s], [power_watts], timestep_units='s'
  )
  integrator.integrate()


def run_depletion_simulation(fuel, gap, clad, water, materials, geometry, settings,
                             chain, powers, dt_seconds, fuel_mass_g,
                             worker_id, results_dir):
    """Step through the power history, running coupled transport + depletion."""
    num_steps = len(powers)

    for i in range(num_steps):
        step_power_W = powers[i] * fuel_mass_g  # specific power -> total power
        print(f"Worker {worker_id} | Step {i+1}/{num_steps} | "
              f"P = {powers[i]:.2f} W/g | P_total = {step_power_W:.1f} W")

        # Materials are fixed (no temperature/density updates per step)
        # If you later want step-dependent boron letdown, update water here.
        materials.export_to_xml(path=results_dir)

        model = openmc.model.Model(geometry, materials, settings)
        prev_results_file = "depletion_results.h5" if i > 0 else None

        run_depletion_step(model, chain, dt_seconds, step_power_W, prev_results_file)


# ---------------------------------------------------------------------------
# Results extraction
# ---------------------------------------------------------------------------

def extract_results_data(results, powers, dt_seconds):
  """Extract time series of keff and nuclide concentrations."""
  time_arr, k = results.get_keff()
  time_days = time_arr / DAY_IN_SECONDS

  current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
  label = f"beavrs_{current_time}"

  num_points = len(time_days)
  num_steps  = len(powers)

  # Pad powers to match num_points (results have n_steps+1 entries)
  powers_padded = list(powers) + [float('nan')] * (num_points - num_steps)

  data = {
    'run_label':  [label] * num_points,
    'time_days':  time_days,
    'k_eff':    k[:, 0],
    'k_eff_std':  k[:, 1],
    'power_W_g':  powers_padded,
  }

  nuclides = results[0].index_nuc.keys()
  for nuclide in nuclides:
    _, concentration = results.get_atoms("1", nuclide, nuc_units="atom/b-cm")
    data[nuclide] = concentration

  return data, nuclides


def save_results(data, script_dir, worker_id):
    """Append results to worker-specific CSV."""
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    df = pd.DataFrame(data)
    file_path = os.path.join(data_dir, f"worker_{worker_id}_nuclide_concentrations.csv")
    file_exists = os.path.isfile(file_path)
    df.to_csv(file_path, mode='a', index=False, header=not file_exists)
    print(f"Worker {worker_id} | Results saved to {file_path}")


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------

def generate_data(config):
    """Single worker: build model, run depletion, save results."""
    worker_id  = config['worker_id']
    script_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir, chain = setup_paths(script_dir, worker_id, config['chain_file'])

    print(f"Worker {worker_id} | seed={config['seed']} | "
          f"steps={len(config['powers'])} | dt={config['dt_seconds']/DAY_IN_SECONDS:.1f} d | "
          f"particles={config['particles']}")

    np.random.seed(config['seed'])

    fuel, gap, clad, water, materials, geometry, settings = setup_reactor_model(config, results_dir)
    os.chdir(results_dir)

    fuel_mass_g = config['fuel_density'] * fuel.volume

    run_depletion_simulation(
        fuel, gap, clad, water, materials, geometry, settings, chain,
        config['powers'], config['dt_seconds'], fuel_mass_g,
        worker_id, results_dir
    )

    results = openmc.deplete.Results("depletion_results.h5")
    data, nuclides = extract_results_data(results, config['powers'], config['dt_seconds'])
    save_results(data, script_dir, worker_id)


# ---------------------------------------------------------------------------
# Parallel orchestration
# ---------------------------------------------------------------------------

def create_worker_configs(base_config, num_workers, master_seed=None):
    """Create per-worker configs with unique IDs and MC seeds."""
    if master_seed is not None:
        random.seed(master_seed)
        print(f"Master seed: {master_seed}")
    else:
        random.seed()
        print("No master seed (random)")

    configs = []
    for i in range(1, num_workers + 1):
        config = base_config.copy()
        config['worker_id'] = f"{i}_{uuid.uuid4().hex[:8]}"
        config['seed'] = random.randint(1, 2**31 - 1)
        configs.append(config)
    return configs


def run_parallel_simulations(configs):
    """Launch workers as separate processes."""
    processes = []
    try:
        for config in configs:
            p = mp.Process(target=generate_data, args=(config,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    except Exception as e:
        print(f"Error during parallel generation: {e}")
        for p in processes:
            if p.is_alive():
                p.terminate()
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Load power history
    powers, dt_seconds = load_power_history(args.power_file, dt_days=args.dt)
    print(f"Loaded {len(powers)} power steps from {args.power_file}")
    print(f"  dt = {dt_seconds / DAY_IN_SECONDS:.2f} days")
    print(f"  Total irradiation = {len(powers) * dt_seconds / DAY_IN_SECONDS:.1f} days")
    print(f"  Power range: [{min(powers):.2f}, {max(powers):.2f}] W/g")

    base_config = {
        # --- Depletion ---
        'powers':      powers,
        'dt_seconds':  dt_seconds,

        # --- MC transport ---
        'particles':   args.particles,
        'inactive':    args.inactive,
        'batches':     args.batches,
        'temp_method': args.temp_method,
        'chain_file':  args.chain_file,

        # --- Fixed BEAVRS state ---
        'enrichment':   args.enrichment,
        'fuel_density': args.fuel_density,
        'fuel_temp':    args.fuel_temp,
        'mod_temp':     args.mod_temp,
        'clad_temp':    args.clad_temp,
        'mod_density':  args.mod_density,
        'boron_ppm':    args.boron_ppm,

        # --- Geometry (BEAVRS pin cell: fuel_or, gap_or, clad_or) ---
        'geometry_radii': [0.39218, 0.40005, 0.45720],
        'geometry_pitch': 1.25984,
    }

    NUM_RUNS    = args.runs
    NUM_WORKERS = args.cores

    for i in range(NUM_RUNS):
        configs = create_worker_configs(base_config, NUM_WORKERS, master_seed=args.seed)

        t0 = time.perf_counter()
        run_parallel_simulations(configs)
        elapsed = time.perf_counter() - t0

        print(f"Run {i+1}/{NUM_RUNS} complete | {NUM_WORKERS} workers | {elapsed:.1f}s")