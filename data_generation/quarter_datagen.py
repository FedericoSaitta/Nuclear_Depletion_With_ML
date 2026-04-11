"""
BEAVRS Cycle 1 depletion data generation with critical boron search
and detailed tally extraction (flux, reaction rates, fission power fractions).

Usage:
  python quarter_datagen.py -p power_history.csv -c 1 -t 10
  python quarter_datagen.py -p power_history.csv --particles 50000 --dt 1 -c 1 -t 32
"""
import sys
import argparse

parser = argparse.ArgumentParser(
  description='Generate depletion data using BEAVRS Cycle 1 conditions',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# --- I/O ---
parser.add_argument('-p', '--power-file', type=str, required=True,
          help='Path to BEAVRS power history CSV (Day, Percent Rated Power)')
parser.add_argument('-f', '--chain-file', type=str, default='chain_casl_pwr.xml',
          help='Depletion chain file name')

# --- Parallelism ---
parser.add_argument('-n', '--runs', type=int, default=1,
          help='Number of data generation runs (outer loop)')
parser.add_argument('-c', '--cores', type=int, default=1,
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
parser.add_argument('--dt', type=float, default=1.0,
          help='Depletion time step size in days')

# --- BEAVRS nominal power ---
parser.add_argument('--rated-power', type=float, default=41.7,
          help='100%% rated specific power [W/g] for converting percent to W/g')

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
          help='Initial boron for materials setup [ppm]')
parser.add_argument('--boron-bracket', type=float, nargs=2, default=[0.0, 6000.0],
          help='Bracket range for boron search [ppm]')
parser.add_argument('--boron-tol', type=float, default=1e-2,
          help='keff tolerance for boron search (1e-2 = 1000 pcm)')
parser.add_argument('--enrichment', type=float, default=3.1,
          help='U-235 enrichment [%%]')
parser.add_argument('--fuel-density', type=float, default=10.4,
          help='UO2 fuel density [g/cm3]')

# --- Reproducibility ---
parser.add_argument('-s', '--seed', type=int, default=None,
          help='Master random seed (None = random)')


if __name__ == "__main__":
  args = parser.parse_args()
else:
  args = None

import os
os.environ["OMP_NUM_THREADS"] = str(args.threads if args else 1)

import openmc
import openmc.deplete
import time
import uuid
import glob
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
  create_tallies,
  update_water_composition,
  FISSION_NUCLIDES,
  CAPTURE_NUCLIDES,
  FISSION_Q_VALUES,
)

HOUR_IN_SECONDS = 3600
DAY_IN_SECONDS = 24 * HOUR_IN_SECONDS


# ---------------------------------------------------------------------------
# BEAVRS power history loading
# ---------------------------------------------------------------------------

def load_beavrs_power_history(filepath, nominal_specific_power_W_per_g, dt_days):
  """Load BEAVRS power history and interpolate onto uniform grid.

  Expected CSV format:
    Cycle 1
    Day,Percent Rated Power
    0.0,1.598205
    ...

  Only Cycle 1 data is used.
  """
  filepath = os.path.abspath(filepath)
  if not os.path.exists(filepath):
    raise FileNotFoundError(f"Power history file not found: {filepath}")

  days = []
  percents = []
  header_found = False

  with open(filepath) as f:
    for line in f:
      line = line.strip()
      if not line:
        if header_found:
          break
        continue
      if line.lower().startswith('cycle 1'):
        continue
      if line.lower().startswith('day'):
        header_found = True
        continue
      if line.lower().startswith('cycle') or line[0].isalpha():
        break
      try:
        parts = line.split(',')
        days.append(float(parts[0]))
        percents.append(float(parts[1]))
      except (ValueError, IndexError):
        break

  days = np.array(days)
  powers_raw = np.array(percents) / 100.0 * nominal_specific_power_W_per_g

  t_end = days[-1]
  num_steps = int(np.round(t_end / dt_days))
  t_uniform = np.arange(num_steps) * dt_days
  powers_interp = np.interp(t_uniform, days, powers_raw)

  print(f"Loaded BEAVRS Cycle 1 power history: {len(days)} raw points")
  print(f"  Interpolated to {num_steps} steps at dt = {dt_days:.2f} days")
  print(f"  Duration: {t_end:.1f} days")
  print(f"  Power range: [{powers_interp.min():.2f}, {powers_interp.max():.2f}] W/g")

  return list(powers_interp), t_uniform


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
  tallies = create_tallies(fuel)

  geometry.export_to_xml(path=results_dir)
  settings.export_to_xml(path=results_dir)
  materials.export_to_xml(path=results_dir)

  return fuel, gap, clad, water, materials, geometry, settings, tallies


# ---------------------------------------------------------------------------
# Critical boron search
# ---------------------------------------------------------------------------

def find_critical_boron(water, materials, geometry, settings, results_dir,
            bracket, tol, prev_crit_boron=None):
  """Find the boron concentration that gives keff = 1.0.

  Note: the boron search models do NOT include custom tallies — only the
  depletion transport solve gets tallies. This avoids unnecessary I/O
  during the search iterations.
  """
  mod_density = water.density

  def build_model(boron_ppm):
    update_water_composition(water, boron_ppm, mod_density)
    materials.export_to_xml(path=results_dir)
    # No tallies here — search models are lightweight
    return openmc.model.Model(geometry, materials, settings)

  if prev_crit_boron is not None:
    margin = 200.0
    lo = max(bracket[0], prev_crit_boron - margin)
    hi = min(bracket[1], prev_crit_boron + margin)
    search_bracket = [lo, hi]
  else:
    search_bracket = list(bracket)

  crit_ppm, guesses, keffs = openmc.search_for_keff(
    build_model,
    bracket=search_bracket,
    tol=tol,
    bracketed_method='bisect',
    print_iterations=True,
  )

  update_water_composition(water, crit_ppm, mod_density)
  return crit_ppm, guesses, keffs


# ---------------------------------------------------------------------------
# Statepoint tally extraction
# ---------------------------------------------------------------------------

def extract_tallies_from_statepoint(batches):
  """Read the latest statepoint and extract flux + reaction rate tallies.

  Returns a dict with tally data, or None values if statepoint not found.
  """
  sp_file = f"statepoint.{batches}.h5"

  result = {
    'flux': float('nan'),
    'flux_std': float('nan'),
  }
  for nuc in FISSION_NUCLIDES:
    result[f'{nuc}_fission'] = float('nan')
    result[f'{nuc}_fission_std'] = float('nan')
  for nuc in CAPTURE_NUCLIDES:
    result[f'{nuc}_capture'] = float('nan')
    result[f'{nuc}_capture_std'] = float('nan')

  if not os.path.exists(sp_file):
    # Try glob as fallback
    sp_files = sorted(glob.glob('statepoint.*.h5'))
    if sp_files:
      sp_file = sp_files[-1]
    else:
      print("  WARNING: No statepoint file found, tallies will be NaN")
      return result

  try:
    sp = openmc.StatePoint(sp_file)

    # Flux
    try:
      t = sp.get_tally(id=9001)
      result['flux'] = t.mean.flatten()[0]
      result['flux_std'] = t.std_dev.flatten()[0]
    except Exception as e:
      print(f"  WARNING: Could not read flux tally: {e}")

    # Fission rates
    try:
      t = sp.get_tally(id=9002)
      means = t.mean.flatten()
      stds  = t.std_dev.flatten()
      for j, nuc in enumerate(FISSION_NUCLIDES):
        result[f'{nuc}_fission'] = means[j]
        result[f'{nuc}_fission_std'] = stds[j]
    except Exception as e:
      print(f"  WARNING: Could not read fission tally: {e}")

    # Capture rates
    try:
      t = sp.get_tally(id=9003)
      means = t.mean.flatten()
      stds  = t.std_dev.flatten()
      for j, nuc in enumerate(CAPTURE_NUCLIDES):
        result[f'{nuc}_capture'] = means[j]
        result[f'{nuc}_capture_std'] = stds[j]
    except Exception as e:
      print(f"  WARNING: Could not read capture tally: {e}")

    sp.close()

  except Exception as e:
    print(f"  WARNING: Could not open statepoint {sp_file}: {e}")

  return result


def compute_fission_power_fractions(tally_data):
  """Compute fraction of fission power from each nuclide.

  Power_i = fission_rate_i * Q_i
  Fraction_i = Power_i / sum(Power_j)
  """
  fracs = {}
  total_power = 0.0
  for nuc in FISSION_NUCLIDES:
    rate = tally_data.get(f'{nuc}_fission', 0.0)
    if np.isnan(rate):
      rate = 0.0
    total_power += rate * FISSION_Q_VALUES[nuc]

  for nuc in FISSION_NUCLIDES:
    rate = tally_data.get(f'{nuc}_fission', 0.0)
    if np.isnan(rate):
      rate = 0.0
    if total_power > 0:
      fracs[f'{nuc}_fission_power_frac'] = (rate * FISSION_Q_VALUES[nuc]) / total_power
    else:
      fracs[f'{nuc}_fission_power_frac'] = 0.0

  return fracs


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
               tallies, chain, powers, dt_seconds, fuel_mass_g,
               worker_id, results_dir, boron_bracket, boron_tol):
  """Step through the power history with critical boron search and tally extraction.

  At each depletion step:
    1. Find critical boron via search_for_keff
    2. Run coupled transport + depletion (with tallies)
    3. Read statepoint to extract flux and reaction rates
    4. Compute fission power fractions

  Returns:
    step_data: dict of lists, one entry per depletion step
  """
  num_steps = len(powers)
  batches = settings.batches

  # Initialise per-step storage
  step_keys = ['boron_ppm', 'flux', 'flux_std']
  for nuc in FISSION_NUCLIDES:
    step_keys.extend([f'{nuc}_fission', f'{nuc}_fission_std', f'{nuc}_fission_power_frac'])
  for nuc in CAPTURE_NUCLIDES:
    step_keys.extend([f'{nuc}_capture', f'{nuc}_capture_std'])

  step_data = {k: [] for k in step_keys}
  prev_crit_boron = None

  for i in range(num_steps):
    step_power_W = powers[i] * fuel_mass_g

    # --- 1. Critical boron search ---
    print(f"\nWorker {worker_id} | Step {i+1}/{num_steps} | "
        f"P = {powers[i]:.2f} W/g | Searching for critical boron...")

    crit_ppm, guesses, keffs = find_critical_boron(
      water, materials, geometry, settings, results_dir,
      bracket=boron_bracket, tol=boron_tol,
      prev_crit_boron=prev_crit_boron
    )
    step_data['boron_ppm'].append(crit_ppm)
    prev_crit_boron = crit_ppm

    print(f"Worker {worker_id} | Step {i+1}/{num_steps} | "
        f"P = {powers[i]:.2f} W/g | B_crit = {crit_ppm:.1f} ppm | "
        f"({len(guesses)} search iters)")

    # --- 2. Depletion with tallies ---
    materials.export_to_xml(path=results_dir)
    model = openmc.model.Model(geometry, materials, settings, tallies)
    prev_results_file = "depletion_results.h5" if i > 0 else None

    run_depletion_step(model, chain, dt_seconds, step_power_W, prev_results_file)

    # --- 3. Extract tallies from statepoint ---
    tally_data = extract_tallies_from_statepoint(batches)

    step_data['flux'].append(tally_data['flux'])
    step_data['flux_std'].append(tally_data['flux_std'])

    for nuc in FISSION_NUCLIDES:
      step_data[f'{nuc}_fission'].append(tally_data[f'{nuc}_fission'])
      step_data[f'{nuc}_fission_std'].append(tally_data[f'{nuc}_fission_std'])

    for nuc in CAPTURE_NUCLIDES:
      step_data[f'{nuc}_capture'].append(tally_data[f'{nuc}_capture'])
      step_data[f'{nuc}_capture_std'].append(tally_data[f'{nuc}_capture_std'])

    # --- 4. Fission power fractions ---
    fracs = compute_fission_power_fractions(tally_data)
    for nuc in FISSION_NUCLIDES:
      step_data[f'{nuc}_fission_power_frac'].append(fracs[f'{nuc}_fission_power_frac'])

    # Log key results for this step
    u235_frac = fracs.get('U235_fission_power_frac', 0)
    pu239_frac = fracs.get('Pu239_fission_power_frac', 0)
    print(f"  flux = {tally_data['flux']:.4e} +/- {tally_data['flux_std']:.4e} | "
        f"U235 power frac = {u235_frac:.3f} | Pu239 power frac = {pu239_frac:.3f}")

  return step_data


# ---------------------------------------------------------------------------
# Results extraction
# ---------------------------------------------------------------------------

def extract_results_data(results, powers, dt_seconds, step_data):
  """Combine depletion results with per-step tally data into a single dict."""
  time_arr, k = results.get_keff()
  time_days = time_arr / DAY_IN_SECONDS

  current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
  label = f"beavrs_{current_time}"

  num_points = len(time_days)
  num_steps  = len(powers)

  def pad(lst):
    """Pad step-level data (n values) to match results length (n+1)."""
    return list(lst) + [float('nan')] * (num_points - num_steps)

  data = {
    'run_label':   [label] * num_points,
    'time_days':   time_days,
    'k_eff':       k[:, 0],
    'k_eff_std':   k[:, 1],
    'power_W_g':   pad(powers),
  }

  # Add all per-step data (boron, flux, reaction rates, fission fracs)
  for key, values in step_data.items():
    data[key] = pad(values)

  # Add nuclide concentrations
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
  """Single worker: build model, run depletion with boron search + tallies."""
  worker_id  = config['worker_id']
  script_dir = os.path.dirname(os.path.abspath(__file__))

  results_dir, chain = setup_paths(script_dir, worker_id, config['chain_file'])

  print(f"Worker {worker_id} | seed={config['seed']} | "
      f"steps={len(config['powers'])} | "
      f"dt={config['dt_seconds']/DAY_IN_SECONDS:.2f} d | "
      f"particles={config['particles']} | "
      f"boron bracket={config['boron_bracket']}")

  np.random.seed(config['seed'])

  fuel, gap, clad, water, materials, geometry, settings, tallies = \
    setup_reactor_model(config, results_dir)
  os.chdir(results_dir)

  fuel_mass_g = config['fuel_density'] * fuel.volume

  step_data = run_depletion_simulation(
    fuel, gap, clad, water, materials, geometry, settings, tallies,
    chain, config['powers'], config['dt_seconds'], fuel_mass_g,
    worker_id, results_dir,
    boron_bracket=config['boron_bracket'],
    boron_tol=config['boron_tol']
  )

  results = openmc.deplete.Results("depletion_results.h5")
  data, nuclides = extract_results_data(
    results, config['powers'], config['dt_seconds'], step_data
  )
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

  # Load BEAVRS power history
  powers, time_days = load_beavrs_power_history(
    args.power_file,
    nominal_specific_power_W_per_g=args.rated_power,
    dt_days=args.dt
  )
  dt_seconds = args.dt * DAY_IN_SECONDS

  base_config = {
    # --- Depletion ---
    'powers':        powers,
    'dt_seconds':    dt_seconds,

    # --- MC transport ---
    'particles':     args.particles,
    'inactive':      args.inactive,
    'batches':       args.batches,
    'temp_method':   args.temp_method,
    'chain_file':    args.chain_file,

    # --- Critical boron search ---
    'boron_bracket': args.boron_bracket,
    'boron_tol':     args.boron_tol,

    # --- Fixed BEAVRS state ---
    'enrichment':    args.enrichment,
    'fuel_density':  args.fuel_density,
    'fuel_temp':     args.fuel_temp,
    'mod_temp':      args.mod_temp,
    'clad_temp':     args.clad_temp,
    'mod_density':   args.mod_density,
    'boron_ppm':     args.boron_ppm,

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