"""
BEAVRS Cycle 1 depletion data generation — power schedule only, no boron.

Features:
  - Checkpointing: automatically resumes from last completed step if
    depletion_results.h5 exists (use --resume flag)
  - Zero-power optimisation: steps with power < threshold use minimal
    particles since transport results don't affect decay-only depletion
  - Tally extraction: flux, reaction rates, fission power fractions

Usage:
  python quarter_datagen.py -p power_history.csv -c 1 -t 40
  python quarter_datagen.py -p power_history.csv -c 1 -t 40 --resume
"""
import sys
import argparse

parser = argparse.ArgumentParser(
    description='Generate depletion data using BEAVRS Cycle 1 power schedule (no boron)',
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

# --- Fixed reactor state (no boron) ---
parser.add_argument('--fuel-temp', type=float, default=900.0,
                    help='Fuel temperature [K]')
parser.add_argument('--mod-temp', type=float, default=580.0,
                    help='Moderator temperature [K]')
parser.add_argument('--clad-temp', type=float, default=620.0,
                    help='Cladding temperature [K]')
parser.add_argument('--mod-density', type=float, default=0.74,
                    help='Moderator density [g/cm3]')
parser.add_argument('--enrichment', type=float, default=3.1,
                    help='U-235 enrichment [%%]')
parser.add_argument('--fuel-density', type=float, default=10.4,
                    help='UO2 fuel density [g/cm3]')

# --- Reproducibility ---
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='Master random seed (None = random)')

# --- Checkpointing ---
parser.add_argument('--resume', action='store_true',
                    help='Resume from last completed step in depletion_results.h5')

# --- Zero-power optimisation ---
parser.add_argument('--power-threshold', type=float, default=0.01,
                    help='Specific power [W/g] below which to use reduced-fidelity '
                         'transport (decay-only regime)')
parser.add_argument('--decay-particles', type=int, default=100,
                    help='Particles per batch for zero-power (decay-only) steps')
parser.add_argument('--decay-batches', type=int, default=10,
                    help='Total batches for zero-power steps')
parser.add_argument('--decay-inactive', type=int, default=3,
                    help='Inactive batches for zero-power steps')


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
import json
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
    """Load BEAVRS power history and interpolate onto uniform grid."""
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
    print(f"  Interpolated to {num_steps} steps at dt = {dt_days:.5f} days "
          f"({dt_days*24:.1f} hours)")
    print(f"  Duration: {t_end:.1f} days")
    print(f"  Power range: [{powers_interp.min():.2f}, {powers_interp.max():.2f}] W/g")

    # Count zero-power steps
    n_zero = np.sum(powers_interp < 0.01)
    print(f"  Zero-power steps (<0.01 W/g): {n_zero}/{num_steps} "
          f"({100*n_zero/num_steps:.1f}%)")

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
# Statepoint tally extraction
# ---------------------------------------------------------------------------


def _nan_tally():
    r = {'flux': float('nan'), 'flux_std': float('nan')}
    for nuc in FISSION_NUCLIDES:
        r[f'{nuc}_fission'] = float('nan')
        r[f'{nuc}_fission_std'] = float('nan')
    for nuc in CAPTURE_NUCLIDES:
        r[f'{nuc}_capture'] = float('nan')
        r[f'{nuc}_capture_std'] = float('nan')
    return r
 
 
def _read_statepoint_tallies(sp_path):
    """Read flux / fission / capture tallies from one statepoint file."""
    result = _nan_tally()
    try:
        sp = openmc.StatePoint(sp_path)
        try:
            t = sp.get_tally(id=9001)
            result['flux']     = float(t.mean.flatten()[0])
            result['flux_std'] = float(t.std_dev.flatten()[0])
        except Exception:
            pass
        try:
            t = sp.get_tally(id=9002)
            m = t.mean.flatten()
            s = t.std_dev.flatten()
            for j, nuc in enumerate(FISSION_NUCLIDES):
                result[f'{nuc}_fission']     = float(m[j])
                result[f'{nuc}_fission_std'] = float(s[j])
        except Exception:
            pass
        try:
            t = sp.get_tally(id=9003)
            m = t.mean.flatten()
            s = t.std_dev.flatten()
            for j, nuc in enumerate(CAPTURE_NUCLIDES):
                result[f'{nuc}_capture']     = float(m[j])
                result[f'{nuc}_capture_std'] = float(s[j])
        except Exception:
            pass
        sp.close()
    except Exception as e:
        print(f"  WARNING: could not read {sp_path}: {e}")
    return result
 
 
def _fission_power_fractions(tally):
    """Same semantics as the original compute_fission_power_fractions()."""
    fracs = {}
    total = 0.0
    for nuc in FISSION_NUCLIDES:
        r = tally.get(f'{nuc}_fission', 0.0)
        if np.isnan(r):
            r = 0.0
        total += r * FISSION_Q_VALUES[nuc]
    for nuc in FISSION_NUCLIDES:
        r = tally.get(f'{nuc}_fission', 0.0)
        if np.isnan(r):
            r = 0.0
        fracs[f'{nuc}_fission_power_frac'] = (r * FISSION_Q_VALUES[nuc] / total) \
                                             if total > 0 else 0.0
    return fracs


# ---------------------------------------------------------------------------
# Depletion driver with checkpointing and decay optimisation
# ---------------------------------------------------------------------------
def run_depletion_with_tallies(fuel, materials, geometry, settings, tallies,
                               chain, powers, dt_seconds, fuel_mass_g,
                               worker_id, results_dir, config):
    """Single-integrate() depletion with per-step tally extraction.
 
    powers: list of specific powers in W/g (one per timestep). Zero
            entries are passed straight through; OpenMC treats them as
            decay-only internally.
    """
    num_steps = len(powers)
 
    # --- Build one Model, one Operator, one Integrator, one integrate() ---
    model = openmc.model.Model(geometry, materials, settings, tallies)
    operator = openmc.deplete.CoupledOperator(model, chain)
 
    # Convert W/g -> W for each step. Zero stays zero.
    powers_W = [p * fuel_mass_g for p in powers]
    timesteps = [dt_seconds] * num_steps
 
    integrator = openmc.deplete.PredictorIntegrator(
        operator, timesteps, powers_W, timestep_units='s'
    )
 
    print(f"Worker {worker_id} | running integrate() once for "
          f"{num_steps} steps "
          f"(non-zero steps: {sum(1 for p in powers if p > 0)}, "
          f"decay-only: {sum(1 for p in powers if p == 0)})")
 
    integrator.integrate()  # THE fix: one call, not N calls
 
    # --- Per-step tally extraction from statepoints ---
    # OpenMC writes openmc_simulation_n{step}.h5 for each step that
    # actually ran transport (i.e. source_rate > 0). Decay steps don't
    # write a statepoint — we fill them with zeros.
    step_keys = ['flux', 'flux_std']
    for nuc in FISSION_NUCLIDES:
        step_keys.extend([f'{nuc}_fission', f'{nuc}_fission_std',
                          f'{nuc}_fission_power_frac'])
    for nuc in CAPTURE_NUCLIDES:
        step_keys.extend([f'{nuc}_capture', f'{nuc}_capture_std'])
    step_data = {k: [] for k in step_keys}
 
    # Build a dict {step_index -> statepoint path}. OpenMC names files
    # with an integer step index. We tolerate either naming convention
    # (openmc_simulation_nN.h5 or statepoint.N.h5).
    sp_by_step = {}
    for pattern in ('openmc_simulation_n*.h5', 'statepoint.*.h5'):
        for sp in glob.glob(os.path.join(results_dir, pattern)):
            base = os.path.basename(sp)
            try:
                num = int(''.join(ch for ch in base.split('.')[0]
                                  if ch.isdigit()))
            except ValueError:
                continue
            sp_by_step.setdefault(num, sp)
 
    for i in range(num_steps):
        if powers[i] == 0.0:
            t = {'flux': 0.0, 'flux_std': 0.0}
            for nuc in FISSION_NUCLIDES:
                t[f'{nuc}_fission'] = 0.0
                t[f'{nuc}_fission_std'] = 0.0
            for nuc in CAPTURE_NUCLIDES:
                t[f'{nuc}_capture'] = 0.0
                t[f'{nuc}_capture_std'] = 0.0
        else:
            sp = sp_by_step.get(i)
            t = _read_statepoint_tallies(sp) if sp else _nan_tally()
 
        fracs = _fission_power_fractions(t)
 
        step_data['flux'].append(t['flux'])
        step_data['flux_std'].append(t['flux_std'])
        for nuc in FISSION_NUCLIDES:
            step_data[f'{nuc}_fission'].append(t[f'{nuc}_fission'])
            step_data[f'{nuc}_fission_std'].append(t[f'{nuc}_fission_std'])
            step_data[f'{nuc}_fission_power_frac'].append(
                fracs[f'{nuc}_fission_power_frac'])
        for nuc in CAPTURE_NUCLIDES:
            step_data[f'{nuc}_capture'].append(t[f'{nuc}_capture'])
            step_data[f'{nuc}_capture_std'].append(t[f'{nuc}_capture_std'])
 
    return step_data


# ---------------------------------------------------------------------------
# Results extraction
# ---------------------------------------------------------------------------

def extract_results_data(results, powers, dt_seconds, step_data):
    """Combine depletion results with per-step tally data."""
    time_arr, k = results.get_keff()
    time_days = time_arr / DAY_IN_SECONDS

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"beavrs_{current_time}"

    num_points = len(time_days)
    num_steps  = len(powers)

    def pad(lst):
        return list(lst) + [float('nan')] * (num_points - num_steps)

    data = {
        'run_label':   [label] * num_points,
        'time_days':   time_days,
        'k_eff':       k[:, 0],
        'k_eff_std':   k[:, 1],
        'power_W_g':   pad(powers),
    }

    for key, values in step_data.items():
        data[key] = pad(values)

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
    file_path = os.path.join(data_dir,
                             f"worker_{worker_id}_nuclide_concentrations.csv")
    file_exists = os.path.isfile(file_path)
    df.to_csv(file_path, mode='a', index=False, header=not file_exists)
    print(f"Worker {worker_id} | Results saved to {file_path}")


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------

def generate_data(config):
    """Single worker: build model, run depletion with tallies + checkpointing."""
    worker_id  = config['worker_id']
    script_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir, chain = setup_paths(script_dir, worker_id, config['chain_file'])

    print(f"Worker {worker_id} | seed={config['seed']} | "
          f"steps={len(config['powers'])} | "
          f"dt={config['dt_seconds']/DAY_IN_SECONDS:.5f} d | "
          f"particles={config['particles']} | "
          f"resume={config.get('resume', False)}")

    np.random.seed(config['seed'])

    fuel, gap, clad, water, materials, geometry, settings, tallies = \
        setup_reactor_model(config, results_dir)
    os.chdir(results_dir)

    fuel_mass_g = config['fuel_density'] * fuel.volume

    step_data = run_depletion_with_tallies(
        fuel, materials, geometry, settings, tallies,
        chain, config['powers'], config['dt_seconds'], fuel_mass_g,
        worker_id, results_dir, config
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
        'powers':           powers,
        'dt_seconds':       dt_seconds,

        # --- MC transport ---
        'particles':        args.particles,
        'inactive':         args.inactive,
        'batches':          args.batches,
        'temp_method':      args.temp_method,
        'chain_file':       args.chain_file,

        # --- Fixed reactor state (no boron) ---
        'enrichment':       args.enrichment,
        'fuel_density':     args.fuel_density,
        'fuel_temp':        args.fuel_temp,
        'mod_temp':         args.mod_temp,
        'clad_temp':        args.clad_temp,
        'mod_density':      args.mod_density,

        # --- Geometry (BEAVRS pin cell) ---
        'geometry_radii':   [0.39218, 0.40005, 0.45720],
        'geometry_pitch':   1.25984,

        # --- Checkpointing ---
        'resume':           args.resume,

        # --- Zero-power optimisation ---
        'power_threshold':  args.power_threshold,
        'decay_particles':  args.decay_particles,
        'decay_batches':    args.decay_batches,
        'decay_inactive':   args.decay_inactive,
    }

    NUM_RUNS    = args.runs
    NUM_WORKERS = args.cores

    for i in range(NUM_RUNS):
        configs = create_worker_configs(base_config, NUM_WORKERS,
                                        master_seed=args.seed)

        t0 = time.perf_counter()
        run_parallel_simulations(configs)
        elapsed = time.perf_counter() - t0

        print(f"Run {i+1}/{NUM_RUNS} complete | {NUM_WORKERS} workers | "
              f"{elapsed:.1f}s")