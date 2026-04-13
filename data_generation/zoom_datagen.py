"""
Zoom-in depletion: run hourly resolution over a subset of the BEAVRS cycle,
starting from depleted fuel compositions extracted from a coarser daily run.

Workflow:
  1. Load depletion_results.h5 from the daily (dt=1 day) run
  2. Export depleted materials at the chosen start day
  3. Slice the BEAVRS power history for [start_day, end_day]
  4. Run a fresh hourly depletion from those initial conditions

Usage:
  python zoom_datagen.py \\
      --daily-results path/to/daily/depletion_results.h5 \\
      --power-file data_beavers.txt \\
      --start-day 140 --end-day 170 \\
      -t 40 --particles 50000

The output is a new depletion_results.h5 and CSV in data_generation/data/
covering only the zoomed window at hourly resolution.
"""
import sys
import argparse

parser = argparse.ArgumentParser(
    description='Zoom-in hourly depletion from a daily run',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# --- Zoom window ---
parser.add_argument('--daily-results', type=str, required=True,
                    help='Path to depletion_results.h5 from the daily run')
parser.add_argument('--start-day', type=float, required=True,
                    help='Start day for the zoom window')
parser.add_argument('--end-day', type=float, required=True,
                    help='End day for the zoom window')
parser.add_argument('--daily-dt', type=float, default=1.0,
                    help='Time step of the daily run [days] (to find correct step index)')

# --- I/O ---
parser.add_argument('-p', '--power-file', type=str, required=True,
                    help='Path to full BEAVRS power history CSV')
parser.add_argument('-f', '--chain-file', type=str, default='chain_casl_pwr.xml',
                    help='Depletion chain file name')

# --- Parallelism ---
parser.add_argument('-t', '--threads', type=int, default=1,
                    help='OpenMP threads')

# --- MC transport settings ---
parser.add_argument('--particles', type=int, default=50_000,
                    help='Neutron histories per batch')
parser.add_argument('--inactive', type=int, default=15,
                    help='Inactive batches for source convergence')
parser.add_argument('--batches', type=int, default=60,
                    help='Total batches')
parser.add_argument('--temp-method', type=str, default='interpolation',
                    choices=['interpolation', 'nearest'],
                    help='Cross section temperature treatment')

# --- Depletion settings ---
parser.add_argument('--dt', type=float, default=0.0416667,
                    help='Hourly depletion time step [days] (default 1/24)')

# --- BEAVRS nominal power ---
parser.add_argument('--rated-power', type=float, default=41.7,
                    help='100%% rated specific power [W/g]')

# --- Fixed reactor state ---
parser.add_argument('--fuel-temp', type=float, default=900.0)
parser.add_argument('--mod-temp', type=float, default=580.0)
parser.add_argument('--clad-temp', type=float, default=620.0)
parser.add_argument('--mod-density', type=float, default=0.74)
parser.add_argument('--enrichment', type=float, default=3.1)
parser.add_argument('--fuel-density', type=float, default=10.4)

# --- Reproducibility ---
parser.add_argument('-s', '--seed', type=int, default=42,
                    help='Random seed')

# --- Zero-power optimisation ---
parser.add_argument('--power-threshold', type=float, default=0.01)
parser.add_argument('--decay-particles', type=int, default=100)
parser.add_argument('--decay-batches', type=int, default=10)
parser.add_argument('--decay-inactive', type=int, default=3)


if __name__ == "__main__":
    args = parser.parse_args()
else:
    args = None

import os
os.environ["OMP_NUM_THREADS"] = str(args.threads if args else 1)

import openmc
import openmc.deplete
import math
import time
import glob
import numpy as np
import pandas as pd
from datetime import datetime

from quarter_sim import (
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

GEOMETRY_RADII = [0.39218, 0.40005, 0.45720]
GEOMETRY_PITCH = 1.25984


# ---------------------------------------------------------------------------
# Load depleted materials from daily run
# ---------------------------------------------------------------------------

def load_depleted_materials(daily_results_path, start_day, daily_dt):
    """Extract depleted material compositions from the daily run.

    Uses Results.export_to_materials() to write a materials.xml with
    the depleted fuel composition at the given step, then reads it back.

    Returns:
        materials_path: path to the exported materials.xml
        step_index:     the step index used
    """
    print(f"Loading daily results from: {daily_results_path}")
    results = openmc.deplete.Results(daily_results_path)

    # Find step index for start_day
    step_index = int(round(start_day / daily_dt))

    # Validate
    n_steps = len(results) - 1  # results has n_steps+1 entries
    if step_index > n_steps:
        raise ValueError(
            f"Requested step {step_index} (day {start_day}) but daily run "
            f"only has {n_steps} completed steps"
        )

    # Get time to confirm
    time_arr, k = results.get_keff()
    actual_time_days = time_arr[step_index] / DAY_IN_SECONDS
    actual_keff = k[step_index, 0]

    print(f"  Step index: {step_index}")
    print(f"  Time at step: {actual_time_days:.2f} days")
    print(f"  keff at step: {actual_keff:.5f}")

    # Export materials at this step
    export_dir = "zoom_initial_materials"
    os.makedirs(export_dir, exist_ok=True)
    materials_path = os.path.join(export_dir, "materials.xml")
    results.export_to_materials(step_index, path=materials_path)
    print(f"  Exported depleted materials to: {materials_path}")

    # Also print key nuclide concentrations for verification
    for nuc in ['U235', 'U238', 'Pu239', 'Pu240', 'Pu241']:
        try:
            _, conc = results.get_atoms("1", nuc, nuc_units="atom/b-cm")
            print(f"    {nuc}: {conc[step_index]:.6e} atom/b-cm")
        except Exception:
            pass

    return materials_path, step_index


# ---------------------------------------------------------------------------
# Load and slice power history
# ---------------------------------------------------------------------------

def load_power_window(filepath, start_day, end_day, dt_days, rated_power):
    """Load BEAVRS power history and extract the window [start_day, end_day]
    at hourly resolution.
    """
    filepath = os.path.abspath(filepath)
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
    powers_raw = np.array(percents) / 100.0 * rated_power

    # Interpolate onto hourly grid for the window
    duration = end_day - start_day
    num_steps = int(round(duration / dt_days))
    t_window = start_day + np.arange(num_steps) * dt_days
    powers_window = np.interp(t_window, days, powers_raw)

    n_zero = np.sum(powers_window < 0.01)
    print(f"Power window: day {start_day} to {end_day}")
    print(f"  {num_steps} steps at dt = {dt_days:.5f} days ({dt_days*24:.1f} hours)")
    print(f"  Power range: [{powers_window.min():.2f}, {powers_window.max():.2f}] W/g")
    print(f"  Zero-power steps: {n_zero}/{num_steps} ({100*n_zero/num_steps:.1f}%)")

    return list(powers_window), t_window


# ---------------------------------------------------------------------------
# Build model from exported materials
# ---------------------------------------------------------------------------

def build_model_from_exported(materials_path, config):
    """Load the exported depleted materials and build a full model.

    The materials.xml from export_to_materials contains the depleted fuel
    with all ~200 nuclides. We load it, set volumes, ensure depletable
    flag, then build geometry/settings/tallies.
    """
    materials = openmc.Materials.from_xml(materials_path)

    # Identify materials by name
    fuel = None
    gap = None
    clad = None
    water = None
    for mat in materials:
        if mat.name == "uo2":
            fuel = mat
        elif mat.name == "gap":
            gap = mat
        elif mat.name == "clad":
            clad = mat
        elif mat.name == "water":
            water = mat

    if fuel is None:
        raise RuntimeError("Could not find 'uo2' material in exported file")

    # Ensure fuel is depletable
    fuel.depletable = True

    # Set temperatures (not stored in materials.xml)
    fuel.temperature  = config['fuel_temp']
    if gap:
        gap.temperature = config['fuel_temp']
    if clad:
        clad.temperature = config['clad_temp']
    if water:
        water.temperature = config['mod_temp']

    # Set volumes for quarter-pin geometry
    radii = config['geometry_radii']
    pitch = config['geometry_pitch']
    set_material_volumes_quarter(fuel, gap, clad, water, radii, pitch)

    # Build geometry
    geometry = create_quarter_geometry(
        (fuel, gap, clad, water), radii, pitch
    )

    # Settings and tallies
    settings = create_settings(config)
    tallies = create_tallies(fuel)

    return fuel, gap, clad, water, materials, geometry, settings, tallies


# ---------------------------------------------------------------------------
# Tally extraction (same as quarter_datagen.py)
# ---------------------------------------------------------------------------

def extract_tallies_from_statepoint(batches):
    sp_file = f"statepoint.{batches}.h5"
    result = {'flux': float('nan'), 'flux_std': float('nan')}
    for nuc in FISSION_NUCLIDES:
        result[f'{nuc}_fission'] = float('nan')
        result[f'{nuc}_fission_std'] = float('nan')
    for nuc in CAPTURE_NUCLIDES:
        result[f'{nuc}_capture'] = float('nan')
        result[f'{nuc}_capture_std'] = float('nan')

    if not os.path.exists(sp_file):
        sp_files = sorted(glob.glob('statepoint.*.h5'))
        if sp_files:
            sp_file = sp_files[-1]
        else:
            return result

    try:
        sp = openmc.StatePoint(sp_file)
        try:
            t = sp.get_tally(id=9001)
            result['flux'] = t.mean.flatten()[0]
            result['flux_std'] = t.std_dev.flatten()[0]
        except Exception:
            pass
        try:
            t = sp.get_tally(id=9002)
            means = t.mean.flatten()
            stds = t.std_dev.flatten()
            for j, nuc in enumerate(FISSION_NUCLIDES):
                result[f'{nuc}_fission'] = means[j]
                result[f'{nuc}_fission_std'] = stds[j]
        except Exception:
            pass
        try:
            t = sp.get_tally(id=9003)
            means = t.mean.flatten()
            stds = t.std_dev.flatten()
            for j, nuc in enumerate(CAPTURE_NUCLIDES):
                result[f'{nuc}_capture'] = means[j]
                result[f'{nuc}_capture_std'] = stds[j]
        except Exception:
            pass
        sp.close()
    except Exception:
        pass

    return result


def make_zero_tally_data():
    result = {'flux': 0.0, 'flux_std': 0.0}
    for nuc in FISSION_NUCLIDES:
        result[f'{nuc}_fission'] = 0.0
        result[f'{nuc}_fission_std'] = 0.0
    for nuc in CAPTURE_NUCLIDES:
        result[f'{nuc}_capture'] = 0.0
        result[f'{nuc}_capture_std'] = 0.0
    return result


def compute_fission_power_fractions(tally_data):
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
# Depletion loop
# ---------------------------------------------------------------------------

def run_zoom_depletion(fuel, materials, geometry, settings, tallies,
                       chain, powers, dt_seconds, fuel_mass_g,
                       results_dir, config):
    """Run the zoomed-in depletion with tally extraction."""
    num_steps = len(powers)
    batches = settings.batches
    power_threshold = config.get('power_threshold', 0.01)
    decay_particles = config.get('decay_particles', 100)
    decay_batches = config.get('decay_batches', 10)
    decay_inactive = config.get('decay_inactive', 3)

    step_keys = ['flux', 'flux_std']
    for nuc in FISSION_NUCLIDES:
        step_keys.extend([f'{nuc}_fission', f'{nuc}_fission_std',
                          f'{nuc}_fission_power_frac'])
    for nuc in CAPTURE_NUCLIDES:
        step_keys.extend([f'{nuc}_capture', f'{nuc}_capture_std'])

    step_data = {k: [] for k in step_keys}

    for i in range(num_steps):
        step_power_W = powers[i] * fuel_mass_g
        is_decay_only = powers[i] < power_threshold

        step_type = "DECAY" if is_decay_only else "POWER"
        print(f"\nStep {i+1}/{num_steps} | P = {powers[i]:.2f} W/g | {step_type}")

        if is_decay_only:
            decay_settings = openmc.Settings()
            decay_settings.particles = decay_particles
            decay_settings.inactive  = decay_inactive
            decay_settings.batches   = decay_batches
            decay_settings.verbosity = 1
            decay_settings.output    = {'tallies': False}
            source = openmc.IndependentSource()
            source.space  = openmc.stats.Point((0.05, 0.05, 0))
            source.angle  = openmc.stats.Isotropic()
            source.energy = openmc.stats.Watt()
            decay_settings.source = source
            decay_settings.temperature = settings.temperature
            if settings.seed is not None:
                decay_settings.seed = settings.seed
            decay_settings.export_to_xml(path=results_dir)
            model = openmc.model.Model(geometry, materials, decay_settings)
        else:
            settings.export_to_xml(path=results_dir)
            model = openmc.model.Model(geometry, materials, settings, tallies)

        prev_results_file = "depletion_results.h5" if i > 0 else None
        if prev_results_file and os.path.exists(prev_results_file):
            prev_results = openmc.deplete.Results(prev_results_file)
            operator = openmc.deplete.CoupledOperator(model, chain,
                                                       prev_results=prev_results)
        else:
            operator = openmc.deplete.CoupledOperator(model, chain)

        integrator = openmc.deplete.PredictorIntegrator(
            operator, [dt_seconds], [step_power_W], timestep_units='s'
        )
        integrator.integrate()

        # Extract tallies
        if is_decay_only:
            tally_data = make_zero_tally_data()
        else:
            tally_data = extract_tallies_from_statepoint(batches)

        step_data['flux'].append(tally_data['flux'])
        step_data['flux_std'].append(tally_data['flux_std'])

        for nuc in FISSION_NUCLIDES:
            step_data[f'{nuc}_fission'].append(tally_data[f'{nuc}_fission'])
            step_data[f'{nuc}_fission_std'].append(tally_data[f'{nuc}_fission_std'])
        for nuc in CAPTURE_NUCLIDES:
            step_data[f'{nuc}_capture'].append(tally_data[f'{nuc}_capture'])
            step_data[f'{nuc}_capture_std'].append(tally_data[f'{nuc}_capture_std'])

        fracs = compute_fission_power_fractions(tally_data)
        for nuc in FISSION_NUCLIDES:
            step_data[f'{nuc}_fission_power_frac'].append(
                fracs[f'{nuc}_fission_power_frac'])

        if not is_decay_only:
            u235_frac  = fracs.get('U235_fission_power_frac', 0)
            pu239_frac = fracs.get('Pu239_fission_power_frac', 0)
            print(f"  flux = {tally_data['flux']:.4e} +/- "
                  f"{tally_data['flux_std']:.4e} | "
                  f"U235 frac = {u235_frac:.3f} | Pu239 frac = {pu239_frac:.3f}")

        # Clean up old statepoint files
        if i > 1:
            old_sp = f"openmc_simulation_n{i-2}.h5"
            if os.path.exists(old_sp):
                os.remove(old_sp)

    return step_data


# ---------------------------------------------------------------------------
# Results extraction and saving
# ---------------------------------------------------------------------------

def extract_and_save(results, powers, dt_seconds, step_data, start_day,
                     script_dir):
    """Extract results and save to CSV."""
    time_arr, k = results.get_keff()
    # Shift times to absolute days (relative to BOC, not zoom start)
    time_days = time_arr / DAY_IN_SECONDS + start_day

    label = f"beavrs_zoom_{start_day:.0f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    # Save
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    df = pd.DataFrame(data)
    file_path = os.path.join(
        data_dir,
        f"zoom_day{start_day:.0f}_nuclide_concentrations.csv"
    )
    file_exists = os.path.isfile(file_path)
    df.to_csv(file_path, mode='a', index=False, header=not file_exists)
    print(f"\nResults saved to {file_path}")
    print(f"  {num_points} time points, {len(list(nuclides))} nuclides")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    t_start = time.perf_counter()

    # --- 1. Load depleted materials from daily run ---
    materials_path, step_index = load_depleted_materials(
        args.daily_results,
        args.start_day,
        args.daily_dt
    )

    # --- 2. Slice power history for zoom window ---
    powers, time_days = load_power_window(
        args.power_file,
        args.start_day,
        args.end_day,
        dt_days=args.dt,
        rated_power=args.rated_power
    )
    dt_seconds = args.dt * DAY_IN_SECONDS

    # --- 3. Build model from exported materials ---
    config = {
        'particles':     args.particles,
        'inactive':      args.inactive,
        'batches':       args.batches,
        'temp_method':   args.temp_method,
        'enrichment':    args.enrichment,
        'fuel_density':  args.fuel_density,
        'fuel_temp':     args.fuel_temp,
        'mod_temp':      args.mod_temp,
        'clad_temp':     args.clad_temp,
        'mod_density':   args.mod_density,
        'geometry_radii': GEOMETRY_RADII,
        'geometry_pitch': GEOMETRY_PITCH,
        'seed':          args.seed,
        'power_threshold':  args.power_threshold,
        'decay_particles':  args.decay_particles,
        'decay_batches':    args.decay_batches,
        'decay_inactive':   args.decay_inactive,
    }

    fuel, gap, clad, water, materials, geometry, settings, tallies = \
        build_model_from_exported(materials_path, config)

    # --- 4. Set up working directory ---
    results_dir = os.path.abspath(
        os.path.join(script_dir, "results",
                     f"zoom_day{args.start_day:.0f}_to_{args.end_day:.0f}")
    )
    os.makedirs(results_dir, exist_ok=True)

    # Set up cross sections
    openmc.config['cross_sections'] = os.path.join(
        script_dir, "../data/cross_sections.xml"
    )
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    os.environ['OPENMC_CROSS_SECTIONS'] = str(openmc.config['cross_sections'])

    chain_file = os.path.join(script_dir, "../data", args.chain_file)
    chain = openmc.deplete.Chain.from_xml(chain_file)

    # Export model files
    geometry.export_to_xml(path=results_dir)
    settings.export_to_xml(path=results_dir)
    materials.export_to_xml(path=results_dir)

    os.chdir(results_dir)
    fuel_mass_g = args.fuel_density * fuel.volume

    print(f"\nStarting zoom depletion: day {args.start_day} -> {args.end_day}")
    print(f"  {len(powers)} hourly steps, {args.particles} particles, "
          f"{args.threads} threads")

    # --- 5. Run depletion ---
    step_data = run_zoom_depletion(
        fuel, materials, geometry, settings, tallies,
        chain, powers, dt_seconds, fuel_mass_g,
        results_dir, config
    )

    # --- 6. Extract and save ---
    results = openmc.deplete.Results("depletion_results.h5")
    extract_and_save(results, powers, dt_seconds, step_data,
                     args.start_day, script_dir)

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal runtime: {elapsed/3600:.1f} hours")
