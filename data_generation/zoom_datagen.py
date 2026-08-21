"""Zoom-in depletion: hourly resolution over a window of the BEAVRS cycle.

Starts from the depleted fuel compositions of a coarser daily run rather than
from fresh fuel:

  1. Load `depletion_results.h5` from the daily (dt = 1 day) run
  2. Export the depleted materials at the chosen start day
  3. Slice the BEAVRS power history for [start_day, end_day]
  4. Run a fresh hourly depletion from those initial conditions

Unlike `quarter_datagen.py` this steps one `integrate()` call per timestep,
which is what lets it swap in reduced-fidelity transport for steps below
`transport.power_threshold`, where transport statistics do not matter. The model
itself is whatever `pin_sim` builds from the config, so it is the same pin cell
the daily run depleted.

    DAILY=data_generation/results/worker_1_<hash>/depletion_results.h5
    uv run --extra sim python data_generation/zoom_datagen.py \
        --config data_generation/configs/beavrs_zoom.yaml \
        --daily-results $DAILY -t 80

Output is a new `depletion_results.h5` plus an HDF5 in `data_generation/data/`
covering only the zoomed window, on absolute cycle days.
"""

import argparse

# Parse argv and set OMP_NUM_THREADS BEFORE importing OpenMC, which reads the
# variable at import time. This ordering is why E402 is scoped off for this
# file in pyproject.toml.
parser = argparse.ArgumentParser(
    description="Zoom-in hourly depletion from a daily BEAVRS run",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--config",
    required=True,
    help="simulation config, e.g. data_generation/configs/beavrs_zoom.yaml",
)
parser.add_argument(
    "--daily-results",
    required=True,
    help="depletion_results.h5 from the daily run to start from",
)
parser.add_argument("-t", "--threads", type=int, default=1, help="OpenMP threads")
parser.add_argument("-s", "--seed", type=int, default=42, help="Monte-Carlo seed")
parser.add_argument(
    "--out", default=None, help="output directory (default: data_generation/data)"
)
args = parser.parse_args()

import os

os.environ["OMP_NUM_THREADS"] = str(args.threads)

import glob
import time
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
    deplete_one_step,
    depletion_results_frame,
    setup_paths,
)
from power_history import resample_power

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_depleted_materials(daily_results_path, start_day, daily_dt):
    """Export the daily run's depleted composition at *start_day*.

    `Results.export_to_materials` writes a materials.xml carrying the depleted
    fuel — all ~200 nuclides — which is what makes this a continuation of that
    run rather than a fresh one. Returns (materials_path, step_index).
    """
    logger.info(f"Loading daily results from {daily_results_path}")
    results = openmc.deplete.Results(daily_results_path)

    step_index = int(round(start_day / daily_dt))
    completed = len(results) - 1  # results holds n_steps + 1 entries
    if step_index > completed:
        raise SystemExit(
            f"Requested step {step_index} (day {start_day}) but the daily run "
            f"only has {completed} completed steps"
        )

    time_arr, k = results.get_keff()
    logger.info(
        f"  step {step_index} | t = {time_arr[step_index] / DAY_IN_SECONDS:.2f} d | "
        f"keff = {k[step_index, 0]:.5f}"
    )

    export_dir = os.path.join(SCRIPT_DIR, "results", "zoom_initial_materials")
    os.makedirs(export_dir, exist_ok=True)
    materials_path = os.path.join(export_dir, "materials.xml")
    results.export_to_materials(step_index, path=materials_path)
    logger.info(f"  exported depleted materials to {materials_path}")

    for nuc in ("U235", "U238", "Pu239", "Pu240", "Pu241"):
        try:
            _, conc = results.get_atoms("1", nuc, nuc_units="atom/b-cm")
        except (KeyError, ValueError):
            continue
        logger.info(f"    {nuc}: {conc[step_index]:.6e} atom/b-cm")

    return materials_path, step_index


def build_model_from_exported(materials_path, cfg):
    """Rebuild the pin-cell model around the exported depleted materials.

    Only the compositions come from the daily run. Temperatures are not stored
    in materials.xml so they are reapplied from the config, and the geometry,
    settings and tallies are the ones `pin_sim` builds for any config — this is
    the same pin cell, continued.

    The moderator composition is deliberately *not* rewritten: water is not
    depletable, so what the daily run exported is already this config's water,
    and leaving it alone keeps the continuation exact.
    """
    materials = openmc.Materials.from_xml(materials_path)
    by_name = {mat.name: mat for mat in materials}

    names = ("uo2", "gap", "clad", "water")
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(
            f"{materials_path} has no {missing} material(s) — is it a daily run "
            f"from quarter_datagen.py?"
        )
    mats = tuple(by_name[name] for name in names)

    fuel = mats[0]
    fuel.depletable = True
    pin_sim.set_temperatures(
        mats,
        fuel_temp=cfg["fuel_temp"],
        clad_temp=cfg["clad_temp"],
        mod_temp=cfg["mod_temp"],
    )

    symmetry = cfg["symmetry"]
    radii, pitch = cfg["geometry_radii"], cfg["geometry_pitch"]
    pin_sim.set_volumes(mats, radii, pitch, symmetry)
    geometry = pin_sim.create_geometry(mats, radii, pitch, symmetry)
    settings = pin_sim.create_settings(cfg, symmetry, tally_output=True)
    tallies = pin_sim.create_tallies(fuel)

    return mats, materials, geometry, settings, tallies


def _decay_settings(cfg, results_dir):
    """Reduced-fidelity settings for a step with no meaningful transport.

    The settings the config's `transport` section describes, with the particle
    counts swapped for the `decay_*` ones. Seed, temperature method and source
    point come along unchanged, which is the point of building them through
    `pin_sim` rather than by hand.
    """
    decay = pin_sim.create_settings(
        {
            **cfg,
            "particles": cfg["decay_particles"],
            "batches": cfg["decay_batches"],
            "inactive": cfg["decay_inactive"],
        },
        cfg["symmetry"],
    )
    decay.export_to_xml(path=results_dir)
    return decay


def _read_step_tallies(batches):
    """Read this step's statepoint, falling back to the newest one present."""
    path = f"statepoint.{batches}.h5"
    if not os.path.exists(path):
        candidates = sorted(glob.glob("statepoint.*.h5"))
        if not candidates:
            return tally_io.nan_tally()
        path = candidates[-1]
    return tally_io.read_statepoint_tallies(path)


def run_zoom_depletion(
    model_parts, chain_file, powers, dt_seconds, fuel_mass_g, cfg, results_dir
):
    """Deplete the window one step at a time, extracting tallies as it goes."""
    _mats, materials, geometry, settings, tallies = model_parts
    num_steps = len(powers)
    threshold = cfg["power_threshold"]

    step_data = tally_io.new_step_data()

    for i in range(num_steps):
        decay_only = powers[i] < threshold
        logger.info(
            f"Step {i + 1}/{num_steps} | P = {powers[i]:.2f} W/g | "
            f"{'DECAY' if decay_only else 'POWER'}"
        )

        if decay_only:
            model = openmc.model.Model(
                geometry, materials, _decay_settings(cfg, results_dir)
            )
        else:
            settings.export_to_xml(path=results_dir)
            model = openmc.model.Model(geometry, materials, settings, tallies)

        deplete_one_step(
            model,
            chain_file,
            dt_seconds,
            powers[i] * fuel_mass_g,
            continue_from="depletion_results.h5" if i > 0 else None,
        )

        tally = (
            tally_io.zero_tally()
            if decay_only
            else _read_step_tallies(settings.batches)
        )
        fractions = tally_io.append_step(step_data, tally)

        if not decay_only:
            logger.info(
                f"  flux = {tally['flux']:.4e} +/- {tally['flux_std']:.4e} | "
                f"U235 frac = {fractions.get('U235_fission_power_frac', 0):.3f} | "
                f"Pu239 frac = {fractions.get('Pu239_fission_power_frac', 0):.3f}"
            )

        # Statepoints are large and only the current one is ever read back.
        if i > 1:
            stale = f"openmc_simulation_n{i - 2}.h5"
            if os.path.exists(stale):
                os.remove(stale)

    return step_data


if __name__ == "__main__":
    started = time.perf_counter()

    cfg = config_module.load(args.config)
    cfg["seed"] = args.seed
    output_dir = args.out or os.path.join(SCRIPT_DIR, "data")
    start_day, end_day = cfg["start_day"], cfg["end_day"]

    materials_path, _step_index = load_depleted_materials(
        args.daily_results, start_day, cfg["daily_dt"]
    )

    powers, _times = resample_power(
        os.path.join(SCRIPT_DIR, cfg["power_file"]),
        rated_power_W_g=cfg["rated_power"],
        dt_days=cfg["delta_t_days"],
        start_day=start_day,
        end_day=end_day,
    )
    dt_seconds = cfg["delta_t_days"] * DAY_IN_SECONDS

    model_parts = build_model_from_exported(materials_path, cfg)
    mats, materials, geometry, settings, _tallies = model_parts
    fuel = mats[0]

    # Same cross-sections / chain / working-directory setup every worker gets;
    # this pipeline is single-process, so it uses one fixed "worker" name.
    worker_id = f"zoom_day{start_day:.0f}_to_{end_day:.0f}"
    results_dir, chain_file = setup_paths(SCRIPT_DIR, worker_id, cfg["chain_file"])

    geometry.export_to_xml(path=results_dir)
    settings.export_to_xml(path=results_dir)
    materials.export_to_xml(path=results_dir)
    os.chdir(results_dir)

    fuel_mass_g = cfg["fuel_density"] * fuel.volume
    logger.info(
        f"Zoom depletion day {start_day} -> {end_day} | {len(powers)} hourly steps | "
        f"{cfg['particles']} particles | {args.threads} threads"
    )

    step_data = run_zoom_depletion(
        model_parts, chain_file, powers, dt_seconds, fuel_mass_g, cfg, results_dir
    )

    results = openmc.deplete.Results("depletion_results.h5")
    label = f"beavrs_zoom_{start_day:.0f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # The results start at zero but the window sits partway through the cycle;
    # the absolute day is what makes this comparable to the daily run.
    data = depletion_results_frame(
        results,
        label,
        per_step={"power_W_g": powers, **step_data},
        time_offset_days=start_day,
    )
    dataset_io.write_run_h5(data, os.path.join(output_dir, f"{worker_id}.h5"))

    dataset_io.write_manifest(
        output_dir,
        config=config_module.load(args.config),
        worker_configs=[{"worker_id": worker_id, "seed": args.seed}],
        chain_file=chain_file,
        extra={
            "pipeline": "beavrs_zoom",
            "config_path": args.config,
            "daily_results": args.daily_results,
            "num_steps": len(powers),
        },
    )

    logger.info(f"Total runtime: {(time.perf_counter() - started) / 3600:.1f} hours")
