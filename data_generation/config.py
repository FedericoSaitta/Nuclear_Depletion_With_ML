"""Loading a data-generation config, and deriving per-worker configs from it.

The same split the ML half uses: **the YAML describes the simulation, the
command line describes the machine.** A config under `data_generation/configs/`
says what is modelled, how the transport is solved and what history is applied;
cores, threads, output directory and the master seed are flags, because they
change between a laptop and a cluster without changing the physics.

That split is what makes a config publishable. One file, cited from the dataset,
is the whole answer to "what produced this".

No OpenMC import here, so validation can be tested in CI.
"""

import random
import uuid

import yaml
from loguru import logger

HOUR_IN_SECONDS = 3600
DAY_IN_SECONDS = 24 * HOUR_IN_SECONDS

# `model.symmetry` -> the fraction of one pin cell the model contains. Both
# values describe the same infinite lattice of identical pins: `quarter` cuts on
# the two symmetry planes through the pin centre, which costs a quarter of the
# tracking. Areas scale by exactly this fraction, the water box included, since
# (pitch/2)**2 is a quarter of pitch**2.
#
# It lives here rather than in `pin_sim` so the value can be validated in CI,
# which has no OpenMC build.
SYMMETRY_FRACTION = {"full": 1.0, "quarter": 0.25}

# Section -> keys. Unknown keys are an error rather than a warning: a typo in a
# config is otherwise discovered after a job has been queued for a day, and the
# run it produces is quietly not the one that was intended.
SCHEMA = {
    "model": {
        "enrichment",
        "fuel_density",
        "geometry_radii",
        "geometry_pitch",
        "symmetry",
        # Fixed reactor state. The CASL pipeline samples these per step and
        # gives ranges under `depletion` instead, so they are optional here.
        "fuel_temp",
        "mod_temp",
        "clad_temp",
        "mod_density",
    },
    "transport": {
        "particles",
        "batches",
        "inactive",
        "temp_method",
        "use_wmp",
        # Reduced-fidelity settings for steps below `power_threshold`, where
        # transport statistics do not matter. Zoom pipeline only.
        "power_threshold",
        "decay_particles",
        "decay_batches",
        "decay_inactive",
    },
    "depletion": {
        "chain_file",
        "num_steps",
        "delta_t_days",
        # Measured-history pipelines
        "power_file",
        "rated_power",
        "start_day",
        "end_day",
        "daily_dt",
        # Sampled-history pipeline: [low, high] ranges drawn per step
        "power",
        "t_fuel",
        "t_mod",
        "t_clad",
        "rho_mod",
        "boron_ppm",
    },
}

REQUIRED = {
    "model": {
        "enrichment",
        "fuel_density",
        "geometry_radii",
        "geometry_pitch",
        "symmetry",
    },
    "transport": {"particles", "batches", "inactive"},
    "depletion": {"chain_file"},
}


def load(path):
    """Read and validate a config. Returns a flat dict.

    The sections exist to organise the file for a reader; the pipelines want one
    namespace, so it is flattened on the way out. Keys are unique across
    sections by construction — `_check` fails if that ever stops being true.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise SystemExit(f"{path} is not a YAML mapping")

    _check(path, raw)

    flat = {}
    for section in SCHEMA:
        flat.update(raw.get(section) or {})

    logger.info(f"Loaded data-generation config {path} ({len(flat)} keys)")
    return flat


def _check(path, raw):
    unknown_sections = set(raw) - set(SCHEMA)
    if unknown_sections:
        raise SystemExit(
            f"{path}: unknown section(s) {sorted(unknown_sections)}; "
            f"expected {sorted(SCHEMA)}"
        )

    seen = {}
    for section, allowed in SCHEMA.items():
        values = raw.get(section) or {}
        if not isinstance(values, dict):
            raise SystemExit(f"{path}: section `{section}` must be a mapping")

        unknown = set(values) - allowed
        if unknown:
            raise SystemExit(
                f"{path}: unknown key(s) {sorted(unknown)} under `{section}`.\n"
                f"Known keys there: {sorted(allowed)}"
            )

        missing = REQUIRED[section] - set(values)
        if missing:
            raise SystemExit(
                f"{path}: `{section}` is missing {sorted(missing)} — the run "
                f"would fail once it reached the simulation."
            )

        for key in values:
            if key in seen:
                raise SystemExit(
                    f"{path}: `{key}` appears under both `{seen[key]}` and "
                    f"`{section}`; flattening would lose one"
                )
            seen[key] = section

    _check_model_values(path, raw.get("model") or {})


def _check_model_values(path, model):
    """Reject values a section-and-key check cannot catch.

    Both of these otherwise surface inside OpenMC, minutes into a run: an
    unknown symmetry as a KeyError on a lookup table in `pin_sim`, and a radii
    list of the wrong length as a pin cell missing a region.
    """
    symmetry = model.get("symmetry")
    if symmetry not in SYMMETRY_FRACTION:
        raise SystemExit(
            f"{path}: `model.symmetry` is {symmetry!r}; "
            f"expected one of {sorted(SYMMETRY_FRACTION)}"
        )

    radii = model.get("geometry_radii")
    if not isinstance(radii, list) or len(radii) != 3:
        raise SystemExit(
            f"{path}: `model.geometry_radii` must be three radii in cm — "
            f"[fuel outer, gap outer, clad outer] — but is {radii!r}"
        )
    if list(radii) != sorted(radii):
        raise SystemExit(
            f"{path}: `model.geometry_radii` {radii!r} must increase outwards"
        )


def create_worker_configs(base_config, num_workers, master_seed=None):
    """Per-worker configs with unique IDs and Monte-Carlo seeds.

    Seeding is order-sensitive: worker IDs come from `uuid4` (OS entropy) and
    worker seeds from the `random` module seeded here, so for a given master
    seed the seed sequence is identical to what this code has always produced.
    `tests/test_datagen_io.py` pins that.
    """
    if master_seed is not None:
        random.seed(master_seed)
        logger.info(f"Using master seed: {master_seed}")
    else:
        random.seed()
        logger.info("Using random seed (no master seed specified)")

    configs = []
    for i in range(1, num_workers + 1):
        config = base_config.copy()
        config["worker_id"] = f"{i}_{uuid.uuid4().hex[:8]}"
        config["seed"] = random.randint(1, 2**31 - 1)
        configs.append(config)
    return configs
