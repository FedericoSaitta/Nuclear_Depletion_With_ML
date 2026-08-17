"""Reading OpenMC statepoint tallies into flat per-step dictionaries.

Shared by quarter_datagen.py and zoom_datagen.py, which used to carry
near-identical copies of every function here. Column names follow
``{nuclide}_fission`` / ``{nuclide}_capture`` (each with a ``_std`` partner),
driven by the nuclide lists in `nuclides` — extending those lists automatically
extends the dataset schema.

`openmc` is imported inside `read_statepoint_tallies` rather than at module
level. That one function is the only thing here that needs it, and keeping the
import local means the column schema and the power-fraction maths can be
imported — and tested — in an environment with no OpenMC build, which is what
CI has.
"""

import contextlib

import numpy as np
from loguru import logger

from nuclides import CAPTURE_NUCLIDES, FISSION_NUCLIDES, FISSION_Q_VALUES


def step_keys():
    """Ordered column names for one step's tally data."""
    keys = ["flux", "flux_std"]
    for nuc in FISSION_NUCLIDES:
        keys.extend(
            [f"{nuc}_fission", f"{nuc}_fission_std", f"{nuc}_fission_power_frac"]
        )
    for nuc in CAPTURE_NUCLIDES:
        keys.extend([f"{nuc}_capture", f"{nuc}_capture_std"])
    return keys


def _filled_tally(value):
    result = {"flux": value, "flux_std": value}
    for nuc in FISSION_NUCLIDES:
        result[f"{nuc}_fission"] = value
        result[f"{nuc}_fission_std"] = value
    for nuc in CAPTURE_NUCLIDES:
        result[f"{nuc}_capture"] = value
        result[f"{nuc}_capture_std"] = value
    return result


def nan_tally():
    """Placeholder row for a transport step whose statepoint could not be read."""
    return _filled_tally(float("nan"))


def zero_tally():
    """All-zero row for a decay-only step, which runs no transport."""
    return _filled_tally(0.0)


def read_statepoint_tallies(sp_path):
    """Read flux / fission / capture tallies from one statepoint file.

    Tally IDs match ``quarter_sim.create_tallies`` (9001 flux, 9002 fission,
    9003 capture). Each block is independent, so a missing tally leaves its
    columns at NaN without aborting the others.
    """
    import openmc

    result = nan_tally()
    try:
        sp = openmc.StatePoint(sp_path)
        # Each block is suppressed independently and on purpose: one tally that
        # failed to score should leave its own columns NaN, not cost the run the
        # other two. A NaN column is visible downstream; a lost one is not.
        with contextlib.suppress(Exception):
            t = sp.get_tally(id=9001)
            result["flux"] = float(t.mean.flatten()[0])
            result["flux_std"] = float(t.std_dev.flatten()[0])
        with contextlib.suppress(Exception):
            t = sp.get_tally(id=9002)
            m = t.mean.flatten()
            s = t.std_dev.flatten()
            for j, nuc in enumerate(FISSION_NUCLIDES):
                result[f"{nuc}_fission"] = float(m[j])
                result[f"{nuc}_fission_std"] = float(s[j])
        with contextlib.suppress(Exception):
            t = sp.get_tally(id=9003)
            m = t.mean.flatten()
            s = t.std_dev.flatten()
            for j, nuc in enumerate(CAPTURE_NUCLIDES):
                result[f"{nuc}_capture"] = float(m[j])
                result[f"{nuc}_capture_std"] = float(s[j])
        sp.close()
    except Exception as exc:  # noqa: BLE001 - a bad statepoint must not kill the run
        logger.warning(f"Could not read {sp_path}: {exc}")
    return result


def compute_fission_power_fractions(tally_data):
    """Per-nuclide share of fission power, weighted by the Q-values."""
    fracs = {}
    total_power = 0.0
    for nuc in FISSION_NUCLIDES:
        rate = tally_data.get(f"{nuc}_fission", 0.0)
        if np.isnan(rate):
            rate = 0.0
        total_power += rate * FISSION_Q_VALUES[nuc]
    for nuc in FISSION_NUCLIDES:
        rate = tally_data.get(f"{nuc}_fission", 0.0)
        if np.isnan(rate):
            rate = 0.0
        if total_power > 0:
            fracs[f"{nuc}_fission_power_frac"] = (
                rate * FISSION_Q_VALUES[nuc]
            ) / total_power
        else:
            fracs[f"{nuc}_fission_power_frac"] = 0.0
    return fracs
