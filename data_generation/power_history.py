"""Reading and resampling the BEAVRS Cycle 1 measured power history.

No OpenMC here, deliberately: this is the one piece of physics input that can be
checked without a nuclear-data library, so it lives where CI can test it.

`quarter_datagen` resamples the whole cycle onto a uniform grid; `zoom_datagen`
resamples one window of it at a finer step. Both are `resample_power` with
different bounds — they used to be two functions that had drifted apart.
"""

import numpy as np
from loguru import logger


def parse_power_history(filepath):
    """Parse a BEAVRS Cycle 1 power history into (days, percent) arrays.

    The committed `beavrs_cycle1_power.csv` is 527 bare ``day,percent`` rows with
    no header. The banner and header handling below is for the file as it comes
    out of the BEAVRS release, which prefixes a "Cycle 1" line and a
    "Day, Percent" header — both forms parse, so a freshly downloaded table can
    be dropped in without editing.
    """
    filepath = str(filepath)
    days, percents = [], []
    header_found = False

    with open(filepath) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                # A blank line terminates the table, but only once it has begun:
                # the release format opens with blank lines above the banner.
                if header_found:
                    break
                continue
            if line.lower().startswith("cycle 1"):
                continue
            if line.lower().startswith("day"):
                header_found = True
                continue
            # Any other text marks the start of the next cycle's table.
            if line.lower().startswith("cycle") or line[0].isalpha():
                break
            try:
                parts = line.split(",")
                days.append(float(parts[0]))
                percents.append(float(parts[1]))
            except (ValueError, IndexError):
                break

    if not days:
        raise ValueError(
            f"{filepath} contained no `day,percent` rows — is it a BEAVRS power "
            f"history table?"
        )

    return np.array(days), np.array(percents)


def resample_power(filepath, rated_power_W_g, dt_days, start_day=0.0, end_day=None):
    """Resample the measured history onto a uniform grid, in W/g.

    *end_day* defaults to the last day in the file, which is the whole-cycle
    case. Returns ``(powers, times)`` where *times* are absolute days from
    beginning-of-cycle, so a zoomed window keeps its true position in the cycle
    rather than restarting at zero.

    Percentages are converted with *rated_power_W_g* as 100 % rated specific
    power. Steps that land at exactly zero are decay-only inside OpenMC: it runs
    no transport and writes no statepoint for them, which is why the callers
    record their tally columns as zeros rather than reading a file.
    """
    days, percents = parse_power_history(filepath)
    powers_measured = percents / 100.0 * rated_power_W_g

    if end_day is None:
        end_day = float(days[-1])

    num_steps = int(np.round((end_day - start_day) / dt_days))
    if num_steps < 1:
        raise ValueError(
            f"[{start_day}, {end_day}] at dt={dt_days} days gives {num_steps} "
            f"steps; widen the window or shorten the step"
        )

    times = start_day + np.arange(num_steps) * dt_days
    powers = np.interp(times, days, powers_measured)

    n_zero = int(np.sum(powers == 0.0))
    logger.info(
        f"Power history {filepath}: {len(days)} measured points -> "
        f"{num_steps} steps of {dt_days:.5f} d ({dt_days * 24:.1f} h) "
        f"over days {start_day:g}-{end_day:g}"
    )
    logger.info(
        f"  range [{powers.min():.2f}, {powers.max():.2f}] W/g, "
        f"decay-only steps (P=0): {n_zero}/{num_steps} "
        f"({100 * n_zero / num_steps:.1f}%)"
    )

    return list(powers), times
