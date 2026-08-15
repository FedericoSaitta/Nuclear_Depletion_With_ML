"""The learned depletion matrix A(t), extracted and converted to physical units.

Only meaningful for `model.matrix_ode`, where the right-hand side is
``dy/dt = A(u(t), y(t)) · y``. The matrix is read directly out of the network at
sampled points and rescaled from model units into 1/day.

Two caveats carried by the conversion below are recorded in AUDIT.md and are
NOT fixed here, because fixing either moves a published figure:

* **P1** — the physical span is hardcoded at 1000 days while the data spans 990,
  so every coefficient is ~1 % low.
* **P2** — the MinMax offset is dropped, which is exact only where a target's
  minimum is zero. U238 depletes by a few percent, so its diagonal is not
  interpretable as a depletion rate.
"""

import os

import numpy as np
import torch
from loguru import logger

from nuclear_surrogates.analysis.rollout import JACOBIAN_BATCH, SolveContext
from nuclear_surrogates.utils import plot

# See P1 above. Deliberately not derived from the time array.
FIGURE_T_TOTAL_DAYS = 1000


def unscaling_matrix(ctx: SolveContext, target_names):
    """Elementwise factors converting the scaled matrix to units of 1/day.

    In scaled space dy_s/dt_n = A_s @ y_s; in physical space
    A_phys[i,j] = A_s[i,j] * (range_i / range_j) / T_total_days. This ignores
    the MinMax offset (AUDIT.md P2) and uses a hardcoded span (P1).

    Returns (scale_matrix, T_total_days, time_unit).
    """
    n_target = len(target_names)

    # Scaled 0 and 1 invert to the physical min and max, so their difference
    # is the per-feature range without reaching into the scaler's internals.
    phys_min = ctx.unscale_targets(np.zeros((1, n_target)))[0]
    phys_max = ctx.unscale_targets(np.ones((1, n_target)))[0]
    ranges = phys_max - phys_min

    T_total_days = FIGURE_T_TOTAL_DAYS

    logger.info(f"  Physical time span: {T_total_days:.2f} days")
    for idx, name in enumerate(target_names):
        logger.info(f"  {name} range: {ranges[idx]:.6e}")

    return np.outer(ranges, 1.0 / ranges) / T_total_days, T_total_days, "days"


def depletion_matrix_analysis(
    ctx: SolveContext, all_inputs_scaled, all_trues_scaled, target_names, max_runs=20
):
    """Extract and visualise the learned depletion matrix A(t) over time."""
    logger.info(f"\n{'=' * 20}")
    logger.info("DEPLETION MATRIX ANALYSIS")
    logger.info(f"{'=' * 20}")

    num_runs, steps, _ = all_inputs_scaled.shape
    t_span = ctx.t_on_device()

    scale_matrix, _, time_unit = unscaling_matrix(ctx, target_names)

    max_runs = min(max_runs, num_runs)
    timestep_indices = np.linspace(0, steps - 1, min(30, steps), dtype=int)
    matrices_by_time = {t: [] for t in timestep_indices}

    for start in range(0, max_runs, JACOBIAN_BATCH):
        end = min(start + JACOBIAN_BATCH, max_runs)

        inputs_batch = ctx.as_tensor(all_inputs_scaled[start:end])
        trues_batch = ctx.as_tensor(all_trues_scaled[start:end])
        ctx.func.set_forcing(t_span, inputs_batch)

        for t_idx in timestep_indices:
            with torch.no_grad():
                forcing_t = ctx.func.interpolate_forcing(t_span[int(t_idx)])
                A = ctx.func.build_matrix(forcing_t, trues_batch[:, int(t_idx), :])

            for b in range(A.shape[0]):
                matrices_by_time[t_idx].append(A[b].cpu().numpy() * scale_matrix)

    all_matrices = [m for t in timestep_indices for m in matrices_by_time[t]]
    plot.plot_depletion_matrix_mean(
        np.mean(all_matrices, axis=0),
        np.std(all_matrices, axis=0),
        target_names,
        time_unit,
        os.path.join(ctx.result_dir, "depletion_matrix_mean.png"),
    )

    sorted_ts = sorted(timestep_indices)
    plot.plot_depletion_matrix_evolution(
        ctx.t_span.cpu().numpy()[sorted_ts],
        np.array([np.mean(matrices_by_time[t], axis=0) for t in sorted_ts]),
        np.array([np.std(matrices_by_time[t], axis=0) for t in sorted_ts]),
        target_names,
        time_unit,
        max_runs,
        os.path.join(ctx.result_dir, "depletion_matrix_evolution.png"),
    )
