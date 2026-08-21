"""Per-step teacher-forced permutation importance.

At each sampled timestep the true y(t) is the initial condition, one feature is
replaced with values from a randomly chosen donor run, the model integrates one
step, and the change in |error| at t+1 is that feature's importance there.

Sampling per step is what makes the zero-start isotopes measurable: at t=0 U239
is zero in every run so permuting it does nothing, but by t=5 it has built up
and permuting it registers. The time-evolution plot shows exactly when each
isotope starts mattering.
"""

import csv
import os

import numpy as np
from loguru import logger

from nuclear_surrogates.analysis.rollout import SolveContext, single_step_batch
from nuclear_surrogates.utils import plot

TABLE_FILENAME = "stepwise_importance.md"


def stepwise_importance(
    ctx: SolveContext,
    all_inputs_scaled,
    all_trues_scaled,
    target_names,
    forcing_names,
    log,
    n_permutations=5,
    seed=0,
    max_timesteps=50,
):
    """Run the sweep, write its tables and figures, and return its arrays.

    *log* is the LightningModule's `self.log`, so the per-feature numbers reach
    the same metric sink as everything else.

    Returns (mean_delta, sem_delta, mean_pct, sem_pct, mean_by_time).
    """
    logger.info(f"\n{'=' * 20}")
    logger.info(f"PER-STEP TEACHER-FORCED IMPORTANCE (K={n_permutations})")
    logger.info(f"{'=' * 20}")

    num_runs, steps, n_input = all_inputs_scaled.shape
    n_state = all_trues_scaled.shape[2]
    n_features = n_input + n_state
    rng = np.random.default_rng(seed)

    all_feature_names = list(forcing_names) + list(target_names)
    feature_types = ["Forcing"] * n_input + ["State"] * n_state

    # Stop at steps-2 so that t+1 always exists.
    timestep_indices = np.linspace(
        0, steps - 2, min(max_timesteps, steps - 1), dtype=int
    )
    n_sampled = len(timestep_indices)
    t_fractions = ctx.t_span.cpu().numpy()[timestep_indices]

    logger.info(
        f"  {num_runs} runs × {n_sampled} sampled steps × "
        f"{n_features} features × {n_permutations} permutations"
    )

    # per_run_avg:  (runs, features, states)   time-averaged ΔMAE per run
    # mean_by_time: (sampled, features, states) mean ΔMAE at each timestep
    per_run_avg = np.zeros((num_runs, n_features, n_state))
    mean_by_time = np.zeros((n_sampled, n_features, n_state))

    # The unperturbed forcing is reused by the baseline and by every state
    # permutation — the large majority of the thousands of solves below — so
    # it is converted once instead of on each call.
    inputs_tensor = ctx.as_tensor(all_inputs_scaled)

    for t_enum, t_idx in enumerate(timestep_indices):
        if t_enum % 10 == 0:
            logger.info(f"  Timestep {t_enum + 1}/{n_sampled} (idx={t_idx})")

        y_t = all_trues_scaled[:, t_idx, :]
        y_tp1_unscaled = ctx.unscale_targets(all_trues_scaled[:, t_idx + 1, :])

        base_pred = single_step_batch(ctx, y_t, t_idx, inputs_tensor)
        base_ae = np.abs(ctx.unscale_targets(base_pred) - y_tp1_unscaled)

        for j in range(n_features):
            deltas_k = np.zeros((n_permutations, num_runs, n_state))

            for k in range(n_permutations):
                perm = rng.permutation(num_runs)

                if j < n_input:
                    # Forcing: swap column j at this timestep only. ZOH means
                    # forcing[:, t_idx, :] governs the whole interval.
                    perturbed_forcing = all_inputs_scaled.copy()
                    perturbed_forcing[:, t_idx, j] = all_inputs_scaled[perm, t_idx, j]
                    pert_pred = single_step_batch(ctx, y_t, t_idx, perturbed_forcing)
                else:
                    # State: swap one isotope in y(t); everything else stays
                    # at ground truth.
                    j_s = j - n_input
                    perturbed_y_t = y_t.copy()
                    perturbed_y_t[:, j_s] = y_t[perm, j_s]
                    pert_pred = single_step_batch(
                        ctx, perturbed_y_t, t_idx, inputs_tensor
                    )

                pert_ae = np.abs(ctx.unscale_targets(pert_pred) - y_tp1_unscaled)
                deltas_k[k] = pert_ae - base_ae

            delta_at_step = deltas_k.mean(axis=0)
            per_run_avg[:, j, :] += delta_at_step / n_sampled
            mean_by_time[t_enum, j, :] = delta_at_step.mean(axis=0)

    mean_delta = per_run_avg.mean(axis=0)
    sem_delta = per_run_avg.std(axis=0, ddof=1) / np.sqrt(num_runs)

    # Normalise to percentages per run, then average, so the spread reflects
    # run-to-run variation rather than the magnitude of the errors.
    delta_clipped = np.clip(per_run_avg, 0.0, None)
    totals = delta_clipped.sum(axis=1, keepdims=True)
    pct_per_run = np.where(
        totals > 0, 100.0 * delta_clipped / np.maximum(totals, 1e-30), 0.0
    )
    mean_pct = pct_per_run.mean(axis=0)
    sem_pct = pct_per_run.std(axis=0, ddof=1) / np.sqrt(num_runs)

    _log_importance(
        mean_delta,
        sem_delta,
        mean_pct,
        sem_pct,
        all_feature_names,
        feature_types,
        target_names,
        n_features,
        log,
    )
    _write_tables(
        ctx.result_dir,
        mean_delta,
        sem_delta,
        mean_pct,
        sem_pct,
        all_feature_names,
        feature_types,
        target_names,
        num_runs,
        n_sampled,
        n_permutations,
    )

    plot.plot_stepwise_importance_bar(
        mean_pct,
        sem_pct,
        all_feature_names,
        feature_types,
        target_names,
        num_runs,
        ctx.result_dir,
    )
    plot.plot_stepwise_importance_over_time(
        mean_by_time,
        t_fractions,
        all_feature_names,
        feature_types,
        target_names,
        ctx.result_dir,
    )

    return mean_delta, sem_delta, mean_pct, sem_pct, mean_by_time


def _log_importance(
    mean_delta,
    sem_delta,
    mean_pct,
    sem_pct,
    feature_names,
    feature_types,
    target_names,
    n_features,
    log,
):
    for k, tname in enumerate(target_names):
        logger.info(f"\n  Per-step importance — target: {tname}")
        logger.info(
            f"    {'Feature':<28} {'Type':<10} "
            f"{'ΔMAE (mean±SEM)':>28} {'Imp. [%]':>18}"
        )
        for j in np.argsort(mean_pct[:, k])[::-1]:
            logger.info(
                f"    {feature_names[j]:<28} {feature_types[j]:<10} "
                f"{mean_delta[j, k]:>10.4e} ± {sem_delta[j, k]:.2e}   "
                f"{mean_pct[j, k]:>7.2f} ± {sem_pct[j, k]:.2f}"
            )
        for j in range(n_features):
            safe = feature_names[j].replace(" ", "_")
            log(f"{tname}/stepwise_pct/{safe}", float(mean_pct[j, k]))
            log(f"{tname}/stepwise_dMAE/{safe}", float(mean_delta[j, k]))


def _write_tables(
    result_dir,
    mean_delta,
    sem_delta,
    mean_pct,
    sem_pct,
    feature_names,
    feature_types,
    target_names,
    num_runs,
    n_sampled,
    n_permutations,
):
    """One CSV per target, plus a single markdown file for the write-up.

    The markdown used to be printed to stdout between copy-paste markers, which
    needed a Unicode-safe print shim to survive a cp1252 Windows console. A file
    is written as UTF-8 regardless, so both the shim and the markers are gone.
    """
    lines = [
        f"_Per-step teacher-forced permutation importance — mean ± SEM across "
        f"{num_runs} runs, {n_sampled} sampled timesteps, K={n_permutations}_",
        "",
    ]

    for k, tname in enumerate(target_names):
        order = np.argsort(mean_pct[:, k])[::-1]

        lines.append(f"#### Target: `{tname}`")
        lines.append("")
        lines.append("| Feature | Type | ΔMAE | MAE Imp. [%] |")
        lines.append("|---|---|---|---|")
        for j in order:
            lines.append(
                f"| {feature_names[j]} | {feature_types[j]} | "
                f"{mean_delta[j, k]:.4e} ± {sem_delta[j, k]:.2e} | "
                f"{mean_pct[j, k]:.2f} ± {sem_pct[j, k]:.2f} |"
            )
        lines.append("")

        output_dir = os.path.join(result_dir, tname)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "stepwise_importance.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Feature",
                    "Type",
                    "dMAE_mean",
                    "dMAE_SEM",
                    "MAE_Imp_pct_mean",
                    "MAE_Imp_pct_SEM",
                ]
            )
            writer.writerows(
                [
                    feature_names[j],
                    feature_types[j],
                    f"{mean_delta[j, k]:.6e}",
                    f"{sem_delta[j, k]:.6e}",
                    f"{mean_pct[j, k]:.4f}",
                    f"{sem_pct[j, k]:.4f}",
                ]
                for j in order
            )
        logger.info(f"  Stepwise importance CSV: {csv_path}")

    table_path = os.path.join(result_dir, TABLE_FILENAME)
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  Stepwise importance tables (markdown): {table_path}")
