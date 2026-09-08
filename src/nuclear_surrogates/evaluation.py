"""Evaluation maths shared by both surrogates, and the report loops built on it.

The `*_curves` / `*_comparison` functions take plain arrays and return plain
numbers, so they can be tested without a Trainer and recomputed from saved
predictions. The `report_*` functions wrap them with the per-target logging and
figures that the two LightningModules used to implement separately.

Array convention throughout: ``(runs, steps, targets)``, in physical units.
"""

from __future__ import annotations

import json
import os

import numpy as np
from loguru import logger

from nuclear_surrogates.utils import metrics, plot

# Well below the smallest physical concentration (~1e-10 atom/b-cm), so it
# regularises log(0) without biasing any real value.
MALE_EPSILON = 1e-20

TEST_METRICS_NAME = "test_metrics.json"


def align_to_initial(series, initial):
    """Shift a c(1)…c(N) series onto the c(0)…c(N-1) grid.

    The DNN's targets are one step ahead of its inputs, so every series it
    produces starts at c(1) and ends one step past where the NODE's data stops.
    Prepending the true c(0) and dropping that last step puts both models on the
    same 100 points — which they must be on, since MARE normalises by the
    maximum truth in the window and the error-growth curves index by step.
    """
    return np.concatenate([np.asarray(initial)[:, None], series[:, :-1]], axis=1)


def integrate_deltas(deltas, initial):
    """Integrate per-step concentration changes into absolute concentrations.

    *deltas* is ``(runs, steps)``, *initial* the ``(runs,)`` concentration each
    run starts from. The result starts **at** ``initial`` — c(0), c(1) …
    c(steps-1) — the same grid the NODE reports on, whose trajectory likewise
    begins at the true y(0).
    """
    return align_to_initial(
        np.asarray(initial)[:, None] + np.cumsum(deltas, axis=1), initial
    )


def teacher_forced_from_deltas(deltas, true_conc):
    """Absolute concentrations one step ahead of the **true** state.

    The DNN twin of `analysis.rollout.teacher_forced_predictions`, matched to it
    step for step: the first point is copied from truth because it has no
    predecessor, and every later point is a single prediction added to the true
    concentration rather than to an accumulated one.

    This is what makes MARE(TF) a one-step quantity for both models. Adding the
    deltas to each other instead — c(0) plus every predicted delta — is an
    open-loop integrator whose errors compound along the trajectory, so it
    measures drift rather than one-step accuracy and is not what the NODE's
    teacher-forced number reports.
    """
    return align_to_initial(true_conc + deltas, true_conc[:, 0])


def error_growth_curves(trues, ar_preds, tf_preds, epsilon=MALE_EPSILON):
    """How prediction error grows along a trajectory, per target.

    Returns one dict per target, each holding the full per-run error arrays
    ``(runs, steps)`` and their mean/std across runs ``(steps,)``, for both
    absolute error (MAE) and absolute log error (MALE), under teacher forcing
    and autoregressive rollout.
    """

    def log10_abs(a):
        """log10|a| at double precision.

        MALE subtracts two logs that, on a well-predicted channel, agree to
        about seven digits — so in float32 it is almost pure cancellation.
        U238's is ~5e-6 against logs of magnitude 1.65, a few ULPs, which leaves
        the result at the mercy of the platform's log10 rounding: Windows and
        Linux disagree there by 1-3 ULPs, which is percent-level on the metric.
        Widening the log costs nothing and makes the number reproducible; the
        predictions themselves stay float32, as does the MAE, whose subtraction
        of two nearby floats is exact.
        """
        return np.log10(np.abs(np.asarray(a, dtype=np.float64)) + epsilon)

    results = []
    for idx in range(trues.shape[2]):
        gt = trues[:, :, idx]
        ar = ar_preds[:, :, idx]
        tf = tf_preds[:, :, idx]

        tf_mae_errors = np.abs(tf - gt)
        ar_mae_errors = np.abs(ar - gt)

        log_gt = log10_abs(gt)
        tf_male_errors = np.abs(log10_abs(tf) - log_gt)
        ar_male_errors = np.abs(log10_abs(ar) - log_gt)

        results.append(
            {
                "tf_mae_errors": tf_mae_errors,
                "ar_mae_errors": ar_mae_errors,
                "tf_male_errors": tf_male_errors,
                "ar_male_errors": ar_male_errors,
                "avg_tf_mae": np.mean(tf_mae_errors, axis=0),
                "avg_ar_mae": np.mean(ar_mae_errors, axis=0),
                "std_tf_mae": np.std(tf_mae_errors, axis=0),
                "std_ar_mae": np.std(ar_mae_errors, axis=0),
                "avg_tf_male": np.mean(tf_male_errors, axis=0),
                "avg_ar_male": np.mean(ar_male_errors, axis=0),
                "std_tf_male": np.std(tf_male_errors, axis=0),
                "std_ar_male": np.std(ar_male_errors, axis=0),
            }
        )
    return results


def mare_comparison(trues, ar_preds, tf_preds):
    """Teacher-forced vs autoregressive MARE, per target.

    Note `metrics.mare` is not mean absolute *relative* error despite the name —
    it divides by the global maximum of the truth, not per-sample. Pinned here
    so its meaning cannot change silently.
    """
    results = []
    for idx in range(trues.shape[2]):
        gt = trues[:, :, idx].flatten()
        results.append(
            {
                "mare_tf": float(metrics.mare(gt, tf_preds[:, :, idx].flatten())),
                "mare_ar": float(metrics.mare(gt, ar_preds[:, :, idx].flatten())),
            }
        )
    return results


# ── Report loops (shared by both models) ─────────────────────────────────────


def _banner(title):
    logger.info(f"\n{'=' * 20}")
    logger.info(title)
    logger.info(f"{'=' * 20}")


def report_mare_comparison(
    trues, ar_preds, tf_preds, target_names, log, per_target_metrics=None
):
    """Log teacher-forced vs autoregressive MARE, one entry per target."""
    _banner("MARE: Teacher-Forcing vs Autoregressive")

    comparison = mare_comparison(trues, ar_preds, tf_preds)
    for idx, target_name in enumerate(target_names):
        mare_tf = comparison[idx]["mare_tf"]
        mare_ar = comparison[idx]["mare_ar"]

        log(f"{target_name}/MARE_TeacherForcing", float(mare_tf))
        log(f"{target_name}/MARE_Autoregressive", float(mare_ar))
        logger.info(f"  {target_name}: MARE(TF)={mare_tf:.6f}, MARE(AR)={mare_ar:.6f}")

        if per_target_metrics is not None:
            per_target_metrics[idx]["mare_tf"] = float(mare_tf)
            per_target_metrics[idx]["mare_ar"] = float(mare_ar)


def report_prediction_comparisons(
    trues, ar_preds, tf_preds, target_names, result_dir, run=0
):
    """Plot teacher-forced vs autoregressive predictions for one run."""
    _banner("PREDICTION COMPARISON (Teacher-Forcing vs Autoregressive)")

    for idx, target_name in enumerate(target_names):
        output_dir = os.path.join(result_dir, target_name)
        plot.plot_prediction_comparison(
            trues[run, :, idx],
            tf_preds[run, :, idx],
            ar_preds[run, :, idx],
            target_name,
            output_dir,
        )
        logger.info(f"  {target_name}: comparison plot saved to {output_dir}")


def report_per_target(
    trues, preds, target_names, result_dir, log, steps_per_run, figures=True
):
    """Per-target MAE / RMSE / R², the scatter and the residual figures.

    *figures* is the one place `--no-analyses` has to reach inside a report
    function rather than gate it from outside: this one computes the metrics a
    run always needs and draws figures it sometimes does not.

    Both models reduce to a 2-D ``(samples, targets)`` array before this point —
    the DNN from its stacked test batches, the NODE by flattening runs and steps
    — so one implementation serves both. It used to be `_plot_single_output` plus
    `_compute_and_log_overall_metrics` on one side and the whole of
    `_report_pointwise_metrics` on the other.

    Returns the `per_target_metrics` list the callers go on to enrich with the
    MARE comparison and write to `test_metrics.json`.
    """
    mae_per_output = metrics.mae(trues, preds)
    rmse_per_output = metrics.rmse(trues, preds)
    r2_per_output = metrics.r2(trues, preds)

    log("Mean Absolute Error (avg)", float(mae_per_output.mean()))
    log("Root Mean Squared Error (avg)", float(rmse_per_output.mean()))
    log("R-squared coefficient (avg)", float(r2_per_output.mean()))

    logger.info("TEST SET — UNSCALED metrics:")
    logger.info(f"  R² (avg):   {r2_per_output.mean():.6f}")
    logger.info(f"  RMSE (avg): {rmse_per_output.mean():.6f}")
    logger.info(f"  MAE (avg):  {mae_per_output.mean():.6f}")

    per_target_metrics = []
    for idx, target_name in enumerate(target_names):
        if figures:
            output_dir = os.path.join(result_dir, target_name)
            os.makedirs(output_dir, exist_ok=True)

            plot.plot_predictions_vs_actuals(
                trues[:, idx],
                preds[:, idx],
                mae_per_output[idx],
                rmse_per_output[idx],
                r2_per_output[idx],
                output_dir,
            )
            plot.plot_residuals_combined(
                trues[:, idx], preds[:, idx], output_dir, steps_per_run=steps_per_run
            )

        per_target_metrics.append(
            {
                "name": target_name,
                "mae": float(mae_per_output[idx]),
                "rmse": float(rmse_per_output[idx]),
                "r2": float(r2_per_output[idx]),
            }
        )

    return per_target_metrics, mae_per_output, rmse_per_output, r2_per_output


def report_trajectories(
    t,
    trues,
    ar_preds,
    forcing,
    target_names,
    result_dir,
    xlabel="Time",
    num_examples=2,
):
    """A couple of individual test trajectories per target, plus an all-runs overlay.

    *t* is the x-axis and *xlabel* names it, because the two models measure it
    differently: the NODE integrates over a real time span, while the DNN is a
    one-step map with no notion of elapsed time and can only count steps. That
    distinction is worth showing on the axis rather than hiding behind a shared
    default.
    """
    num_runs = trues.shape[0]

    for target_idx, target_name in enumerate(target_names):
        target_dir = os.path.join(result_dir, target_name)
        os.makedirs(target_dir, exist_ok=True)

        for i in range(min(num_examples, num_runs)):
            plot.plot_trajectory(
                t,
                ar_preds[i, :, target_idx],
                trues[i, :, target_idx],
                forcing[i],
                title=f"{target_name} — Test Trajectory {i + 1}",
                save_path=os.path.join(target_dir, f"test_traj_{i + 1}.png"),
                xlabel=xlabel,
            )

        plot.plot_trajectory_summary(
            t,
            ar_preds[:, :, target_idx],
            trues[:, :, target_idx],
            title=f"{target_name} — All Test Trajectories ({num_runs} runs)",
            save_path=os.path.join(target_dir, "test_all_trajectories.png"),
            xlabel=xlabel,
        )


def write_test_metrics(result_dir, test_metrics):
    """Write the run's test metrics to `<result_dir>/test_metrics.json`.

    The per-target numbers are the manuscript's headline results, and this is
    the only place they are recorded in machine-readable form: the bundle is
    written before `trainer.test` runs, so `metadata.json` carries the
    *validation* metrics and cannot carry these.

    Best-effort, like the bundle — a finished evaluation must not be lost to a
    failed write.
    """
    path = os.path.join(result_dir, TEST_METRICS_NAME)
    try:
        os.makedirs(result_dir, exist_ok=True)
        with open(path, "w") as f:
            # allow_nan=False: `json.dump` writes a bare `Infinity` for a metric
            # that came out non-finite, which Python reads back and nothing else
            # does. Same rule the bundle's metadata.json follows.
            json.dump(test_metrics, f, indent=2, allow_nan=False)
    except (OSError, ValueError) as exc:
        logger.error(f"Failed to write {path}: {exc}")
        return
    logger.info(f"Test metrics written to {path}")


def report_error_growth(trues, ar_preds, tf_preds, target_names, result_dir, log):
    """Plot how error grows along the trajectory, per target and metric."""
    _banner("ERROR GROWTH ANALYSIS (Teacher-Forcing vs Autoregressive)")

    num_runs = trues.shape[0]
    logger.info(f"Analyzing error growth across {num_runs} runs")

    curves = error_growth_curves(trues, ar_preds, tf_preds)
    for idx, target_name in enumerate(target_names):
        c = curves[idx]
        log(f"{target_name}/Final MALE (AR)", float(c["avg_ar_male"][-1]))
        log(f"{target_name}/Final MALE (TF)", float(c["avg_tf_male"][-1]))

        output_dir = os.path.join(result_dir, target_name)
        for metric_name, ylabel in (
            ("MAE", "Mean Absolute Error"),
            ("MALE", "Mean Absolute Log Error"),
        ):
            key = metric_name.lower()
            plot.plot_error_growth_metric(
                c[f"avg_tf_{key}"],
                c[f"avg_ar_{key}"],
                c[f"std_tf_{key}"],
                c[f"std_ar_{key}"],
                target_name,
                output_dir,
                num_runs,
                metric_name=metric_name,
                ylabel=ylabel,
                skip_first_n=0,
            )
        logger.info(f"  {target_name}: error growth plots saved to {output_dir}")
