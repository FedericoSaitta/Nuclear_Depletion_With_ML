"""Evaluation maths shared by both surrogates, and the report loops built on it.

The `*_curves` / `*_comparison` functions take plain arrays and return plain
numbers, so they can be tested without a Trainer and recomputed from saved
predictions. The `report_*` functions wrap them with the per-target logging and
figures that the two LightningModules used to implement separately.

Array convention throughout: ``(runs, steps, targets)``, in physical units.
"""

from __future__ import annotations

import os

import numpy as np
from loguru import logger

from nuclear_surrogates.utils import metrics, plot

# Well below the smallest physical concentration (~1e-10 atom/b-cm), so it
# regularises log(0) without biasing any real value.
MALE_EPSILON = 1e-20


def deltas_to_absolute(deltas, initial):
    """Integrate per-step concentration changes into absolute concentrations.

    *deltas* is ``(runs, steps)``, *initial* the ``(runs,)`` concentration each
    run starts from.
    """
    return np.asarray(initial)[:, None] + np.cumsum(deltas, axis=1)


def error_growth_curves(trues, ar_preds, tf_preds, epsilon=MALE_EPSILON):
    """How prediction error grows along a trajectory, per target.

    Returns one dict per target, each holding the full per-run error arrays
    ``(runs, steps)`` and their mean/std across runs ``(steps,)``, for both
    absolute error (MAE) and absolute log error (MALE), under teacher forcing
    and autoregressive rollout.
    """
    results = []
    for idx in range(trues.shape[2]):
        gt = trues[:, :, idx]
        ar = ar_preds[:, :, idx]
        tf = tf_preds[:, :, idx]

        tf_mae_errors = np.abs(tf - gt)
        ar_mae_errors = np.abs(ar - gt)

        log_gt = np.log10(np.abs(gt) + epsilon)
        tf_male_errors = np.abs(np.log10(np.abs(tf) + epsilon) - log_gt)
        ar_male_errors = np.abs(np.log10(np.abs(ar) + epsilon) - log_gt)

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
                tf_errors_all=c[f"tf_{key}_errors"],
                ar_errors_all=c[f"ar_{key}_errors"],
            )
        logger.info(f"  {target_name}: error growth plots saved to {output_dir}")
