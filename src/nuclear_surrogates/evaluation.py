"""Pure evaluation maths, extracted from the LightningModules.

The numbers behind the paper's error-growth and MARE figures used to be computed
inline inside methods that also call `self.log` and write PNGs — so they needed
a Trainer, a datamodule and a writable output directory to run at all. That made
them untestable and impossible to recompute from saved predictions.

These functions take plain arrays and return plain numbers. The model methods
now call them and are left with logging and drawing. The per-target loop is kept
exactly as it was, rather than vectorised, so the values are bit-for-bit what
the previous inline code produced.

Array convention throughout: ``(runs, steps, targets)``, in physical units.
"""

from __future__ import annotations

import numpy as np

from nuclear_surrogates.utils import metrics

# Well below the smallest physical concentration (~1e-10 atom/b-cm), so it
# regularises log(0) without biasing any real value.
MALE_EPSILON = 1e-20


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
    it divides by the global maximum of the truth, not per-sample
    (AUDIT.md §3.7). Pinned here so its meaning cannot change silently.
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
