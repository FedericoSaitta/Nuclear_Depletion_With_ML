"""Metrics, permutation importance, and the DNN autoregressive rollout.

The per-output metrics wrap scikit-learn rather than re-deriving the formulas;
the wrappers exist to fix `multioutput="raw_values"` (every caller wants the
per-target array) and to log loudly when a result contains NaN or Inf, which
sklearn passes through silently. `mare` stays hand-rolled: it is this project's
own (mis)named quantity — see its docstring and AUDIT.md P7.
"""

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from tqdm import tqdm

from nuclear_surrogates.datamodule.dataset_helper import ensure_2d

# ── Metric helpers ───────────────────────────────────────────────────────────


def _validate(name, result):
    """Log an error if any metric values are NaN or Inf."""
    if np.any(np.isnan(result)):
        logger.error(f"{name}: result contains NaN values: {result}")
    if np.any(np.isinf(result)):
        logger.error(f"{name}: result contains Inf values: {result}")
    return result


def _metric(name, result):
    """Ensure 1-D array and validate."""
    return _validate(name, np.atleast_1d(result))


# ── Per-output metrics (always return arrays) ────────────────────────────────


def mae(y_true, y_pred):
    """Mean Absolute Error per output."""
    return _metric("MAE", mean_absolute_error(y_true, y_pred, multioutput="raw_values"))


def mse(y_true, y_pred):
    """Mean Squared Error per output."""
    return _metric("MSE", mean_squared_error(y_true, y_pred, multioutput="raw_values"))


def rmse(y_true, y_pred):
    """Root Mean Squared Error per output."""
    return _metric(
        "RMSE", root_mean_squared_error(y_true, y_pred, multioutput="raw_values")
    )


def r2(y_true, y_pred):
    """R² (Coefficient of Determination) per output.

    sklearn's `force_finite` default already handles a zero-variance truth
    column (0.0 for an imperfect fit, 1.0 for a perfect one) instead of
    dividing by zero.
    """
    return _metric("R2", r2_score(y_true, y_pred, multioutput="raw_values"))


def mare(y_true, y_pred):
    """Mean absolute error normalised by the largest |truth| (scalar).

    Despite the name this is NOT mean absolute *relative* error: it divides by a
    single global maximum, not per-sample. Published numbers depend on it, so
    the definition must not drift — see AUDIT.md.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    max_abs = np.max(np.abs(y_true))
    result = np.mean(np.abs(y_true - y_pred) / max_abs) if max_abs > 0 else float("nan")
    return _metric("MARE", result).item()


# ── Model inference helpers ──────────────────────────────────────────────────


def _collect_loader(loader):
    """Concatenate all batches from a DataLoader into numpy arrays."""
    xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in loader], strict=False)
    return np.concatenate(xs), ensure_2d(np.concatenate(ys))


@torch.no_grad()
def _predict_numpy(model, x_np, device):
    """Run model inference on a numpy array and return 2-D numpy output."""
    tensor = torch.FloatTensor(x_np).to(device)
    return ensure_2d(model(tensor).cpu().numpy())


# ── Feature importance (permutation-based) ───────────────────────────────────

METRIC_REGISTRY = {
    "r2": {"fn": lambda yt, yp: r2(yt, yp).mean(), "higher_is_better": True},
    "mae": {"fn": lambda yt, yp: mae(yt, yp).mean(), "higher_is_better": False},
    "mse": {"fn": lambda yt, yp: mse(yt, yp).mean(), "higher_is_better": False},
    "rmse": {"fn": lambda yt, yp: rmse(yt, yp).mean(), "higher_is_better": False},
}


def calculate_feature_importance(
    model,
    test_loader,
    device,
    n_repeats=10,
    metric=None,
    output_idx=None,
    seed=0,
):
    """Permutation feature importance.

    *seed* fixes the permutations so the importance figures are reproducible
    across re-runs; previously they changed on every invocation.
    """
    if metric is None:
        metric = {"name": "r2", "direction": "increasing"}
    rng = np.random.default_rng(seed)
    X_test, y_test = _collect_loader(test_loader)

    if output_idx is not None:
        y_test = y_test[:, output_idx : output_idx + 1]

    # Resolve metric
    metric_name = metric["name"].lower()
    metric_direction = metric["direction"].lower()
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unsupported metric: {metric_name}. Choose from {list(METRIC_REGISTRY)}"
        )
    if metric_direction not in ("increasing", "decreasing"):
        raise ValueError("Direction must be 'increasing' or 'decreasing'")

    metric_fn = METRIC_REGISTRY[metric_name]["fn"]
    sign = 1 if metric_direction == "increasing" else -1

    # Baseline
    model.eval()
    preds = _predict_numpy(model, X_test, device)
    if output_idx is not None:
        preds = preds[:, output_idx : output_idx + 1]
    baseline_score = metric_fn(y_test, preds)

    label = f"output {output_idx}" if output_idx is not None else "all outputs"
    logger.info(
        f"Feature importance ({label}) — {metric_name} ({metric_direction}), baseline={baseline_score:.4f}"
    )

    # Permutation loop. One scratch copy is reused across every (feature,
    # repeat) instead of copying the whole test set each time. The column is
    # reset to its original values before every shuffle — shuffling an already
    # shuffled column would compose two permutations and give a different
    # arrangement than a fresh copy does — and restored once the feature is done,
    # so the rest of the array stays pristine. Restoring a column touches neither
    # the other columns nor the generator state, so the permutation sequence is
    # unchanged.
    n_features = X_test.shape[1]
    importances = np.zeros((n_features, n_repeats))
    X_perm = X_test.copy()

    for feat in range(n_features):
        original_column = X_test[:, feat].copy()
        for rep in range(n_repeats):
            X_perm[:, feat] = original_column
            rng.shuffle(X_perm[:, feat])

            perm_preds = _predict_numpy(model, X_perm, device)
            if output_idx is not None:
                perm_preds = perm_preds[:, output_idx : output_idx + 1]

            importances[feat, rep] = sign * (
                baseline_score - metric_fn(y_test, perm_preds)
            )
        X_perm[:, feat] = original_column

    return importances.mean(axis=1), importances.std(axis=1), baseline_score


# ── Autoregressive rollout ───────────────────────────────────────────────────


def model_autoregress(
    model,
    X_data,
    Y_data,
    x_scaler,
    y_scaler,
    steps_per_run,
    inputs_indices,
    target_col_indices,
    delta_conc,
):
    """Roll the model forward on its own predictions, one step at a time.

    Only the step axis is sequential — runs are independent — so every run is
    advanced together and the model, the scalers and the feedback write all see
    whole `(runs, ...)` arrays instead of one row at a time.

    Returns two {target: flat array} dicts in run-major order, matching the
    layout the caller reshapes back to `(runs, steps)`.
    """
    # Work on a copy: the rollout feeds predictions back into the next
    # timestep's inputs, and the caller reuses this same array afterwards for
    # the comparison and error-growth figures.
    X_data = np.array(X_data, copy=True)

    n_runs = len(X_data) // steps_per_run
    n_features = X_data.shape[1]
    target_names = list(target_col_indices.keys())

    # Views onto the same buffers, addressed as (runs, steps, features).
    X = X_data.reshape(n_runs, steps_per_run, n_features)
    Y = Y_data.reshape(n_runs, steps_per_run, -1)
    unscaled_x = x_scaler.inverse_transform(X_data).reshape(
        n_runs, steps_per_run, n_features
    )

    # Targets that are also inputs are the ones fed back; the rest of the input
    # row (power, temperatures, boron) keeps its true value throughout.
    fed_back = [
        (n, target_col_indices[n], inputs_indices[n])
        for n in target_names
        if n in inputs_indices
    ]
    concentrations = (
        {n: unscaled_x[:, 0, in_idx].copy() for n, _, in_idx in fed_back}
        if delta_conc
        else {}
    )

    logger.info(f"Autoregressive: {n_runs} runs × {steps_per_run} steps")

    # Collected per step and stacked, so the result keeps the model's dtype
    # rather than being widened by a preallocated float64 buffer.
    predictions, ground_truth = [], []

    model.eval()
    device = next(model.parameters()).device

    for t in tqdm(range(steps_per_run), desc="Autoregressive MARE", unit="step"):
        pred = y_scaler.inverse_transform(_predict_numpy(model, X[:, t, :], device))
        predictions.append(pred)
        ground_truth.append(y_scaler.inverse_transform(Y[:, t, :]))

        if t == steps_per_run - 1:
            break

        for name, target_idx, input_idx in fed_back:
            value = pred[:, target_idx]
            if delta_conc:
                concentrations[name] += value
                value = concentrations[name]
            unscaled_x[:, t + 1, input_idx] = value

        X[:, t + 1, :] = x_scaler.transform(unscaled_x[:, t + 1, :])

    # (steps, runs, targets) -> (runs, steps, targets), then flatten run-major,
    # which is the order the caller reshapes back to (runs, steps).
    predictions = np.stack(predictions, axis=1)
    ground_truth = np.stack(ground_truth, axis=1)

    predictions_dict = {
        name: predictions[:, :, idx].reshape(-1)
        for name, idx in target_col_indices.items()
    }
    ground_truth_dict = {
        name: ground_truth[:, :, idx].reshape(-1)
        for name, idx in target_col_indices.items()
    }

    logger.info(f"Collected predictions for targets: {target_names}")
    return predictions_dict, ground_truth_dict
