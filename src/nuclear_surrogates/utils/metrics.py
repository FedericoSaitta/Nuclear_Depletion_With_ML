# Functions to calculate metrics and probe the performance of ML models
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

import nuclear_surrogates.datamodule.data_scalers as data_scaler
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
    return _metric("MAE", np.mean(np.abs(y_true - y_pred), axis=0))


def mse(y_true, y_pred):
    """Mean Squared Error per output."""
    return _metric("MSE", np.mean((y_true - y_pred) ** 2, axis=0))


def rmse(y_true, y_pred):
    """Root Mean Squared Error per output."""
    return _metric("RMSE", np.sqrt(mse(y_true, y_pred)))


def r2(y_true, y_pred):
    """R² (Coefficient of Determination) per output."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    return _metric("R2", np.where(ss_tot != 0, 1 - ss_res / ss_tot, 0.0))


def mare(y_true, y_pred):
    """Max-Absolute-Range Error (scalar)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    max_abs = np.max(np.abs(y_true))
    result = np.mean(np.abs(y_true - y_pred) / max_abs) if max_abs > 0 else float("nan")
    return _metric("MARE", result).item()


# ── Model inference helpers ──────────────────────────────────────────────────


def _collect_loader(loader):
    """Concatenate all batches from a DataLoader into numpy arrays."""
    xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in loader])
    return np.concatenate(xs), ensure_2d(np.concatenate(ys))


@torch.no_grad()
def _predict_numpy(model, x_np, device):
    """Run model inference on a numpy array and return 2-D numpy output."""
    tensor = torch.FloatTensor(x_np).to(device)
    return ensure_2d(model(tensor).cpu().numpy())


def get_model_prediction(model, x_input, y_scaler):
    """Single-sample prediction, inverse-scaled to original units."""
    x_input = np.atleast_2d(x_input)  # (13,) → (1, 13)
    device = next(model.parameters()).device

    model.eval()
    pred_scaled = _predict_numpy(model, x_input, device)
    return data_scaler.inverse_transformer(y_scaler, pred_scaled)


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
    metric={"name": "r2", "direction": "increasing"},
    output_idx=None,
    seed=0,
):
    """Permutation feature importance.

    *seed* fixes the permutations so the importance figures are reproducible
    across re-runs; previously they changed on every invocation.
    """
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

    # Permutation loop
    n_features = X_test.shape[1]
    importances = np.zeros((n_features, n_repeats))

    for feat in range(n_features):
        for rep in range(n_repeats):
            X_perm = X_test.copy()
            rng.shuffle(X_perm[:, feat])

            perm_preds = _predict_numpy(model, X_perm, device)
            if output_idx is not None:
                perm_preds = perm_preds[:, output_idx : output_idx + 1]

            importances[feat, rep] = sign * (
                baseline_score - metric_fn(y_test, perm_preds)
            )

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
    # Work on a copy: the rollout feeds predictions back into the next
    # timestep's inputs, and the caller reuses this same array afterwards for
    # the comparison and error-growth figures (AUDIT Pass 2 §A8).
    X_data = np.array(X_data, copy=True)

    total_samples = len(X_data)
    n_runs = total_samples // steps_per_run

    unscaled_x = data_scaler.inverse_transformer(x_scaler, X_data)
    target_names = list(target_col_indices.keys())

    logger.info(f"Autoregressive: {n_runs} runs × {steps_per_run} steps")

    predictions_dict = {name: [] for name in target_names}
    ground_truth_dict = {name: [] for name in target_names}

    for run in tqdm(range(n_runs), desc="Autoregressive MARE", unit="run"):
        start = run * steps_per_run
        end = start + steps_per_run

        # Initialise concentrations for delta mode
        concentrations = {}
        if delta_conc:
            init_x = data_scaler.inverse_transformer(
                x_scaler, X_data[start].reshape(1, -1)
            )[0]
            concentrations = {
                name: init_x[inputs_indices[name]]
                for name in target_names
                if name in inputs_indices
            }

        for t in range(start, end):
            pred = get_model_prediction(model, X_data[t], y_scaler)
            gt = data_scaler.inverse_transformer(y_scaler, Y_data[t].reshape(1, -1))

            for name, idx in target_col_indices.items():
                predictions_dict[name].append(pred[0, idx])
                ground_truth_dict[name].append(gt[0, idx])

            # Feed prediction back into next timestep
            if t < end - 1:
                _update_next_input(
                    unscaled_x,
                    X_data,
                    x_scaler,
                    t,
                    pred,
                    concentrations,
                    inputs_indices,
                    target_col_indices,
                    delta_conc,
                )

    # Convert to arrays
    predictions_dict = {k: np.array(v) for k, v in predictions_dict.items()}
    ground_truth_dict = {k: np.array(v) for k, v in ground_truth_dict.items()}

    logger.info(f"Collected predictions for targets: {target_names}")
    return predictions_dict, ground_truth_dict


def _update_next_input(
    unscaled_x,
    X_data,
    x_scaler,
    t,
    pred,
    concentrations,
    inputs_indices,
    target_col_indices,
    delta_conc,
):
    """Write model output back into the next timestep's input features."""
    for name, target_idx in target_col_indices.items():
        if name not in inputs_indices:
            continue

        value = pred[0, target_idx]
        if delta_conc:
            concentrations[name] += value
            value = concentrations[name]

        unscaled_x[t + 1, inputs_indices[name]] = value

    X_data[t + 1] = x_scaler.transform(unscaled_x[t + 1].reshape(1, -1))[0]
