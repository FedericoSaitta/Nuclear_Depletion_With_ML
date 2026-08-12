"""Every figure the surrogates produce.

Models collect arrays; these functions draw them. Keeping the two apart is what
lets the evaluation code be tested without a display or a Trainer.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.patches import Patch

from nuclear_surrogates.utils import metrics

STATE_COLOR = "#2196F3"
FORCING_COLOR = "#FF9800"
TF_COLOR = "#2E86AB"
AR_COLOR = "#A23B72"


def _save(fig, path, message, dpi=300):
    """Write *fig* to *path*, close it, and say where it went."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"{message}: {path}")


# ── Data and training ────────────────────────────────────────────────────────


def plot_data_distributions(X, col_index_map, save_dir=None, name="Raw_Data"):
    """Histogram per feature, with the mean marked."""
    feature_list = sorted(col_index_map.items(), key=lambda x: x[1])
    n_features = len(feature_list)

    if n_features == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        axes = [ax]
    else:
        n_cols = 4
        n_rows = int(np.ceil(n_features / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()

    for idx, (feature_name, col_idx) in enumerate(feature_list):
        ax = axes[idx]
        data = X if X.ndim == 1 else X[:, col_idx]

        _, _, patches = ax.hist(
            data, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5
        )
        for patch in patches:
            patch.set_facecolor("steelblue")

        mean_val = np.mean(data)
        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.3e}",
        )
        ax.set_title(feature_name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"{name} - Feature Distributions", fontsize=16, fontweight="bold", y=1.0
    )
    fig.tight_layout()
    _save(
        fig,
        os.path.join(save_dir, f"{name}_distribution_features.png"),
        "Feature distributions saved to",
    )


def plot_losses(train_losses, val_losses, save_dir, nfes=None):
    """Log10 train/validation loss curves, with the NFE count for the NODE."""
    eps = 1e-10
    fig, ax_loss = plt.subplots(figsize=(10, 6))

    ax_loss.plot(
        np.log10(np.array(train_losses) + eps),
        label="Training Loss (log10)",
        linewidth=2,
        color=TF_COLOR,
    )
    ax_loss.plot(
        np.log10(np.array(val_losses) + eps),
        label="Validation Loss (log10)",
        linewidth=2,
        color=AR_COLOR,
    )
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Log10(Loss)", fontsize=12)
    ax_loss.set_title("Training and Validation Loss Over Time", fontsize=14)
    ax_loss.grid(True, alpha=0.3)

    if nfes and len(nfes) == len(train_losses):
        ax_nfe = ax_loss.twinx()
        ax_nfe.plot(
            nfes,
            label="NFE",
            linewidth=1.5,
            color=FORCING_COLOR,
            linestyle="--",
            alpha=0.8,
        )
        ax_nfe.set_ylabel(
            "Function Evaluations (NFE)", fontsize=12, color=FORCING_COLOR
        )
        ax_nfe.tick_params(axis="y", labelcolor=FORCING_COLOR)

        lines, labels = ax_loss.get_legend_handles_labels()
        nfe_lines, nfe_labels = ax_nfe.get_legend_handles_labels()
        ax_loss.legend(
            lines + nfe_lines, labels + nfe_labels, fontsize=10, loc="upper right"
        )
    else:
        ax_loss.legend(fontsize=10)

    fig.tight_layout()
    _save(
        fig,
        os.path.join(save_dir, "training_loss_log.png"),
        "Logarithmic train and validation losses saved to",
    )


# ── Predictions ──────────────────────────────────────────────────────────────


def plot_predictions_vs_actuals(actuals, predictions, mae, rmse, r2, plots_folder):
    """Scatter of prediction against truth, with the identity line."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(actuals, predictions, alpha=0.5, s=20, edgecolors="k", linewidth=0.5)

    lo = min(actuals.min(), predictions.min())
    hi = max(actuals.max(), predictions.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect Prediction")

    ax.set_xlabel("Actual Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.set_title(
        f"Predictions vs Actual Values\n"
        f"R² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(
        fig,
        os.path.join(plots_folder, "predictions_vs_actual.png"),
        "Model predictions saved to",
    )


def _within_run_deltas(values, steps_per_run):
    """Successive differences, excluding those that would cross a run boundary."""
    n_runs = len(values) // steps_per_run
    deltas = np.diff(values)
    keep = np.ones(len(values) - 1, dtype=bool)
    for r in range(1, n_runs):
        keep[r * steps_per_run - 1] = False
    return deltas[keep]


def _plot_residuals_linear(delta_predictions, residuals, plots_folder):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(
        delta_predictions, residuals, alpha=0.5, s=20, edgecolors="k", linewidth=0.5
    )
    axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Predicted Δc", fontsize=12)
    axes[0].set_ylabel("Residuals (Actual Δc - Predicted Δc)", fontsize=12)
    axes[0].set_title("Concentration Change Residuals", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Residuals (Δc)", fontsize=12)
    axes[1].set_title(
        f"Δc Residual Distribution\n"
        f"Mean: {residuals.mean():.4e}, Std: {residuals.std():.4e}",
        fontsize=14,
    )
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save(
        fig,
        os.path.join(plots_folder, "residuals_combined.png"),
        "Concentration-change residuals (linear scale) saved to",
    )


def _plot_residuals_loglog(delta_predictions, residuals, plots_folder):
    """Same residuals on log-log axes, which needs the zeros filtered out."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    abs_deltas = np.abs(delta_predictions)
    abs_residuals = np.abs(residuals)
    nonzero_residual = abs_residuals > 0
    plottable = (abs_deltas > 0) & nonzero_residual

    if not np.any(plottable):
        for ax, message in zip(
            axes,
            (
                "Insufficient data for log-log plot\n(need non-zero Δc and residuals)",
                "Insufficient data for log-log plot",
            ),
            strict=False,
        ):
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    else:
        for mask, color, label in (
            (delta_predictions > 0, "blue", "Positive Δc"),
            (delta_predictions < 0, "red", "Negative Δc"),
        ):
            selected = mask & plottable
            if np.any(selected):
                axes[0].scatter(
                    abs_deltas[selected],
                    abs_residuals[selected],
                    alpha=0.5,
                    s=20,
                    edgecolors="k",
                    linewidth=0.5,
                    color=color,
                    label=f"{label} ({np.sum(selected)} pts)",
                )

        # Reference line: |residual| = |prediction|, i.e. 100% error.
        span = [abs_deltas[plottable].min(), abs_deltas[plottable].max()]
        axes[0].plot(span, span, "k--", alpha=0.3, linewidth=1, label="100% error line")

        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("|Predicted Δc| (log scale)", fontsize=12)
        axes[0].set_ylabel("|Residuals| (log scale)", fontsize=12)
        axes[0].set_title("Δc Residuals (Log-Log Scale)", fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, which="both", alpha=0.3)

        valid = abs_residuals[nonzero_residual]
        axes[1].hist(
            valid,
            bins=np.logspace(np.log10(valid.min()), np.log10(valid.max()), 50),
            edgecolor="black",
            alpha=0.7,
            color="steelblue",
        )
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("|Δc Residuals| (log scale)", fontsize=12)
        axes[1].set_ylabel("Frequency (log scale)", fontsize=12)
        axes[1].set_title(
            f"|Δc Residual| Distribution (Log Scale)\n"
            f"Median: {np.median(valid):.4e}, Mean: {np.mean(valid):.4e}",
            fontsize=14,
        )
        axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    _save(
        fig,
        os.path.join(plots_folder, "residuals_combined_loglog.png"),
        "Concentration-change residuals (log-log scale) saved to",
    )


def plot_residuals_combined(actuals, predictions, plots_folder, steps_per_run):
    """Residuals of the concentration *change*, on linear and log-log axes."""
    delta_actuals = _within_run_deltas(actuals, steps_per_run)
    delta_predictions = _within_run_deltas(predictions, steps_per_run)
    residuals = delta_actuals - delta_predictions

    _plot_residuals_linear(delta_predictions, residuals, plots_folder)
    _plot_residuals_loglog(delta_predictions, residuals, plots_folder)


def plot_prediction_comparison(
    ground_truth, teacher_forced_preds, autoregressive_preds, target_name, result_dir
):
    """Truth vs teacher-forced vs autoregressive for one run, over its residuals."""
    mare_teacher = metrics.mare(ground_truth, teacher_forced_preds)
    mare_autoregressive = metrics.mare(ground_truth, autoregressive_preds)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    # Shifted by one: t=0 is the initial condition, never a prediction.
    time_steps = np.arange(len(ground_truth)) + 1

    ax1.plot(
        time_steps,
        ground_truth,
        "k-",
        label="Ground Truth",
        linewidth=2.5,
        alpha=0.9,
        zorder=1,
    )
    for values, color, marker, label in (
        (
            teacher_forced_preds,
            "blue",
            "s",
            f"Teacher-Forced (MARE: {mare_teacher:.4f})",
        ),
        (
            autoregressive_preds,
            "red",
            "^",
            f"Autoregressive (MARE: {mare_autoregressive:.4f})",
        ),
    ):
        ax1.plot(time_steps, values, color=color, linewidth=2, alpha=0.8, zorder=2)
        ax1.plot(
            time_steps,
            values,
            marker,
            color=color,
            markersize=5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=label,
            zorder=3,
        )

    ax1.set_ylabel(f"{target_name}", fontsize=14, fontweight="bold")
    ax1.set_title(
        f"{target_name} - Prediction Comparison", fontsize=16, fontweight="bold", pad=15
    )
    ax1.legend(loc="best", fontsize=12, framealpha=0.95, edgecolor="black")
    ax1.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)
    ax1.tick_params(labelbottom=False, labelsize=11)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for preds, color, marker, label in (
        (teacher_forced_preds, "blue", "s", "Teacher-Forced"),
        (autoregressive_preds, "red", "^", "Autoregressive"),
    ):
        residuals = ground_truth - preds
        ax2.plot(time_steps, residuals, color=color, linewidth=1.8, alpha=0.8, zorder=2)
        ax2.plot(
            time_steps,
            residuals,
            marker,
            color=color,
            markersize=5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=label,
            zorder=3,
        )

    ax2.axhline(y=0, color="k", linestyle="--", linewidth=1.5, alpha=0.6, zorder=1)
    ax2.set_ylabel("Residuals", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time Steps", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.4, linestyle="--", linewidth=0.8)
    ax2.legend(loc="best", fontsize=10, framealpha=0.95, edgecolor="black")
    ax2.tick_params(labelsize=11)

    _save(
        fig,
        os.path.join(result_dir, f"{target_name}_prediction_comparison.png"),
        f"{target_name} prediction comparison saved to",
    )


def plot_error_growth_metric(
    avg_tf_error,
    avg_ar_error,
    std_tf_error,
    std_ar_error,
    target_name,
    output_dir,
    num_runs,
    metric_name="MALE",
    ylabel="Error",
    skip_first_n=0,
    tf_errors_all=None,
    ar_errors_all=None,
):
    """Error against timestep, drawn once linear and once log-scaled."""
    os.makedirs(output_dir, exist_ok=True)

    if skip_first_n > 0:
        time_steps = np.arange(skip_first_n, len(avg_tf_error))
        avg_tf_error = avg_tf_error[skip_first_n:]
        avg_ar_error = avg_ar_error[skip_first_n:]
        std_tf_error = std_tf_error[skip_first_n:]
        std_ar_error = std_ar_error[skip_first_n:]
        if tf_errors_all is not None:
            tf_errors_all = tf_errors_all[:, skip_first_n:]
            ar_errors_all = ar_errors_all[:, skip_first_n:]
    else:
        time_steps = np.arange(len(avg_tf_error))

    have_raw = tf_errors_all is not None and ar_errors_all is not None

    for scale_type in ("linear", "log"):
        fig, ax = plt.subplots(figsize=(12, 7))

        for avg, color, label in (
            (avg_tf_error, TF_COLOR, "Teacher-Forcing (mean)"),
            (avg_ar_error, AR_COLOR, "Autoregressive (mean)"),
        ):
            ax.plot(
                time_steps,
                avg,
                label=label,
                color=color,
                linewidth=2.5,
                alpha=0.9,
                zorder=5,
            )

        for avg, std, raw, color, tag in (
            (avg_tf_error, std_tf_error, tf_errors_all, TF_COLOR, "TF"),
            (avg_ar_error, std_ar_error, ar_errors_all, AR_COLOR, "AR"),
        ):
            if scale_type == "linear":
                lower, upper, band = avg - std, avg + std, f"{tag} ±1σ"
            elif have_raw:
                # Percentiles, not mean ± std: an arithmetic band is meaningless
                # once the axis is logarithmic.
                lower = np.percentile(raw, 16, axis=0)
                upper = np.percentile(raw, 84, axis=0)
                band = f"{tag} 16-84%ile"
            else:
                # Geometric fallback — approximate, but better than arithmetic.
                factor = np.exp(std / (avg + 1e-20))
                lower, upper, band = avg / factor, avg * factor, f"{tag} approx. ±1σ"
            ax.fill_between(
                time_steps, lower, upper, alpha=0.2, color=color, label=band
            )

        ax.set_xlabel("Time Step", fontsize=13, fontweight="bold")
        if scale_type == "log":
            ax.set_yscale("log")
            ax.set_ylabel(f"{ylabel} (log scale)", fontsize=13, fontweight="bold")
        else:
            ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")

        ax.set_title(
            f"{metric_name} Growth Over Time: {target_name}"
            f"{' (Log Scale)' if scale_type == 'log' else ''}\n"
            f"(Averaged over {num_runs} runs — final AR {metric_name}: "
            f"{float(avg_ar_error[-1]):.4f})",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.legend(fontsize=10, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

        _save(
            fig,
            os.path.join(
                output_dir, f"{target_name}_{metric_name}_growth_{scale_type}.png"
            ),
            f"{metric_name} growth plot ({scale_type} scale) saved to",
        )


def plot_feature_importance(
    importance_means,
    importance_stds,
    feature_names,
    baseline,
    plots_folder,
    metric_name,
    n_top=20,
):
    """Horizontal bar chart of the top-N permutation importances."""
    n_top = min(n_top, len(importance_means))
    top_indices = np.argsort(importance_means)[::-1][:n_top]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(
        range(n_top),
        importance_means[top_indices],
        xerr=importance_stds[top_indices],
        align="center",
        alpha=0.8,
        edgecolor="black",
        color=plt.cm.viridis(np.linspace(0.3, 0.9, n_top)),
    )

    for position, idx in enumerate(top_indices):
        value, std = importance_means[idx], importance_stds[idx]
        ax.text(
            value + std + 0.003,
            position,
            f"{value:.4f}±{std:.4f}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    # Room for the value labels.
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 1.25)
    ax.set_yticks(range(n_top))
    ax.set_yticklabels([feature_names[i] for i in top_indices])
    ax.set_xlabel(f"Permutation Importance {metric_name}", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.set_title(
        f"Top {n_top} Most Important Features\n"
        f"Baseline {metric_name} = {baseline:.4f}",
        fontsize=14,
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    _save(
        fig,
        os.path.join(plots_folder, f"{metric_name}_importance.png"),
        f"{metric_name} importances plot saved to",
    )


# ── Trajectories ─────────────────────────────────────────────────────────────


def plot_node_trajectory(t, pred, true, power, title, save_path):
    """Three-panel plot: power forcing, prediction vs truth, residual."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        height_ratios=[1, 2, 1],
        sharex=True,
        gridspec_kw={"hspace": 0.1},
    )

    ax1.plot(t, power, color="tab:orange", linewidth=1.0)
    ax1.set_ylabel("Power (scaled)")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, true, label="Truth", linewidth=1.5)
    ax2.plot(t, pred, label="Prediction", linewidth=1.5, linestyle="--")
    ax2.set_ylabel("Concentration (scaled)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, pred - true, color="tab:red", linewidth=0.8)
    ax3.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax3.set_ylabel("Residual")
    ax3.set_xlabel("Time")
    ax3.grid(True, alpha=0.3)

    _save(fig, save_path, "NODE trajectory plot saved to")


def plot_node_trajectory_summary(t, all_preds, all_trues, title, save_path):
    """Overlay every trajectory, with the mean absolute residual beneath."""
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        height_ratios=[3, 1],
        sharex=True,
        gridspec_kw={"hspace": 0.05},
    )

    for i in range(len(all_preds)):
        ax1.plot(
            t,
            all_trues[i],
            color="tab:blue",
            alpha=0.3,
            linewidth=0.5,
            label="Truth" if i == 0 else None,
        )
        ax1.plot(
            t,
            all_preds[i],
            color="tab:red",
            alpha=0.3,
            linewidth=0.5,
            label="Prediction" if i == 0 else None,
        )

    ax1.set_ylabel("Concentration (scaled)", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    mean_abs_res = np.mean(np.abs(np.array(all_preds) - np.array(all_trues)), axis=0)
    ax2.plot(t, mean_abs_res, color="tab:red", linewidth=1.0)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Mean |Residual|", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.grid(True, alpha=0.3)

    _save(fig, save_path, "NODE trajectory summary saved to")


# ── Jacobian sensitivity ─────────────────────────────────────────────────────


def plot_jacobian_heatmap(
    matrix, col_names, row_names, title, xlabel, ylabel, save_path
):
    """Annotated heatmap of a mean absolute Jacobian."""
    fig, ax = plt.subplots(
        figsize=(max(8, len(col_names) * 1.2), max(6, len(row_names) * 0.8))
    )

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean absolute sensitivity")

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i in range(len(row_names)):
        for j in range(len(col_names)):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > matrix.max() * 0.6 else "black",
            )

    fig.tight_layout()
    _save(fig, save_path, "Jacobian heatmap saved to", dpi=150)


def plot_combined_sensitivity(
    sensitivities, stds, all_names, n_state, n_forcing, title, save_path
):
    """One derivative's sensitivity to every state and forcing variable."""
    colors = [STATE_COLOR] * n_state + [FORCING_COLOR] * n_forcing
    order = np.argsort(sensitivities)[::-1]

    fig, ax = plt.subplots(figsize=(max(8, len(all_names) * 0.6), 5))
    ax.bar(
        range(len(order)),
        sensitivities[order],
        yerr=stds[order],
        color=[colors[i] for i in order],
        capsize=3,
        edgecolor="gray",
        linewidth=0.5,
    )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [all_names[i] for i in order], rotation=45, ha="right", fontsize=9
    )
    ax.set_ylabel("Mean |∂f/∂·|")
    ax.set_title(title)
    ax.legend(
        handles=[
            Patch(facecolor=STATE_COLOR, label="State (∂f/∂y)"),
            Patch(facecolor=FORCING_COLOR, label="Forcing (∂f/∂u)"),
        ],
        loc="upper right",
    )

    fig.tight_layout()
    _save(fig, save_path, "Combined sensitivity plot saved to", dpi=150)


def plot_jacobian_over_time(
    time_fractions,
    state_over_time,
    forcing_over_time,
    target_names,
    forcing_names,
    title,
    save_path,
):
    """How state and forcing sensitivities evolve along the trajectory."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, series, names, ylabel, panel_title in (
        (ax1, state_over_time, target_names, "Mean |∂f/∂y_j|", "State sensitivities"),
        (
            ax2,
            forcing_over_time,
            forcing_names,
            "Mean |∂f/∂u_j|",
            "Forcing sensitivities",
        ),
    ):
        for j, name in enumerate(names):
            ax.plot(time_fractions, series[:, j], label=name, linewidth=1.5)
        ax.set_xlabel("Normalised time")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path, "Jacobian time-evolution plot saved to", dpi=150)


# ── Depletion matrix ─────────────────────────────────────────────────────────


def plot_depletion_matrix_mean(mean_A, std_A, target_names, time_unit, save_path):
    """Time-averaged learned depletion matrix, in physical units."""
    n_target = len(target_names)
    limit = np.abs(mean_A).max()

    fig, ax = plt.subplots(figsize=(max(6, n_target * 1.5), max(5, n_target * 1.2)))
    im = ax.imshow(mean_A, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit)
    fig.colorbar(im, ax=ax, label=f"Rate coefficient [1/{time_unit}]")

    ax.set_xticks(range(n_target))
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_target))
    ax.set_yticklabels([f"d{name}/dt" for name in target_names], fontsize=10)
    ax.set_title("Learned Depletion Matrix A (time-averaged, physical units)")

    for i in range(n_target):
        for j in range(n_target):
            ax.text(
                j,
                i,
                f"{mean_A[i, j]:.4e}\n(±{std_A[i, j]:.2e})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(mean_A[i, j]) > limit * 0.6 else "black",
            )

    fig.tight_layout()
    _save(fig, save_path, "Depletion matrix plot saved to", dpi=150)


def plot_depletion_matrix_evolution(
    time_fractions,
    mean_over_time,
    std_over_time,
    target_names,
    time_unit,
    num_runs,
    save_path,
):
    """Every matrix entry against time, with a ±1σ band across runs."""
    n_target = len(target_names)
    fig, axes = plt.subplots(
        n_target, n_target, figsize=(4 * n_target, 3.5 * n_target), sharex=True
    )

    for i in range(n_target):
        for j in range(n_target):
            ax = axes[i, j] if n_target > 1 else axes
            mean_vals = mean_over_time[:, i, j]
            std_vals = std_over_time[:, i, j]
            color = "tab:red" if i == j else "tab:blue"

            ax.plot(time_fractions, mean_vals, linewidth=1.5, color=color)
            ax.fill_between(
                time_fractions,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.2,
                color=color,
            )
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_title(f"A[{target_names[i]},{target_names[j]}]", fontsize=9)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel(f"[1/{time_unit}]", fontsize=8)
            if i == n_target - 1:
                ax.set_xlabel("Normalised time")

    fig.suptitle(
        f"Depletion Matrix Entries Over Time (physical units, 1/{time_unit})\n"
        f"shaded = ±1σ across {num_runs} runs",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, save_path, "Depletion matrix evolution plot saved to", dpi=150)


# ── Per-step importance ──────────────────────────────────────────────────────


def plot_stepwise_importance_bar(
    mean_pct, sem_pct, feature_names, feature_types, target_names, num_runs, result_dir
):
    """Time-averaged % importance with SEM, sorted descending, one plot per target."""
    n_features = len(feature_names)
    colors = [
        FORCING_COLOR if kind == "Forcing" else STATE_COLOR for kind in feature_types
    ]

    for k, target_name in enumerate(target_names):
        order = np.argsort(mean_pct[:, k])[::-1]
        means = mean_pct[order, k]
        sems = sem_pct[order, k]

        fig, ax = plt.subplots(figsize=(max(8, n_features * 0.9), 5))
        ax.bar(
            range(n_features),
            means,
            yerr=sems,
            color=[colors[i] for i in order],
            capsize=4,
            edgecolor="gray",
            linewidth=0.5,
        )

        y_top = float((means + sems).max()) if n_features else 1.0
        for i, (mean, sem) in enumerate(zip(means, sems, strict=False)):
            ax.text(
                i,
                mean + sem + y_top * 0.02,
                f"{mean:.1f}±{sem:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(range(n_features))
        ax.set_xticklabels(
            [feature_names[i] for i in order], rotation=45, ha="right", fontsize=9
        )
        ax.set_ylabel("MAE Importance [%]", fontsize=12)
        ax.set_title(
            f"Per-step teacher-forced importance — target: {target_name}\n"
            f"(time-averaged mean ± SEM, {num_runs} runs)",
            fontsize=11,
        )
        ax.legend(
            handles=[
                Patch(facecolor=FORCING_COLOR, label="Forcing input"),
                Patch(facecolor=STATE_COLOR, label="Isotope state"),
            ],
            fontsize=9,
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, y_top * 1.20 if y_top > 0 else 1.0)
        fig.tight_layout()

        _save(
            fig,
            os.path.join(result_dir, target_name, "stepwise_importance_bar.png"),
            f"{target_name} stepwise importance bar chart saved to",
            dpi=150,
        )


def plot_stepwise_importance_over_time(
    mean_by_time, t_fractions, feature_names, feature_types, target_names, result_dir
):
    """Importance against normalised time, forcings and isotope states side by side.

    The isotopes that start at zero should show importance rising as their
    concentrations build up — that is what this plot is for.
    """
    forcing_idx = [j for j, kind in enumerate(feature_types) if kind == "Forcing"]
    state_idx = [j for j, kind in enumerate(feature_types) if kind == "State"]

    for k, target_name in enumerate(target_names):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for ax, indices, panel_title in (
            (ax1, forcing_idx, "Forcing importance over time"),
            (ax2, state_idx, "Isotope state importance over time"),
        ):
            for j in indices:
                ax.plot(
                    t_fractions,
                    mean_by_time[:, j, k],
                    label=feature_names[j],
                    linewidth=1.8,
                )
            ax.set_xlabel("Normalised time", fontsize=11)
            ax.set_ylabel("Mean ΔMAE per step", fontsize=11)
            ax.set_title(panel_title, fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Per-step importance evolution — target: {target_name}",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()

        _save(
            fig,
            os.path.join(result_dir, target_name, "stepwise_importance_over_time.png"),
            f"{target_name} stepwise importance evolution saved to",
            dpi=150,
        )
