# Neural ODE for U238 trajectory prediction with power as external forcing
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from loguru import logger

import nuclear_surrogates.datamodule.dataset_helper as data_help
import nuclear_surrogates.datamodule.data_scalers as data_scalers

from sklearn.preprocessing import MinMaxScaler, StandardScaler

RESULTS_DIR = "../results/NODE"

# ─── Model ───────────────────────────────────────────────────────────────────


class ODEFuncForced(nn.Module):
    """
    Neural ODE that learns dU238/dt = f(power(t), U238(t)).

    Power is NOT part of the ODE state — it's an external forcing function
    that is interpolated at whatever time the solver requests.
    The ODE state is just [U238], a 1D variable.

    BATCHED VERSION: power_profiles is (batch, steps) so a single odeint call
    can integrate the whole mini-batch in parallel on the GPU.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.nfe = 0
        self.t_points = None  # 1-D time grid, shape (steps,)
        self.power_profiles = None  # (batch, steps) — one profile per trajectory

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # inputs: [power_at_t, U238]
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # output: dU238/dt
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def set_forcing(self, t_points, power_profiles):
        """
        Set the known power schedule before calling odeint.
        t_points:      (steps,)
        power_profiles: (batch, steps)  ← note: now batched
        """
        self.t_points = t_points
        self.power_profiles = power_profiles  # (batch, steps)

    def _interpolate_power(self, t):
        """
        Linearly interpolate power at an arbitrary scalar time t.
        Returns (batch, 1) — a distinct interpolated value for every trajectory.
        """
        t_clamped = t.clamp(self.t_points[0], self.t_points[-1])

        # searchsorted expects a 1-D sorted sequence; t_clamped is a scalar tensor
        idx = torch.searchsorted(self.t_points, t_clamped.unsqueeze(0)).squeeze() - 1
        idx = idx.clamp(0, len(self.t_points) - 2)

        t0 = self.t_points[idx]  # scalar
        t1 = self.t_points[idx + 1]  # scalar
        p0 = self.power_profiles[:, idx]  # (batch,)
        p1 = self.power_profiles[:, idx + 1]  # (batch,)

        frac = (t_clamped - t0) / (t1 - t0 + 1e-8)  # scalar
        return (p0 + frac * (p1 - p0)).unsqueeze(-1)  # (batch, 1)

    def forward(self, t, y):
        """
        t: scalar time (called by the ODE solver)
        y: (batch, 1) — U238 for the whole batch
        """
        self.nfe += 1
        power = self._interpolate_power(t)  # (batch, 1) — no expand needed
        combined = torch.cat([power, y], dim=-1)  # (batch, 2)
        return self.net(combined)  # (batch, 1) = dU238/dt


# ─── Training & Evaluation ──────────────────────────────────────────────────


def _run_batch(func, batch, t_span):
    """
    Forward pass for a single mini-batch of trajectories.

    batch:  (batch_size, steps, 2)  — already on device
    t_span: (steps,)                — already on device

    Returns pred (steps, batch_size, 1) and true (batch_size, steps, 1).
    """
    power_profiles = batch[:, :, 0]  # (batch_size, steps)
    u238_true = batch[:, :, 1:2]  # (batch_size, steps, 1)
    y0 = u238_true[:, 0, :]  # (batch_size, 1) — initial conditions

    func.set_forcing(t_span, power_profiles)

    # odeint: y0 (batch_size, 1) → output (steps, batch_size, 1)
    u238_pred = odeint(func, y0, t_span, method="dopri5", rtol=1e-5, atol=1e-7)

    # rearrange pred to (batch_size, steps, 1) to match true
    u238_pred = u238_pred.permute(1, 0, 2)

    return u238_pred, u238_true


def train_one_epoch(func, trajectories, t_span, optimizer, batch_size):
    """
    trajectories: (num_runs, steps, 2) — on device
    t_span:       (steps,)             — on device
    """
    func.train()
    func.nfe = 0
    epoch_loss = 0.0
    num_runs = trajectories.shape[0]

    # Shuffle runs each epoch and split into mini-batches
    perm = torch.randperm(num_runs, device=trajectories.device)
    shuffled = trajectories[perm]

    num_batches = 0
    for start in range(0, num_runs, batch_size):
        batch = shuffled[start : start + batch_size]  # (B, steps, 2)

        optimizer.zero_grad()
        u238_pred, u238_true = _run_batch(func, batch, t_span)

        loss = ((u238_pred - u238_true) ** 2).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    return epoch_loss / num_batches


@torch.no_grad()
def evaluate_trajectories(func, trajectories, t_span, batch_size):
    """
    trajectories: (num_runs, steps, 2) — on device
    t_span:       (steps,)             — on device
    """
    func.eval()
    total_loss = 0.0
    num_runs = trajectories.shape[0]
    num_batches = 0

    for start in range(0, num_runs, batch_size):
        batch = trajectories[start : start + batch_size]

        u238_pred, u238_true = _run_batch(func, batch, t_span)
        loss = ((u238_pred - u238_true) ** 2).mean()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def predict_all(func, trajectories, t_span):
    """
    Predict all trajectories in one batched pass.
    Returns (preds, trues) as numpy arrays of shape (num_runs, steps).
    trajectories: (num_runs, steps, 2) — on device
    """
    func.eval()
    u238_pred, u238_true = _run_batch(func, trajectories, t_span)
    # Both are (num_runs, steps, 1) — squeeze the last dim
    return (
        u238_pred.squeeze(-1).cpu().numpy(),  # (num_runs, steps)
        u238_true.squeeze(-1).cpu().numpy(),  # (num_runs, steps)
        trajectories[:, :, 0].cpu().numpy(),  # (num_runs, steps) power
    )


# ─── Plotting ───────────────────────────────────────────────────────────────


def plot_loss_curves(train_losses, val_losses, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Validation")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_trajectory_prediction(t, u238_pred, u238_true, power, title, save_path):
    """Three-panel plot: power input, U238 prediction vs truth, residual."""
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
    ax1.set_title(title)

    ax2.plot(t, u238_true, label="Truth", linewidth=1.0)
    ax2.plot(t, u238_pred, label="Prediction", linewidth=1.0, linestyle="--")
    ax2.set_ylabel("U238 (scaled)")
    ax2.legend()

    residuals = u238_pred - u238_true
    ax3.plot(t, residuals, color="tab:red", linewidth=0.8)
    ax3.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax3.set_ylabel("Residual")
    ax3.set_xlabel("Time")

    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_multi_trajectory_summary(t, all_preds, all_trues, title, save_path):
    """Overlay multiple trajectory predictions."""
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        height_ratios=[3, 1],
        sharex=True,
        gridspec_kw={"hspace": 0.05},
    )

    all_residuals = []
    for i, (pred, true) in enumerate(zip(all_preds, all_trues)):
        ax1.plot(
            t,
            true,
            color="tab:blue",
            alpha=0.3,
            linewidth=0.5,
            label="Truth" if i == 0 else None,
        )
        ax1.plot(
            t,
            pred,
            color="tab:red",
            alpha=0.3,
            linewidth=0.5,
            label="Prediction" if i == 0 else None,
        )
        all_residuals.append(pred - true)

    ax1.set_ylabel("U238 (scaled)")
    ax1.set_title(title)
    ax1.legend()

    mean_abs_res = np.mean(np.abs(np.array(all_residuals)), axis=0)
    ax2.plot(t, mean_abs_res, color="tab:red", linewidth=1.0)
    ax2.set_ylabel("Mean |Residual|")
    ax2.set_xlabel("Time")

    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ─── Data Reshaping ─────────────────────────────────────────────────────────


def reshape_to_trajectories(X, steps_per_run):
    """
    Reshape flat array (num_runs * steps_per_run, features)
    → (num_runs, steps_per_run, features)
    """
    num_samples, num_features = X.shape
    num_runs = num_samples // steps_per_run
    assert (
        num_runs * steps_per_run == num_samples
    ), f"Data length {num_samples} not divisible by steps_per_run {steps_per_run}"
    return torch.tensor(
        X.reshape(num_runs, steps_per_run, num_features),
        dtype=torch.float32,
    )


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_frac = 128 * 3 / 3305
    steps_per_run = 101
    batch_size = 128  # number of trajectories per mini-batch — tune to GPU memory

    data_df, run_length, time_array = data_help.read_data(
        "../data/casl_3305_runs_inter.h5",
        data_frac,
        drop_run_label=True,
    )
    data_help.print_dataset_stats(data_df)

    all_columns = ["power_W_g", "U238"]
    data_df = data_help.filter_columns(data_df, all_columns)

    input_data_arr, col_index_map = data_help.split_df(data_df, all_columns)

    # ── Scale ──
    input_scaler = data_scalers.create_column_transformer(
        {"power_W_g": MinMaxScaler(), "U238": StandardScaler()},
        col_index_map,
    )
    scaled_data = input_scaler.fit_transform(input_data_arr)

    # ── Reshape → (num_runs, steps, 2) ──
    all_trajectories = reshape_to_trajectories(scaled_data, steps_per_run)

    # Drop last timestep (NaN power at final step)
    # Note: this means the model is trained to predict up to t=100, not t=101, but that's fine since the last power value is missing.
    all_trajectories = all_trajectories[:, :-1, :]
    actual_steps = steps_per_run - 1

    assert not torch.isnan(
        all_trajectories
    ).any(), f"NaNs found in trajectories! Count: {torch.isnan(all_trajectories).sum()}"

    num_runs = all_trajectories.shape[0]
    print(f"Total trajectories: {num_runs}, each {actual_steps} steps")

    # ── Train / val / test split by run (not timestep) ──
    perm = torch.randperm(num_runs)
    all_trajectories = all_trajectories[perm]

    if num_runs <= 3:
        train_trajs = all_trajectories[:1]
        val_trajs = all_trajectories[1:2]
        test_trajs = all_trajectories[2:]
    else:
        n_train = int(num_runs * 0.6)
        n_val = int(num_runs * 0.2)
        train_trajs = all_trajectories[:n_train]
        val_trajs = all_trajectories[n_train : n_train + n_val]
        test_trajs = all_trajectories[n_train + n_val :]

    # ── Move full splits to device once ──
    # All subsequent tensor operations happen on-device; no per-sample transfers.
    # make sure to keep the full trajectories on device for efficient batch processing in the ODE solver.

    train_trajs = train_trajs.to(device)
    val_trajs = val_trajs.to(device)
    test_trajs = test_trajs.to(device)

    print(
        f"Train: {len(train_trajs)} runs, Val: {len(val_trajs)} runs, Test: {len(test_trajs)} runs"
    )

    # ── Time span (on device) ──
    t_span = torch.tensor(time_array[:actual_steps], dtype=torch.float32, device=device)

    # ── Model, optimiser, scheduler ──
    hidden_dim = 64
    func = ODEFuncForced(hidden_dim).to(device)
    optimizer = torch.optim.AdamW(func.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=50,
        factor=0.5,
    )

    # ── Training loop ──
    num_epochs = 1000
    best_loss = float("inf")
    best_model_state = None
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        avg_train = train_one_epoch(func, train_trajs, t_span, optimizer, batch_size)
        avg_val = evaluate_trajectories(func, val_trajs, t_span, batch_size)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_loss:
            best_loss = avg_val
            best_model_state = {k: v.clone() for k, v in func.state_dict().items()}

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:4d} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | LR: {lr:.2e} | NFE: {func.nfe}"
        )

    print(f"\nBest val loss: {best_loss:.6f} — restoring best weights")
    func.load_state_dict(best_model_state)

    # ── Loss curves ──
    plot_loss_curves(train_losses, val_losses, f"{RESULTS_DIR}/loss_curves.png")

    # ── Test set predictions (single batched pass) ──
    t_np = t_span.cpu().numpy()
    all_preds_full, all_trues_full, all_powers = predict_all(func, test_trajs, t_span)

    # Plot up to 5 individual trajectories
    num_to_plot = min(5, len(test_trajs))
    for i in range(num_to_plot):
        plot_trajectory_prediction(
            t_np,
            all_preds_full[i],
            all_trues_full[i],
            all_powers[i],
            title=f"Test Trajectory {i+1}",
            save_path=f"{RESULTS_DIR}/test_traj_{i+1}.png",
        )

    # Summary overlay plot
    plot_multi_trajectory_summary(
        t_np,
        list(all_preds_full),
        list(all_trues_full),
        title=f"All Test Trajectories ({len(test_trajs)} runs)",
        save_path=f"{RESULTS_DIR}/test_all_trajectories.png",
    )

    # ── Test metrics ──
    residuals = all_preds_full - all_trues_full
    print(f"\nTEST SET ({len(test_trajs)} trajectories):")
    print(f"  MSE:  {np.mean(residuals**2):.6f}")
    print(f"  MAE:  {np.mean(np.abs(residuals)):.6f}")
    print(f"  Mean residual: {np.mean(residuals):.6f}")
    print(f"  Std residual:  {np.std(residuals):.6f}")
    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()  # python playground/neural_ode_nuclear.py
