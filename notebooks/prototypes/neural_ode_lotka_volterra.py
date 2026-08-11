import os

# Make BLAS/OpenMP-backed operations use 12 threads.
# Set these before heavy numerical work starts.
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

# Tell PyTorch to use 12 CPU threads.
torch.set_num_threads(12)
torch.set_num_interop_threads(12)

import matplotlib.pyplot as plt
import random
from itertools import product

# ============================================================
# Thesis-quality plotting configuration
# ============================================================

plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6

THESIS_FIGSIZE = (10, 7)

TITLE_SIZE = 26
LABEL_SIZE = 24
TICK_SIZE = 20
LEGEND_SIZE = 20

LINEWIDTH = 2.5
SCATTER_SIZE = 40

# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

os.makedirs("results/", exist_ok=True)

# ============================================================
# Differential equation
# ============================================================


def lotka_volterra(t, p, alpha=1.5, beta=1.0, delta=0.3, gamma=1.0):
    x, y = p

    dxdt = alpha * x - beta * x * y
    dydt = -gamma * y + delta * x * y

    return torch.stack([dxdt, dydt])


# ============================================================
# Initial trajectory
# ============================================================

Num_Steps = 500

p0 = torch.tensor([1.0, 1.0])

t_span = torch.linspace(0, 20, Num_Steps)

true_traj = odeint(lotka_volterra, p0, t_span)

# ============================================================
# Time portraits
# ============================================================

fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)

ax.plot(t_span, true_traj[:, 0], label="Prey Population", linewidth=LINEWIDTH)

ax.plot(t_span, true_traj[:, 1], label="Predator Population", linewidth=LINEWIDTH)

ax.set_xlabel("Time", fontsize=LABEL_SIZE)

ax.set_ylabel("Population", fontsize=LABEL_SIZE)

ax.tick_params(axis="both", labelsize=TICK_SIZE)

ax.legend(fontsize=LEGEND_SIZE)

ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("results/time_portraits.png", dpi=400, bbox_inches="tight")

plt.close()

# ============================================================
# Phase portrait
# ============================================================

fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)

ax.plot(true_traj[:, 0], true_traj[:, 1], linewidth=LINEWIDTH)

ax.set_xlabel("Prey Population", fontsize=LABEL_SIZE)

ax.set_ylabel("Predator Population", fontsize=LABEL_SIZE)

ax.tick_params(axis="both", labelsize=TICK_SIZE)

ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("results/phase_portrait.png", dpi=400, bbox_inches="tight")

plt.close()

# ============================================================
# Multiple phase portraits
# ============================================================


def plot_phase_space_diff_conditions(p0_arr):
    fig, ax = plt.subplots(figsize=(10, 8))

    def run_simulation(x0, y0):
        p0 = torch.tensor([x0, y0])

        t_span = torch.linspace(0, 15, 300)

        true_traj = odeint(lotka_volterra, p0, t_span)

        return true_traj

    for x0, y0 in p0_arr:
        result = run_simulation(x0, y0)

        ax.plot(result[:, 0], result[:, 1], linewidth=2, label=f"({x0}, {y0})")

    ax.set_xlabel("Prey Population", fontsize=LABEL_SIZE)

    ax.set_ylabel("Predator Population", fontsize=LABEL_SIZE)

    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    ax.legend(fontsize=14, ncol=2)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig("results/phase_portraits_pairs.png", dpi=400, bbox_inches="tight")

    plt.close()


x0_vals = [0.1, 0.5, 1.0, 2.0]
y0_vals = [0.1, 0.5, 1.0, 2.0]

pairs = list(product(x0_vals, y0_vals))

plot_phase_space_diff_conditions(pairs)

# ============================================================
# Noisy trajectories
# ============================================================


def noisy_volterra(std, p0, Num_steps, t_max):
    p0 = torch.tensor(p0).to(device)

    t_span = torch.linspace(0, t_max, Num_steps).to(device)

    true_traj = odeint(lotka_volterra, p0, t_span)

    noise = (2 * torch.rand(Num_steps, 2).to(device) - 1) * std

    noisy_traj = true_traj + true_traj * noise

    return noisy_traj


def even_subsample(data, num_samples):
    spacing = len(data) // num_samples

    return data[::spacing]


def random_subsample(data, num_samples):
    indices = sorted(random.sample(range(len(data)), num_samples))

    return data[indices]


# ============================================================
# Neural ODE model
# ============================================================


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.nfe = 0

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)

                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        self.nfe += 1

        return self.net(y)


# ============================================================
# Training data (with time normalization)
# ============================================================

losses = []
fn_evals = []

hidden_dim = 64

func = ODEFunc(hidden_dim).to(device)

optimizer = torch.optim.AdamW(func.parameters(), lr=1e-3)

noise_level = 0.0

# Physical time span
T_MAX = 15.0
N_POINTS = 600

t_true = torch.linspace(0, T_MAX, N_POINTS).to(device)

# Normalized time span ∈ [0, 1] — used for training
t_true_norm = t_true / T_MAX

y_true = noisy_volterra(noise_level, [1.0, 1.0], N_POINTS, T_MAX)

y_mean = y_true.mean(0)
y_std = y_true.std(0)

y_true = (y_true - y_mean) / y_std

y0_raw = torch.tensor([1.0, 1.0]).to(device)

y0 = (y0_raw - y_mean) / y_std

y_obs = even_subsample(y_true, 200)
t_obs = even_subsample(t_true, 200)
t_obs_norm = even_subsample(t_true_norm, 200)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=200, factor=0.5
)

best_loss = float("inf")
best_model_state = None

for epoch in range(2000):
    func.nfe = 0

    optimizer.zero_grad()

    # Integrate over normalized time ∈ [0, 1]
    y_pred = odeint(func, y0, t_obs_norm, method="dopri5", rtol=1e-7, atol=1e-9)

    loss = ((y_pred - y_obs) ** 2).mean()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)

    optimizer.step()

    scheduler.step(loss)

    if epoch % 10 == 0:
        print(
            f"\rEpoch {epoch}, " f"Loss: {loss.item():.6f}, " f"NFE: {func.nfe}", end=""
        )

    if loss.item() < best_loss:
        best_loss = loss.item()

        best_model_state = {k: v.clone() for k, v in func.state_dict().items()}

    losses.append(loss.item())
    fn_evals.append(func.nfe)

# ============================================================
# Loss curve
# ============================================================

fig, ax1 = plt.subplots(figsize=THESIS_FIGSIZE)

ax1.plot(losses, color="tab:blue", label="Loss", linewidth=LINEWIDTH)

ax1.set_yscale("log")

ax1.set_xlabel("Epoch", fontsize=LABEL_SIZE)

ax1.set_ylabel("Loss", fontsize=LABEL_SIZE, color="tab:blue")

ax1.tick_params(axis="both", labelsize=TICK_SIZE)

ax1.tick_params(axis="y", labelcolor="tab:blue")

ax1.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

ax2 = ax1.twinx()

ax2.plot(fn_evals, color="tab:red", alpha=0.8, label="NFE", linewidth=LINEWIDTH)

ax2.set_ylabel("Function Evaluations", fontsize=LABEL_SIZE, color="tab:red")

ax2.tick_params(axis="y", labelsize=TICK_SIZE)

ax2.tick_params(axis="y", labelcolor="tab:red")

ax2.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

fig.legend(lines1 + lines2, labels1 + labels2, fontsize=LEGEND_SIZE, loc="upper right")

ax1.grid(True, alpha=0.3)

plt.tight_layout()

fig.savefig("results/loss_curve.png", dpi=400, bbox_inches="tight")

plt.close(fig)

# ============================================================
# Restore best model
# ============================================================

print(f"\nBest loss: {best_loss:.6f} " f"— restoring best model weights")

func.load_state_dict(best_model_state)

# ============================================================
# Predictions (integrate over normalized time, plot vs physical)
# ============================================================

y_pred = odeint(func, y0, t_true_norm, method="dopri5", rtol=1e-7, atol=1e-9)

y_pred = y_pred.cpu().detach().numpy()
y_true = y_true.cpu().detach().numpy()

t_true = t_true.cpu().detach().numpy()

y_obs = y_obs.cpu().detach().numpy()
t_obs = t_obs.cpu().detach().numpy()

# ============================================================
# Prey predictions
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    height_ratios=[3, 1],
    sharex=True,
    gridspec_kw={"hspace": 0.08},
)

ax1.set_title("Prey Predictions", fontsize=TITLE_SIZE)

ax1.plot(t_true, y_pred[:, 0], label="Predictions", linewidth=LINEWIDTH)

ax1.plot(t_true, y_true[:, 0], label="Truth", linewidth=LINEWIDTH)

ax1.scatter(t_obs, y_obs[:, 0], s=SCATTER_SIZE, label="Sampled Data", zorder=5)

ax1.set_ylabel("Prey", fontsize=LABEL_SIZE)

ax1.tick_params(axis="both", labelsize=TICK_SIZE)

ax1.legend(fontsize=LEGEND_SIZE)

ax1.grid(True, alpha=0.3)

residuals_prey = y_pred[:, 0] - y_true[:, 0]

ax2.plot(t_true, residuals_prey, color="tab:red", linewidth=2)

ax2.axhline(0, color="black", linewidth=1.2, linestyle="--")

ax2.set_ylabel("Residual", fontsize=LABEL_SIZE)

ax2.set_xlabel("Time", fontsize=LABEL_SIZE)

ax2.tick_params(axis="both", labelsize=TICK_SIZE)

ax2.grid(True, alpha=0.3)

fig.savefig("results/prey_predictions.png", dpi=400, bbox_inches="tight")

plt.close(fig)

# ============================================================
# Predator predictions
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    height_ratios=[3, 1],
    sharex=True,
    gridspec_kw={"hspace": 0.08},
)

ax1.set_title("Predator Predictions", fontsize=TITLE_SIZE)

ax1.scatter(t_obs, y_obs[:, 1], s=SCATTER_SIZE, label="Sampled Data", color="black")

ax1.plot(t_true, y_pred[:, 1], label="Predictions", linewidth=LINEWIDTH)

ax1.plot(t_true, y_true[:, 1], label="Truth", linewidth=LINEWIDTH)

ax1.set_ylabel("Predator", fontsize=LABEL_SIZE)

ax1.tick_params(axis="both", labelsize=TICK_SIZE)

ax1.legend(fontsize=LEGEND_SIZE)

ax1.grid(True, alpha=0.3)

residuals_pred = y_pred[:, 1] - y_true[:, 1]

ax2.plot(t_true, residuals_pred, color="tab:red", linewidth=2)

ax2.axhline(0, color="black", linewidth=1.2, linestyle="--")

ax2.set_ylabel("Residual", fontsize=LABEL_SIZE)

ax2.set_xlabel("Time", fontsize=LABEL_SIZE)

ax2.tick_params(axis="both", labelsize=TICK_SIZE)

ax2.grid(True, alpha=0.3)

fig.savefig("results/predator_predictions.png", dpi=400, bbox_inches="tight")

plt.close(fig)

# ============================================================
# Phase plot
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 8), height_ratios=[3, 1], gridspec_kw={"hspace": 0.25}
)

ax1.set_title("Predator–Prey Phase Plot", fontsize=TITLE_SIZE)

ax1.scatter(
    y_obs[:, 0], y_obs[:, 1], s=SCATTER_SIZE, label="Sampled Data", color="black"
)

ax1.plot(y_pred[:, 0], y_pred[:, 1], label="Predictions", linewidth=LINEWIDTH)

ax1.plot(y_true[:, 0], y_true[:, 1], label="Truth", linewidth=LINEWIDTH)

ax1.set_xlabel("Prey", fontsize=LABEL_SIZE)

ax1.set_ylabel("Predator", fontsize=LABEL_SIZE)

ax1.tick_params(axis="both", labelsize=TICK_SIZE)

ax1.legend(fontsize=LEGEND_SIZE)

ax1.grid(True, alpha=0.3)

phase_residual = np.sqrt(residuals_prey**2 + residuals_pred**2)

ax2.plot(t_true, phase_residual, color="tab:red", linewidth=2)

ax2.axhline(0, color="black", linewidth=1.2, linestyle="--")

ax2.set_ylabel("Euclidean Error", fontsize=LABEL_SIZE)

ax2.set_xlabel("Time", fontsize=LABEL_SIZE)

ax2.tick_params(axis="both", labelsize=TICK_SIZE)

ax2.grid(True, alpha=0.3)

fig.savefig("results/model_phase_plot.png", dpi=400, bbox_inches="tight")

plt.close(fig)
