import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

from torchdiffeq import odeint

import matplotlib.pyplot as plt
import os


os.makedirs("results/", exist_ok=True)


# define the differential function
def lotka_volterra(t, p, alpha=1.5, beta=1.0, delta=0.3, gamma=1.0):
    x, y = p
    dxdt = alpha * x - beta * x * y
    dydt = -gamma * y + delta * x * y
    return torch.stack(
        [dxdt, dydt]
    )  # concatenates a sequence of tensors along new direction


# Initial conditions
Num_Steps = 500
p0 = torch.tensor([1.0, 1.0])
t_span = torch.linspace(0, 20, Num_Steps)
true_traj = odeint(lotka_volterra, p0, t_span)

# we are giving odeint which is a numerical solver we are giving the rhs of the equation
# x is the prey population
# y is the predator population
# alpha is the birth rate
# beta is the predation rate
# delta how efficiently eating prey results in pradtor birth
# gamma is the predator death rate

# Time portraits
plt.plot(t_span, true_traj[:, 0], label="Prey Pop")
plt.plot(t_span, true_traj[:, 1], label="Predator Pop")
plt.legend(loc="best")
plt.savefig("results/time_portraits.png")
plt.close()

## As expected predator peak always lags behind the prey peak, the prey
## population booms and then the predators follow


# Phase portrait
plt.plot(true_traj[:, 0], true_traj[:, 1])
plt.savefig("results/phase_portrait.png")
plt.close()

## Because the Lotka Volterra system has a conserved quantity then the trajecories
## Form closed loops in phase space, hence why we see the oval shape


## Checking behaviour for different starting conditions:
def plot_phase_space_diff_conditions(p0_arr):
    def run_simulation(x0, y0):
        p0 = torch.tensor([x0, y0])
        t_span = torch.linspace(0, 15, 300)
        true_traj = odeint(lotka_volterra, p0, t_span)
        return true_traj

    for x0, y0 in p0_arr:
        result = run_simulation(x0, y0)
        plt.plot(result[:, 0], result[:, 1])

    plt.savefig("results/phase_portraits_pairs.png")
    plt.close()


x0_vals = [0.1, 0.5, 1.0, 2.0]
y0_vals = [0.1, 0.5, 1.0, 2.0]

from itertools import product

pairs = list(product(x0_vals, y0_vals))
plot_phase_space_diff_conditions(pairs)

## Changing initial parameters leads to trajectories which are still closed
## and appear as contained in one another


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


import random


# Notably the end points may be missing with this function so we can see
# extrapolation both at start and at the end
def random_subsample(data, num_samples):
    indices = sorted(random.sample(range(len(data)), num_samples))
    return data[indices]


## Now defining the NODE module
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.nfe = 0  # number of function evaluations
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


## Now the optimization code:
losses = []
fn_evals = []
hidden_dim = 64
func = ODEFunc(hidden_dim).to(device)
optimizer = torch.optim.AdamW(func.parameters(), lr=1e-3)

# t_obs also needs to be on device
# a t_span that may not be the same one created inside noisy_volterra
noise_level = 0.0
t_true = torch.linspace(0, 15, 600).to(device)
y_true = noisy_volterra(noise_level, [1.0, 1.0], 600, 15)

y_mean = y_true.mean(0)
y_std = y_true.std(0)
y_true = (y_true - y_mean) / y_std
y0_raw = torch.tensor([1.0, 1.0]).to(device)
y0 = (y0_raw - y_mean) / y_std

y_obs = even_subsample(y_true, 200)
t_obs = even_subsample(t_true, 200)

# train on normalized data, then unnormalize predictions

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=200, factor=0.5
)
best_loss = float("inf")
best_model_state = None

for epoch in range(2000):
    func.nfe = 0  # reset before each epoch
    optimizer.zero_grad()
    y_pred = odeint(func, y0, t_obs, method="dopri5", rtol=1e-7, atol=1e-9)

    loss = ((y_pred - y_obs) ** 2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss)

    if epoch % 10 == 0:
        print(f"\rEpoch {epoch}, Loss: {loss.item():.6f}, NFE: {func.nfe}", end="")

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model_state = {k: v.clone() for k, v in func.state_dict().items()}

    losses.append(loss.item())
    fn_evals.append(func.nfe)


fig, ax1 = plt.subplots()

ax1.plot(losses, color="tab:blue", label="Loss")
ax1.set_yscale("log")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(fn_evals, color="tab:red", alpha=0.7, label="NFE")
ax2.set_ylabel("Function Evaluations", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
fig.savefig("results/loss_curve.png", bbox_inches="tight")
plt.close(fig)

print(f"\nBest loss: {best_loss:.6f} — restoring best model weights")
func.load_state_dict(best_model_state)

y_pred = odeint(func, y0, t_true, method="dopri5", rtol=1e-7, atol=1e-9)

y_pred = y_pred.cpu().detach().numpy()
y_true = y_true.cpu().detach().numpy()
t_true = t_true.cpu().detach().numpy()
y_obs = y_obs.cpu().detach().numpy()
t_obs = t_obs.cpu().detach().numpy()

# --- Prey Predictions with Residuals ---
fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(8, 6),
    height_ratios=[3, 1],
    sharex=True,
    gridspec_kw={"hspace": 0.05},
)
ax1.set_title("Prey Predictions")
ax1.plot(t_true, y_pred[:, 0], label="Predictions")
ax1.plot(t_true, y_true[:, 0], label="Truth")
ax1.scatter(t_obs, y_obs[:, 0], s=10, label="Sampled Data", zorder=5)
ax1.legend()
ax1.set_ylabel("Prey")

residuals_prey = y_pred[:, 0] - y_true[:, 0]
ax2.plot(t_true, residuals_prey, color="tab:red", linewidth=0.8)
ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax2.set_ylabel("Residual")
ax2.set_xlabel("Time")
fig.savefig("results/prey_predictions.png", bbox_inches="tight")
plt.close(fig)


# --- Predator Predictions with Residuals ---
fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(8, 6),
    height_ratios=[3, 1],
    sharex=True,
    gridspec_kw={"hspace": 0.05},
)
ax1.set_title("Predator Predictions")

ax1.scatter(t_obs, y_obs[:, 1], s=10, label="Sampled Data", color="black")
ax1.plot(t_true, y_pred[:, 1], label="Predictions")
ax1.plot(t_true, y_true[:, 1], label="Truth")

ax1.legend()
ax1.set_ylabel("Predator")

residuals_pred = y_pred[:, 1] - y_true[:, 1]
ax2.plot(t_true, residuals_pred, color="tab:red", linewidth=0.8)
ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax2.set_ylabel("Residual")
ax2.set_xlabel("Time")
fig.savefig("results/predator_predictions.png", bbox_inches="tight")
plt.close(fig)


# --- Phase Plot with Residuals (Euclidean distance) ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 6), height_ratios=[3, 1], gridspec_kw={"hspace": 0.25}
)
ax1.set_title("Predator Prey Phase Plot")
ax1.scatter(y_obs[:, 0], y_obs[:, 1], s=10, label="Sampled Data", color="black")
ax1.plot(y_pred[:, 0], y_pred[:, 1], label="Predictions")
ax1.plot(y_true[:, 0], y_true[:, 1], label="Truth")
ax1.legend()
ax1.set_xlabel("Prey")
ax1.set_ylabel("Predator")

phase_residual = np.sqrt(residuals_prey**2 + residuals_pred**2)
ax2.plot(t_true, phase_residual, color="tab:red", linewidth=0.8)
ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax2.set_ylabel("Euclidean Error")
ax2.set_xlabel("Time")
fig.savefig("results/model_phase_plot.png", bbox_inches="tight")
plt.close(fig)
