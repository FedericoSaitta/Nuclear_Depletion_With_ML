"""Jacobian sensitivity of the learned right-hand side.

Two derivatives, evaluated at sampled points along the test trajectories:
``∂f/∂y`` (how each isotope's rate responds to every isotope) and ``∂f/∂u``
(how it responds to the forcing). Their mean absolute values are the
sensitivity heatmaps; their evolution along the trajectory is the time plot.
"""

import os

import numpy as np
import torch
from loguru import logger

from nuclear_surrogates.analysis.rollout import JACOBIAN_BATCH, SolveContext
from nuclear_surrogates.utils import plot


def state_jacobian(ctx: SolveContext, t_eval, y_eval, forcing_eval, t_idx):
    """∂f/∂y at one timestep, for every trajectory in the batch."""
    with torch.inference_mode(False):
        # Cloning escapes inference-mode tracking, which forbids autograd.
        t_eval = t_eval.clone()
        forcing_eval = forcing_eval.clone()
        y_eval = y_eval.clone().requires_grad_(True)

        was_training = ctx.func.training
        ctx.func.train()
        ctx.func.set_forcing(t_eval, forcing_eval)

        dydt = ctx.func(t_eval[t_idx], y_eval)
        jacobians = [
            torch.autograd.grad(dydt[:, i].sum(), y_eval, retain_graph=True)[0]
            for i in range(dydt.shape[-1])
        ]

        ctx.func.train(was_training)
        return torch.stack(jacobians, dim=1)


def forcing_jacobian(ctx: SolveContext, t_eval, y_eval, forcing_eval, t_idx):
    """∂f/∂u at one timestep, attributed to the forcing interval actually read."""
    with torch.inference_mode(False):
        t_eval = t_eval.clone()
        y_eval = y_eval.clone().detach()
        forcing_eval = forcing_eval.clone().requires_grad_(True)

        was_training = ctx.func.training
        ctx.func.train()
        ctx.func.set_forcing(t_eval, forcing_eval)

        dydt = ctx.func(t_eval[t_idx], y_eval)
        with torch.no_grad():
            interp_idx = ctx.func.forcing_index(t_eval[t_idx])

        jacobians = [
            torch.autograd.grad(dydt[:, i].sum(), forcing_eval, retain_graph=True)[0][
                :, interp_idx, :
            ]
            for i in range(dydt.shape[-1])
        ]

        ctx.func.train(was_training)
        return torch.stack(jacobians, dim=1)


def jacobian_analysis(
    ctx: SolveContext, all_inputs_scaled, all_trues_scaled, target_names, forcing_names
):
    """State and forcing Jacobians across the test set.

    Produces mean |∂f/∂y| and |∂f/∂u| heatmaps, a per-target sensitivity bar
    chart, and the evolution of both along the trajectory.
    """
    logger.info(f"\n{'=' * 20}")
    logger.info("JACOBIAN SENSITIVITY ANALYSIS")
    logger.info(f"{'=' * 20}")

    num_runs, steps, _ = all_inputs_scaled.shape
    t_span = ctx.t_on_device()

    # Evenly spaced timesteps, skipping the first and last.
    timestep_indices = np.linspace(1, steps - 2, min(20, steps - 2), dtype=int)
    logger.info(
        f"Evaluating Jacobians: {num_runs} runs × {len(timestep_indices)} timesteps"
    )

    all_state_jacs, all_forcing_jacs = [], []
    state_jacs_by_time = {t: [] for t in timestep_indices}
    forcing_jacs_by_time = {t: [] for t in timestep_indices}

    for start in range(0, num_runs, JACOBIAN_BATCH):
        end = min(start + JACOBIAN_BATCH, num_runs)

        inputs_batch = ctx.as_tensor(all_inputs_scaled[start:end])
        trues_batch = ctx.as_tensor(all_trues_scaled[start:end])

        for t_idx in timestep_indices:
            y_t = trues_batch[:, t_idx, :]
            abs_state_jac = (
                state_jacobian(ctx, t_span, y_t, inputs_batch, int(t_idx))
                .abs()
                .cpu()
                .numpy()
            )
            abs_forcing_jac = (
                forcing_jacobian(ctx, t_span, y_t, inputs_batch, int(t_idx))
                .abs()
                .cpu()
                .numpy()
            )

            for b in range(abs_state_jac.shape[0]):
                all_state_jacs.append(abs_state_jac[b])
                all_forcing_jacs.append(abs_forcing_jac[b])
                state_jacs_by_time[t_idx].append(abs_state_jac[b])
                forcing_jacs_by_time[t_idx].append(abs_forcing_jac[b])

    mean_state_jac = np.mean(all_state_jacs, axis=0)
    mean_forcing_jac = np.mean(all_forcing_jacs, axis=0)
    std_state_jac = np.std(all_state_jacs, axis=0)
    std_forcing_jac = np.std(all_forcing_jacs, axis=0)

    logger.info("\nMean |∂f/∂y| (state sensitivities):")
    for i, t_name in enumerate(target_names):
        for j, s_name in enumerate(target_names):
            logger.info(
                f"  ∂(d{t_name}/dt)/∂{s_name}: "
                f"{mean_state_jac[i, j]:.6f} ± {std_state_jac[i, j]:.6f}"
            )

    logger.info("\nMean |∂f/∂u| (forcing sensitivities):")
    for i, t_name in enumerate(target_names):
        for j, f_name in enumerate(forcing_names):
            logger.info(
                f"  ∂(d{t_name}/dt)/∂{f_name}: "
                f"{mean_forcing_jac[i, j]:.6f} ± {std_forcing_jac[i, j]:.6f}"
            )

    plot.plot_jacobian_heatmap(
        mean_state_jac,
        target_names,
        target_names,
        title="State Sensitivity: Mean |∂f/∂y|",
        xlabel="State variable (y_j)",
        ylabel="Derivative (dy_i/dt)",
        save_path=os.path.join(ctx.result_dir, "jacobian_state_heatmap.png"),
    )
    plot.plot_jacobian_heatmap(
        mean_forcing_jac,
        forcing_names,
        target_names,
        title="Forcing Sensitivity: Mean |∂f/∂u|",
        xlabel="Forcing input (u_j)",
        ylabel="Derivative (dy_i/dt)",
        save_path=os.path.join(ctx.result_dir, "jacobian_forcing_heatmap.png"),
    )

    all_names = target_names + forcing_names
    combined_jac = np.concatenate([mean_state_jac, mean_forcing_jac], axis=1)
    combined_std = np.concatenate([std_state_jac, std_forcing_jac], axis=1)

    sorted_timesteps = sorted(timestep_indices)
    time_fractions = ctx.t_span.cpu().numpy()[sorted_timesteps]

    for idx, t_name in enumerate(target_names):
        output_dir = os.path.join(ctx.result_dir, t_name)
        os.makedirs(output_dir, exist_ok=True)

        plot.plot_combined_sensitivity(
            combined_jac[idx],
            combined_std[idx],
            all_names,
            len(target_names),
            len(forcing_names),
            title=f"Sensitivity of d{t_name}/dt",
            save_path=os.path.join(output_dir, "jacobian_combined_sensitivity.png"),
        )
        plot.plot_jacobian_over_time(
            time_fractions,
            np.array(
                [
                    np.mean([j[idx, :] for j in state_jacs_by_time[t]], axis=0)
                    for t in sorted_timesteps
                ]
            ),
            np.array(
                [
                    np.mean([j[idx, :] for j in forcing_jacs_by_time[t]], axis=0)
                    for t in sorted_timesteps
                ]
            ),
            target_names,
            forcing_names,
            title=f"Sensitivity evolution: d{t_name}/dt",
            save_path=os.path.join(output_dir, "jacobian_time_evolution.png"),
        )

    logger.info(f"\n  Jacobian analysis plots saved to: {ctx.result_dir}")
