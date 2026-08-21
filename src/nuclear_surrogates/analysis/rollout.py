"""The solve context, and the two rollouts every other analysis pass builds on.

`SolveContext` is the whole of what the analysis passes need from a trained
model: a way to integrate, the right-hand side they differentiate, the time
grid, and the scaler that turns model units back into atom/b-cm. Bundling them
means an analysis function never touches a LightningModule or a Trainer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Batch sizes for the analysis passes. The single-step ones can be larger
# because they integrate one interval rather than a whole trajectory.
TRAJECTORY_BATCH = 64
JACOBIAN_BATCH = 32
SINGLE_STEP_BATCH = 128


@dataclass
class SolveContext:
    """Everything an analysis pass needs from a trained NODE.

    *odeint* integrates ``(y0, t)`` with the run's configured solver and
    tolerances; *func* is the right-hand side itself, needed by the passes that
    differentiate it or read the matrix it builds.
    """

    func: Any
    odeint: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    t_span: torch.Tensor
    device: Any
    unscale_targets: Callable[[np.ndarray], np.ndarray]
    result_dir: str

    def t_on_device(self) -> torch.Tensor:
        return self.t_span.to(self.device)

    def as_tensor(self, array) -> torch.Tensor:
        """Float32 tensor on the context's device, passing tensors through."""
        if torch.is_tensor(array):
            return array
        return torch.tensor(array, dtype=torch.float32, device=self.device)


def teacher_forced_predictions(ctx: SolveContext, all_inputs_scaled, all_trues_scaled):
    """Single-step predictions: integrate one dt from the true y(t).

    The DNN's teacher-forcing analogue. y(t) is the full state vector, so
    every isotope is fed ground truth at every step. The first timestep has
    no predecessor and is copied from the truth.

    Returns (num_runs, steps, n_target), in model units.
    """
    num_runs, steps, _ = all_trues_scaled.shape
    tf_preds = np.zeros_like(all_trues_scaled)
    tf_preds[:, 0, :] = all_trues_scaled[:, 0, :]

    t_span = ctx.t_on_device()

    for start in range(0, num_runs, TRAJECTORY_BATCH):
        end = min(start + TRAJECTORY_BATCH, num_runs)

        inputs_batch = ctx.as_tensor(all_inputs_scaled[start:end])
        trues_batch = ctx.as_tensor(all_trues_scaled[start:end])
        ctx.func.set_forcing(t_span, inputs_batch)

        for t in range(steps - 1):
            with torch.no_grad():
                pred = ctx.odeint(trues_batch[:, t, :], t_span[t : t + 2])[-1]
            tf_preds[start:end, t + 1, :] = pred.cpu().numpy()

    return tf_preds


def single_step_batch(ctx: SolveContext, y_t_np, t_idx, forcing_profiles):
    """Integrate one teacher-forced step for every run, in batches.

    *y_t_np* is (runs, n_state); *forcing_profiles* is (runs, steps, n_input),
    either numpy or an already-built tensor. The importance sweep calls this
    thousands of times with the same unperturbed forcing, so passing the tensor
    lets the caller convert it once.

    Returns (runs, n_state) in model units.
    """
    num_runs, n_state = y_t_np.shape
    t_span = ctx.t_on_device()
    t_short = t_span[t_idx : t_idx + 2]

    forcing = ctx.as_tensor(forcing_profiles)

    preds = np.zeros((num_runs, n_state))
    with torch.no_grad():
        for start in range(0, num_runs, SINGLE_STEP_BATCH):
            end = min(start + SINGLE_STEP_BATCH, num_runs)
            y_batch = ctx.as_tensor(y_t_np[start:end])
            ctx.func.set_forcing(t_span, forcing[start:end])
            preds[start:end] = ctx.odeint(y_batch, t_short)[-1].cpu().numpy()

    return preds
