import csv
import os
import sys

import lightning as L
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from torchdiffeq import odeint, odeint_adjoint

from nuclear_surrogates import evaluation
from nuclear_surrogates.models.model_architectures import ODEFuncForced, ODEFuncMatrix
from nuclear_surrogates.models.model_helper import get_loss_fn
from nuclear_surrogates.utils import metrics, plot
from nuclear_surrogates.utils.paths import result_dir

# Batch sizes for the analysis passes. The single-step ones can be larger
# because they integrate one interval rather than a whole trajectory.
TRAJECTORY_BATCH = 64
JACOBIAN_BATCH = 32
SINGLE_STEP_BATCH = 128


def _print_unicode_safe(line=""):
    """print() that degrades instead of crashing on a non-UTF-8 stdout.

    The markdown dump below contains Δ and ±. A Windows console running cp1252
    can encode ± but not Δ, so a plain print() raises UnicodeEncodeError
    part-way through the table and aborts the whole test epoch.
    """
    try:
        print(line)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "ascii"
        print(line.encode(encoding, errors="replace").decode(encoding))


class NODE_Model(L.LightningModule):
    """Trajectory surrogate: integrate dy/dt = f(forcing(t), y) over a whole run."""

    def __init__(self, config_object):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(config_object, resolve=True))
        self.cfg = config_object

        self.matrix_ode = getattr(config_object.model, "matrix_ode", False)
        if self.matrix_ode:
            self.func = ODEFuncMatrix(config_object)
            logger.info("Using matrix-based NODE: dy/dt = A(forcing, y) @ y")
        else:
            self.func = ODEFuncForced(config_object)
            logger.info("Using standard NODE: dy/dt = net(forcing, y)")

        self.loss_fn = get_loss_fn(config_object.train.loss)

        self.rtol = getattr(config_object.train, "rtol", 1e-5)
        self.atol = getattr(config_object.train, "atol", 1e-7)
        self.use_adjoint = getattr(config_object.train, "use_adjoint", False)
        self.adjoint_rtol = getattr(config_object.train, "adjoint_rtol", None)
        self.adjoint_atol = getattr(config_object.train, "adjoint_atol", None)
        self.adjoint_method = getattr(config_object.train, "adjoint_method", None)

        if self.use_adjoint:
            logger.info(
                f"Using odeint_adjoint | adjoint_rtol={self.adjoint_rtol} | "
                f"adjoint_atol={self.adjoint_atol} | adjoint_method={self.adjoint_method}"
            )
        else:
            logger.info("Using direct backprop through odeint")

        self.result_dir = result_dir(config_object)

        self._train_losses = []
        self._val_losses = []
        self._train_nfes = []

        self._test_preds = []
        self._test_trues = []
        self._test_input_trajs = []

    def setup(self, stage=None):
        dm = self.trainer.datamodule
        self.t_span = dm.t_span.to(self.device)
        self.n_input_features = dm.n_input_features
        self.n_target_features = dm.n_target_features

    # ── Forward ──────────────────────────────────────────────────────────────

    def _odeint(self, y0, t_span):
        """Central odeint call — solver and backward method come from the yaml.

        With `train.use_adjoint`, gradients come from solving the adjoint system
        backwards instead of storing the forward graph. That is O(1) in the
        number of function evaluations rather than O(NFE), which is what makes a
        large batch fit at all once the learned dynamics turn stiff — at roughly
        twice the wall time, because the backward pass becomes a second solve.
        """
        options = {}
        if self.cfg.train.solver == "rk4":
            step_size = getattr(self.cfg.train, "step_size", None)
            if step_size:
                options["step_size"] = step_size

        # Solver-specific keys the plain config cannot express, e.g.
        # `{solver: BDF}` for scipy_solver. Explicit values win.
        extra = getattr(self.cfg.train, "solver_options", None)
        if extra:
            options.update(OmegaConf.to_container(extra, resolve=True))

        common_kwargs = {
            "method": self.cfg.train.solver,
            "rtol": self.rtol,
            "atol": self.atol,
            "options": options or None,
        }

        if self.use_adjoint:
            adjoint_kwargs = {
                "adjoint_rtol": self.adjoint_rtol,
                "adjoint_atol": self.adjoint_atol,
                "adjoint_method": self.adjoint_method,
            }
            # Kept separate from the forward options: the adjoint may use a
            # different method, and a forward-only key (rk4's `step_size`) is
            # not valid for it.
            adjoint_extra = getattr(self.cfg.train, "adjoint_solver_options", None)
            if adjoint_extra:
                adjoint_kwargs["adjoint_options"] = OmegaConf.to_container(
                    adjoint_extra, resolve=True
                )
            return odeint_adjoint(
                self.func, y0, t_span, **adjoint_kwargs, **common_kwargs
            )
        return odeint(self.func, y0, t_span, **common_kwargs)

    def _forward_batch(self, batch):
        """Integrate a batch of trajectories from their t=0 state.

        Trajectories are laid out as [forcing features..., target features...],
        so `batch[0]` is (batch, steps, n_input + n_target). The integrator feeds
        the full state vector back into the network at every solver step, which
        is what couples all isotopes to each other.

        Returns (target_pred, target_true), both (batch, steps, n_target).
        """
        trajectories = batch[0]
        n_in = self.n_input_features

        forcing_profiles = trajectories[:, :, :n_in]
        target_true = trajectories[:, :, n_in:]
        y0 = target_true[:, 0, :]

        t_span = self.t_span.to(trajectories.device)
        self.func.set_forcing(t_span, forcing_profiles)

        target_pred = self._odeint(y0, t_span).permute(1, 0, 2)
        return target_pred, target_true

    def _unscale_targets(self, scaled_2d):
        """(N, n_target) model units -> atom/b-cm."""
        return self.trainer.datamodule.target_scaler.inverse_transform(scaled_2d)

    def _unscale_inputs(self, scaled_2d):
        """(N, n_input) model units -> physical units."""
        return self.trainer.datamodule.input_scaler.inverse_transform(scaled_2d)

    # ── Training ─────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        self.func.nfe = 0
        target_pred, target_true = self._forward_batch(batch)
        loss = self.loss_fn(target_pred, target_true)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("nfe", float(self.func.nfe), on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self._train_losses.append(self.trainer.callback_metrics["train_loss"].item())
        self._train_nfes.append(self.trainer.callback_metrics["nfe"].item())
        self.log(
            "lr", self.optimizers().param_groups[0]["lr"], on_epoch=True, prog_bar=True
        )

    def on_train_end(self):
        plot.plot_losses(
            self._train_losses, self._val_losses, self.result_dir, nfes=self._train_nfes
        )
        logger.info("Training complete.")

    # ── Validation ───────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        target_pred, target_true = self._forward_batch(batch)
        loss = self.loss_fn(target_pred, target_true)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self._val_losses.append(self.trainer.callback_metrics["val_loss"].item())

    # ── Test ─────────────────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx):
        target_pred, target_true = self._forward_batch(batch)
        loss = self.loss_fn(target_pred, target_true)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        self._test_preds.append(target_pred.cpu().numpy())
        self._test_trues.append(target_true.cpu().numpy())
        self._test_input_trajs.append(
            batch[0][:, :, : self.n_input_features].cpu().numpy()
        )
        return loss

    def on_test_epoch_end(self):
        datamodule = self.trainer.datamodule

        # All three are (num_runs, steps, features), still in model units.
        all_preds_scaled = np.concatenate(self._test_preds, axis=0)
        all_trues_scaled = np.concatenate(self._test_trues, axis=0)
        all_inputs_scaled = np.concatenate(self._test_input_trajs, axis=0)

        num_runs, steps, n_target = all_preds_scaled.shape
        n_input = all_inputs_scaled.shape[2]

        def unscale(array, unscaler, n_features):
            return unscaler(array.reshape(-1, n_features)).reshape(
                num_runs, steps, n_features
            )

        ar_preds_unscaled = unscale(all_preds_scaled, self._unscale_targets, n_target)
        trues_unscaled = unscale(all_trues_scaled, self._unscale_targets, n_target)
        inputs_unscaled = unscale(all_inputs_scaled, self._unscale_inputs, n_input)

        logger.info("Computing teacher-forced (single-step) predictions...")
        tf_preds_unscaled = unscale(
            self._teacher_forced_predictions(all_inputs_scaled, all_trues_scaled),
            self._unscale_targets,
            n_target,
        )

        target_names = list(datamodule.target.keys())
        forcing_names = [
            key
            for key, _ in sorted(datamodule.col_index_map.items(), key=lambda x: x[1])
        ]
        logger.info(
            f"Test set: {num_runs} trajectories, {steps} steps, {n_target} targets"
        )

        per_target_metrics = self._report_pointwise_metrics(
            trues_unscaled, ar_preds_unscaled, target_names, steps
        )

        evaluation.report_mare_comparison(
            trues_unscaled,
            ar_preds_unscaled,
            tf_preds_unscaled,
            target_names,
            self.log,
            per_target_metrics,
        )

        self._compute_jacobian_analysis(
            all_inputs_scaled, all_trues_scaled, target_names, forcing_names
        )
        self._compute_stepwise_importance(
            all_inputs_scaled, all_trues_scaled, target_names, forcing_names
        )
        if self.matrix_ode:
            self._compute_depletion_matrix_analysis(
                all_inputs_scaled, all_trues_scaled, target_names
            )

        evaluation.report_prediction_comparisons(
            trues_unscaled,
            ar_preds_unscaled,
            tf_preds_unscaled,
            target_names,
            self.result_dir,
        )
        evaluation.report_error_growth(
            trues_unscaled,
            ar_preds_unscaled,
            tf_preds_unscaled,
            target_names,
            self.result_dir,
            self.log,
        )
        self._plot_trajectories(
            trues_unscaled, ar_preds_unscaled, inputs_unscaled, target_names
        )
        self._log_to_database(per_target_metrics)

        self._test_preds.clear()
        self._test_trues.clear()
        self._test_input_trajs.clear()

    def _report_pointwise_metrics(self, trues, ar_preds, target_names, steps):
        """Per-target MAE/RMSE/R² over all timesteps, plus the scatter figures."""
        n_target = len(target_names)
        flat_trues = trues.reshape(-1, n_target)
        flat_preds = ar_preds.reshape(-1, n_target)

        mae_per_output = metrics.mae(flat_trues, flat_preds)
        rmse_per_output = metrics.rmse(flat_trues, flat_preds)
        r2_per_output = metrics.r2(flat_trues, flat_preds)

        self.log("Mean Absolute Error (avg)", float(mae_per_output.mean()))
        self.log("Root Mean Squared Error (avg)", float(rmse_per_output.mean()))
        self.log("R-squared coefficient (avg)", float(r2_per_output.mean()))

        logger.info("TEST SET — UNSCALED metrics (Autoregressive):")
        logger.info(f"  R² (avg):   {r2_per_output.mean():.6f}")
        logger.info(f"  RMSE (avg): {rmse_per_output.mean():.6f}")
        logger.info(f"  MAE (avg):  {mae_per_output.mean():.6f}")

        per_target_metrics = []
        for idx, target_name in enumerate(target_names):
            output_dir = os.path.join(self.result_dir, target_name)
            os.makedirs(output_dir, exist_ok=True)

            plot.plot_predictions_vs_actuals(
                flat_trues[:, idx],
                flat_preds[:, idx],
                mae_per_output[idx],
                rmse_per_output[idx],
                r2_per_output[idx],
                output_dir,
            )
            plot.plot_residuals_combined(
                flat_trues[:, idx], flat_preds[:, idx], output_dir, steps_per_run=steps
            )

            per_target_metrics.append(
                {
                    "name": target_name,
                    "mae": float(mae_per_output[idx]),
                    "rmse": float(rmse_per_output[idx]),
                    "r2": float(r2_per_output[idx]),
                }
            )
        return per_target_metrics

    def _plot_trajectories(self, trues, ar_preds, inputs, target_names, num_to_plot=5):
        """A few individual trajectories per target, plus an all-runs overlay."""
        t_np = self.t_span.cpu().numpy()
        num_runs = trues.shape[0]

        for target_idx, target_name in enumerate(target_names):
            target_dir = os.path.join(self.result_dir, target_name)
            os.makedirs(target_dir, exist_ok=True)

            for i in range(min(num_to_plot, num_runs)):
                plot.plot_node_trajectory(
                    t_np,
                    ar_preds[i, :, target_idx],
                    trues[i, :, target_idx],
                    inputs[i, :, 0],  # power, the first forcing input
                    title=f"{target_name} — Test Trajectory {i + 1}",
                    save_path=os.path.join(target_dir, f"test_traj_{i + 1}.png"),
                )

            plot.plot_node_trajectory_summary(
                t_np,
                ar_preds[:, :, target_idx],
                trues[:, :, target_idx],
                title=f"{target_name} — All Test Trajectories ({num_runs} runs)",
                save_path=os.path.join(target_dir, "test_all_trajectories.png"),
            )

    def _log_to_database(self, per_target_metrics):
        if not hasattr(self.trainer.logger, "update_final_results"):
            return

        averages = {
            f"{key}_avg": float(np.mean([m[key] for m in per_target_metrics]))
            for key in ("mae", "rmse", "r2")
        }
        self.trainer.logger.update_final_results(
            train_losses=self._train_losses,
            val_losses=self._val_losses,
            val_r2_scores=[],
            val_mae_scores=[],
            test_metrics={**averages, "per_target": per_target_metrics},
        )

    # ── Teacher-forced predictions ───────────────────────────────────────────

    def _teacher_forced_predictions(self, all_inputs_scaled, all_trues_scaled):
        """Single-step predictions: integrate one dt from the true y(t).

        The DNN's teacher-forcing analogue. y(t) is the full state vector, so
        every isotope is fed ground truth at every step. The first timestep has
        no predecessor and is copied from the truth.

        Returns (num_runs, steps, n_target), in model units.
        """
        num_runs, steps, _ = all_trues_scaled.shape
        tf_preds = np.zeros_like(all_trues_scaled)
        tf_preds[:, 0, :] = all_trues_scaled[:, 0, :]

        device = self.device
        t_span = self.t_span.to(device)

        for start in range(0, num_runs, TRAJECTORY_BATCH):
            end = min(start + TRAJECTORY_BATCH, num_runs)

            inputs_batch = torch.tensor(
                all_inputs_scaled[start:end], dtype=torch.float32, device=device
            )
            trues_batch = torch.tensor(
                all_trues_scaled[start:end], dtype=torch.float32, device=device
            )
            self.func.set_forcing(t_span, inputs_batch)

            for t in range(steps - 1):
                with torch.no_grad():
                    pred = self._odeint(trues_batch[:, t, :], t_span[t : t + 2])[-1]
                tf_preds[start:end, t + 1, :] = pred.cpu().numpy()

        return tf_preds

    # ── Jacobian sensitivity ─────────────────────────────────────────────────

    def compute_state_jacobian(self, t_eval, y_eval, forcing_eval, t_idx):
        """∂f/∂y at one timestep, for every trajectory in the batch."""
        with torch.inference_mode(False):
            # Cloning escapes inference-mode tracking, which forbids autograd.
            t_eval = t_eval.clone()
            forcing_eval = forcing_eval.clone()
            y_eval = y_eval.clone().requires_grad_(True)

            was_training = self.func.training
            self.func.train()
            self.func.set_forcing(t_eval, forcing_eval)

            dydt = self.func(t_eval[t_idx], y_eval)
            jacobians = [
                torch.autograd.grad(dydt[:, i].sum(), y_eval, retain_graph=True)[0]
                for i in range(dydt.shape[-1])
            ]

            self.func.train(was_training)
            return torch.stack(jacobians, dim=1)

    def compute_forcing_jacobian(self, t_eval, y_eval, forcing_eval, t_idx):
        """∂f/∂u at one timestep, attributed to the forcing interval actually read."""
        with torch.inference_mode(False):
            t_eval = t_eval.clone()
            y_eval = y_eval.clone().detach()
            forcing_eval = forcing_eval.clone().requires_grad_(True)

            was_training = self.func.training
            self.func.train()
            self.func.set_forcing(t_eval, forcing_eval)

            dydt = self.func(t_eval[t_idx], y_eval)
            with torch.no_grad():
                interp_idx = self.func.forcing_index(t_eval[t_idx])

            jacobians = [
                torch.autograd.grad(dydt[:, i].sum(), forcing_eval, retain_graph=True)[
                    0
                ][:, interp_idx, :]
                for i in range(dydt.shape[-1])
            ]

            self.func.train(was_training)
            return torch.stack(jacobians, dim=1)

    def _compute_jacobian_analysis(
        self, all_inputs_scaled, all_trues_scaled, target_names, forcing_names
    ):
        """State and forcing Jacobians across the test set.

        Produces mean |∂f/∂y| and |∂f/∂u| heatmaps, a per-target sensitivity bar
        chart, and the evolution of both along the trajectory.
        """
        logger.info(f"\n{'=' * 20}")
        logger.info("JACOBIAN SENSITIVITY ANALYSIS")
        logger.info(f"{'=' * 20}")

        num_runs, steps, _ = all_inputs_scaled.shape
        device = self.device
        t_span = self.t_span.to(device)

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

            inputs_batch = torch.tensor(
                all_inputs_scaled[start:end], dtype=torch.float32, device=device
            )
            trues_batch = torch.tensor(
                all_trues_scaled[start:end], dtype=torch.float32, device=device
            )

            for t_idx in timestep_indices:
                y_t = trues_batch[:, t_idx, :]
                abs_state_jac = (
                    self.compute_state_jacobian(t_span, y_t, inputs_batch, int(t_idx))
                    .abs()
                    .cpu()
                    .numpy()
                )
                abs_forcing_jac = (
                    self.compute_forcing_jacobian(t_span, y_t, inputs_batch, int(t_idx))
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
            save_path=os.path.join(self.result_dir, "jacobian_state_heatmap.png"),
        )
        plot.plot_jacobian_heatmap(
            mean_forcing_jac,
            forcing_names,
            target_names,
            title="Forcing Sensitivity: Mean |∂f/∂u|",
            xlabel="Forcing input (u_j)",
            ylabel="Derivative (dy_i/dt)",
            save_path=os.path.join(self.result_dir, "jacobian_forcing_heatmap.png"),
        )

        all_names = target_names + forcing_names
        combined_jac = np.concatenate([mean_state_jac, mean_forcing_jac], axis=1)
        combined_std = np.concatenate([std_state_jac, std_forcing_jac], axis=1)

        sorted_timesteps = sorted(timestep_indices)
        time_fractions = self.t_span.cpu().numpy()[sorted_timesteps]

        for idx, t_name in enumerate(target_names):
            output_dir = os.path.join(self.result_dir, t_name)
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

        logger.info(f"\n  Jacobian analysis plots saved to: {self.result_dir}")

    # ── Depletion matrix (matrix ODE only) ───────────────────────────────────

    def _get_unscaling_matrix(self, target_names):
        """Elementwise factors converting the scaled matrix to units of 1/day.

        In scaled space dy_s/dt_n = A_s @ y_s; in physical space
        A_phys[i,j] = A_s[i,j] * (range_i / range_j) / T_total_days. This ignores
        the MinMax offset, which is exact only where a target's minimum is zero —
        see AUDIT.md for why that matters for U238.

        Returns (scale_matrix, T_total_days, time_unit).
        """
        dm = self.trainer.datamodule
        n_target = len(target_names)

        # Scaled 0 and 1 invert to the physical min and max, so their difference
        # is the per-feature range without reaching into the scaler's internals.
        phys_min = dm.target_scaler.inverse_transform(np.zeros((1, n_target)))[0]
        phys_max = dm.target_scaler.inverse_transform(np.ones((1, n_target)))[0]
        ranges = phys_max - phys_min

        # Hardcoded: the training data actually spans 990 days, so every
        # coefficient below is ~1% low. Left as-is deliberately — changing it
        # moves the published depletion-matrix figure. See AUDIT.md.
        T_total_days = 1000

        logger.info(f"  Physical time span: {T_total_days:.2f} days")
        for idx, name in enumerate(target_names):
            logger.info(f"  {name} range: {ranges[idx]:.6e}")

        return np.outer(ranges, 1.0 / ranges) / T_total_days, T_total_days, "days"

    def _compute_depletion_matrix_analysis(
        self, all_inputs_scaled, all_trues_scaled, target_names, max_runs=20
    ):
        """Extract and visualise the learned depletion matrix A(t) over time."""
        logger.info(f"\n{'=' * 20}")
        logger.info("DEPLETION MATRIX ANALYSIS")
        logger.info(f"{'=' * 20}")

        num_runs, steps, _ = all_inputs_scaled.shape
        device = self.device
        t_span = self.t_span.to(device)

        scale_matrix, _, time_unit = self._get_unscaling_matrix(target_names)

        max_runs = min(max_runs, num_runs)
        timestep_indices = np.linspace(0, steps - 1, min(30, steps), dtype=int)
        matrices_by_time = {t: [] for t in timestep_indices}

        for start in range(0, max_runs, JACOBIAN_BATCH):
            end = min(start + JACOBIAN_BATCH, max_runs)

            inputs_batch = torch.tensor(
                all_inputs_scaled[start:end], dtype=torch.float32, device=device
            )
            trues_batch = torch.tensor(
                all_trues_scaled[start:end], dtype=torch.float32, device=device
            )
            self.func.set_forcing(t_span, inputs_batch)

            for t_idx in timestep_indices:
                with torch.no_grad():
                    forcing_t = self.func._interpolate_forcing(t_span[int(t_idx)])
                    A = self.func._build_matrix(
                        forcing_t, trues_batch[:, int(t_idx), :]
                    )

                for b in range(A.shape[0]):
                    matrices_by_time[t_idx].append(A[b].cpu().numpy() * scale_matrix)

        all_matrices = [m for t in timestep_indices for m in matrices_by_time[t]]
        plot.plot_depletion_matrix_mean(
            np.mean(all_matrices, axis=0),
            np.std(all_matrices, axis=0),
            target_names,
            time_unit,
            os.path.join(self.result_dir, "depletion_matrix_mean.png"),
        )

        sorted_ts = sorted(timestep_indices)
        plot.plot_depletion_matrix_evolution(
            self.t_span.cpu().numpy()[sorted_ts],
            np.array([np.mean(matrices_by_time[t], axis=0) for t in sorted_ts]),
            np.array([np.std(matrices_by_time[t], axis=0) for t in sorted_ts]),
            target_names,
            time_unit,
            max_runs,
            os.path.join(self.result_dir, "depletion_matrix_evolution.png"),
        )

    # ── Per-step teacher-forced importance ───────────────────────────────────

    def _compute_stepwise_importance(
        self,
        all_inputs_scaled,
        all_trues_scaled,
        target_names,
        forcing_names,
        n_permutations=5,
        seed=0,
        max_timesteps=50,
    ):
        """Per-step teacher-forced permutation importance.

        At each sampled timestep t the true y(t) is the initial condition; one
        feature is replaced with values from a randomly chosen donor run; the
        model integrates one step; and the change in |error| at t+1 is the
        feature's importance there.

        Sampling per step is what makes the zero-start isotopes measurable: at
        t=0 U239 is zero in every run so permuting it does nothing, but by t=5 it
        has built up and permuting it registers. The time-evolution plot shows
        exactly when each isotope starts mattering.

        Returns (mean_delta, sem_delta, mean_pct, sem_pct, mean_by_time).
        """
        logger.info(f"\n{'=' * 20}")
        logger.info(f"PER-STEP TEACHER-FORCED IMPORTANCE (K={n_permutations})")
        logger.info(f"{'=' * 20}")

        num_runs, steps, n_input = all_inputs_scaled.shape
        n_state = all_trues_scaled.shape[2]
        n_features = n_input + n_state
        rng = np.random.default_rng(seed)

        all_feature_names = list(forcing_names) + list(target_names)
        feature_types = ["Forcing"] * n_input + ["State"] * n_state

        # Stop at steps-2 so that t+1 always exists.
        timestep_indices = np.linspace(
            0, steps - 2, min(max_timesteps, steps - 1), dtype=int
        )
        n_sampled = len(timestep_indices)
        t_fractions = self.t_span.cpu().numpy()[timestep_indices]

        logger.info(
            f"  {num_runs} runs × {n_sampled} sampled steps × "
            f"{n_features} features × {n_permutations} permutations"
        )

        # per_run_avg:  (runs, features, states)   time-averaged ΔMAE per run
        # mean_by_time: (sampled, features, states) mean ΔMAE at each timestep
        per_run_avg = np.zeros((num_runs, n_features, n_state))
        mean_by_time = np.zeros((n_sampled, n_features, n_state))

        # The unperturbed forcing is reused by the baseline and by every state
        # permutation — the large majority of the thousands of solves below — so
        # it is converted once instead of on each call.
        inputs_tensor = torch.tensor(
            all_inputs_scaled, dtype=torch.float32, device=self.device
        )

        for t_enum, t_idx in enumerate(timestep_indices):
            if t_enum % 10 == 0:
                logger.info(f"  Timestep {t_enum + 1}/{n_sampled} (idx={t_idx})")

            y_t = all_trues_scaled[:, t_idx, :]
            y_tp1_unscaled = self._unscale_targets(all_trues_scaled[:, t_idx + 1, :])

            base_pred = self._single_step_batch(y_t, t_idx, inputs_tensor)
            base_ae = np.abs(self._unscale_targets(base_pred) - y_tp1_unscaled)

            for j in range(n_features):
                deltas_k = np.zeros((n_permutations, num_runs, n_state))

                for k in range(n_permutations):
                    perm = rng.permutation(num_runs)

                    if j < n_input:
                        # Forcing: swap column j at this timestep only. ZOH means
                        # forcing[:, t_idx, :] governs the whole interval.
                        perturbed_forcing = all_inputs_scaled.copy()
                        perturbed_forcing[:, t_idx, j] = all_inputs_scaled[
                            perm, t_idx, j
                        ]
                        pert_pred = self._single_step_batch(
                            y_t, t_idx, perturbed_forcing
                        )
                    else:
                        # State: swap one isotope in y(t); everything else stays
                        # at ground truth.
                        j_s = j - n_input
                        perturbed_y_t = y_t.copy()
                        perturbed_y_t[:, j_s] = y_t[perm, j_s]
                        pert_pred = self._single_step_batch(
                            perturbed_y_t, t_idx, inputs_tensor
                        )

                    pert_ae = np.abs(self._unscale_targets(pert_pred) - y_tp1_unscaled)
                    deltas_k[k] = pert_ae - base_ae

                delta_at_step = deltas_k.mean(axis=0)
                per_run_avg[:, j, :] += delta_at_step / n_sampled
                mean_by_time[t_enum, j, :] = delta_at_step.mean(axis=0)

        mean_delta = per_run_avg.mean(axis=0)
        sem_delta = per_run_avg.std(axis=0, ddof=1) / np.sqrt(num_runs)

        # Normalise to percentages per run, then average, so the spread reflects
        # run-to-run variation rather than the magnitude of the errors.
        delta_clipped = np.clip(per_run_avg, 0.0, None)
        totals = delta_clipped.sum(axis=1, keepdims=True)
        pct_per_run = np.where(
            totals > 0, 100.0 * delta_clipped / np.maximum(totals, 1e-30), 0.0
        )
        mean_pct = pct_per_run.mean(axis=0)
        sem_pct = pct_per_run.std(axis=0, ddof=1) / np.sqrt(num_runs)

        self._log_stepwise_importance(
            mean_delta,
            sem_delta,
            mean_pct,
            sem_pct,
            all_feature_names,
            feature_types,
            target_names,
            n_features,
        )
        self._write_stepwise_importance_tables(
            mean_delta,
            sem_delta,
            mean_pct,
            sem_pct,
            all_feature_names,
            feature_types,
            target_names,
            num_runs,
            n_sampled,
            n_permutations,
        )

        plot.plot_stepwise_importance_bar(
            mean_pct,
            sem_pct,
            all_feature_names,
            feature_types,
            target_names,
            num_runs,
            self.result_dir,
        )
        plot.plot_stepwise_importance_over_time(
            mean_by_time,
            t_fractions,
            all_feature_names,
            feature_types,
            target_names,
            self.result_dir,
        )

        return mean_delta, sem_delta, mean_pct, sem_pct, mean_by_time

    def _log_stepwise_importance(
        self,
        mean_delta,
        sem_delta,
        mean_pct,
        sem_pct,
        feature_names,
        feature_types,
        target_names,
        n_features,
    ):
        for k, tname in enumerate(target_names):
            logger.info(f"\n  Per-step importance — target: {tname}")
            logger.info(
                f"    {'Feature':<28} {'Type':<10} "
                f"{'ΔMAE (mean±SEM)':>28} {'Imp. [%]':>18}"
            )
            for j in np.argsort(mean_pct[:, k])[::-1]:
                logger.info(
                    f"    {feature_names[j]:<28} {feature_types[j]:<10} "
                    f"{mean_delta[j, k]:>10.4e} ± {sem_delta[j, k]:.2e}   "
                    f"{mean_pct[j, k]:>7.2f} ± {sem_pct[j, k]:.2f}"
                )
            for j in range(n_features):
                safe = feature_names[j].replace(" ", "_")
                self.log(f"{tname}/stepwise_pct/{safe}", float(mean_pct[j, k]))
                self.log(f"{tname}/stepwise_dMAE/{safe}", float(mean_delta[j, k]))

    def _write_stepwise_importance_tables(
        self,
        mean_delta,
        sem_delta,
        mean_pct,
        sem_pct,
        feature_names,
        feature_types,
        target_names,
        num_runs,
        n_sampled,
        n_permutations,
    ):
        """One CSV per target, plus a markdown dump for pasting into the write-up."""
        _print_unicode_safe("\n" + "=" * 70)
        _print_unicode_safe(
            "PER-STEP IMPORTANCE TABLES — paste everything between the markers"
        )
        _print_unicode_safe("=" * 70)
        _print_unicode_safe("<<<BEGIN_STEPWISE_IMPORTANCE_TABLES>>>")
        _print_unicode_safe(
            f"\n_Per-step teacher-forced permutation importance — "
            f"mean ± SEM across {num_runs} runs, "
            f"{n_sampled} sampled timesteps, K={n_permutations}_\n"
        )

        for k, tname in enumerate(target_names):
            order = np.argsort(mean_pct[:, k])[::-1]

            _print_unicode_safe(f"#### Target: `{tname}`\n")
            _print_unicode_safe("| Feature | Type | ΔMAE | MAE Imp. [%] |")
            _print_unicode_safe("|---|---|---|---|")
            for j in order:
                _print_unicode_safe(
                    f"| {feature_names[j]} | {feature_types[j]} | "
                    f"{mean_delta[j, k]:.4e} ± {sem_delta[j, k]:.2e} | "
                    f"{mean_pct[j, k]:.2f} ± {sem_pct[j, k]:.2f} |"
                )
            _print_unicode_safe()

            output_dir = os.path.join(self.result_dir, tname)
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "stepwise_importance.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Feature",
                        "Type",
                        "dMAE_mean",
                        "dMAE_SEM",
                        "MAE_Imp_pct_mean",
                        "MAE_Imp_pct_SEM",
                    ]
                )
                writer.writerows(
                    [
                        feature_names[j],
                        feature_types[j],
                        f"{mean_delta[j, k]:.6e}",
                        f"{sem_delta[j, k]:.6e}",
                        f"{mean_pct[j, k]:.4f}",
                        f"{sem_pct[j, k]:.4f}",
                    ]
                    for j in order
                )
            logger.info(f"  Stepwise importance CSV: {csv_path}")

        _print_unicode_safe("<<<END_STEPWISE_IMPORTANCE_TABLES>>>")
        _print_unicode_safe("=" * 70 + "\n")

    def _single_step_batch(self, y_t_np, t_idx, forcing_profiles):
        """Integrate one teacher-forced step for every run, in batches.

        y_t_np is (runs, n_state); *forcing_profiles* is (runs, steps, n_input),
        either numpy or an already-built tensor. The importance sweep calls this
        thousands of times with the same unperturbed forcing, so passing the
        tensor lets the caller convert it once.

        Returns (runs, n_state) in model units.
        """
        num_runs, n_state = y_t_np.shape
        device = self.device
        t_span = self.t_span.to(device)
        t_short = t_span[t_idx : t_idx + 2]

        forcing = (
            forcing_profiles
            if torch.is_tensor(forcing_profiles)
            else torch.tensor(forcing_profiles, dtype=torch.float32, device=device)
        )

        preds = np.zeros((num_runs, n_state))
        with torch.no_grad():
            for start in range(0, num_runs, SINGLE_STEP_BATCH):
                end = min(start + SINGLE_STEP_BATCH, num_runs)
                y_batch = torch.tensor(
                    y_t_np[start:end], dtype=torch.float32, device=device
                )
                self.func.set_forcing(t_span, forcing[start:end])
                preds[start:end] = self._odeint(y_batch, t_short)[-1].cpu().numpy()

        return preds

    # ── Predict / optimiser ──────────────────────────────────────────────────

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        target_pred, target_true = self._forward_batch(batch)
        return {
            "pred": target_pred.squeeze(-1).cpu(),
            "true": target_true.squeeze(-1).cpu(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.cfg.train.lr_scheduler_patience,
        )

        logger.info(
            f"Optimizer: AdamW | LR: {self.cfg.train.learning_rate} | "
            f"Weight Decay: {self.cfg.train.weight_decay}"
        )
        logger.info(
            f"Scheduler: ReduceLROnPlateau | "
            f"Patience: {self.cfg.train.lr_scheduler_patience} | Factor: 0.5"
        )

        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]
