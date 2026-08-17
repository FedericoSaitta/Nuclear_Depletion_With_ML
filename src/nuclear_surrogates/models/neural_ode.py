"""The Neural-ODE surrogate: integrate dy/dt = f(forcing(t), y) over a whole run.

This module is the model — construction, the solver call, the Lightning training
and evaluation hooks. The post-hoc analysis passes that run once at test time
(teacher-forced rollout, Jacobians, the depletion matrix, permutation
importance) live in `nuclear_surrogates.analysis`, and the metric maths shared
with the DNN lives in `nuclear_surrogates.evaluation`.
"""

import lightning as L
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from torchdiffeq import odeint, odeint_adjoint

from nuclear_surrogates import analysis, evaluation
from nuclear_surrogates.datamodule.dataset_helper import ordered_names
from nuclear_surrogates.models.model_architectures import ODEFuncForced, ODEFuncMatrix
from nuclear_surrogates.models.model_helper import get_loss_fn
from nuclear_surrogates.utils import plot
from nuclear_surrogates.utils.paths import result_dir


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

        common_kwargs = {
            "method": self.cfg.train.solver,
            "rtol": self.rtol,
            "atol": self.atol,
            "options": options or None,
        }

        if self.use_adjoint:
            return odeint_adjoint(
                self.func,
                y0,
                t_span,
                adjoint_rtol=self.adjoint_rtol,
                adjoint_atol=self.adjoint_atol,
                adjoint_method=self.adjoint_method,
                **common_kwargs,
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

    def solve_context(self):
        """Bundle what the analysis passes need — see `nuclear_surrogates.analysis`."""
        return analysis.SolveContext(
            func=self.func,
            odeint=self._odeint,
            t_span=self.t_span,
            device=self.device,
            unscale_targets=self._unscale_targets,
            result_dir=self.result_dir,
        )

    def _teacher_forced_predictions(self, all_inputs_scaled, all_trues_scaled):
        """Single-step predictions from the true y(t) — see `analysis.rollout`."""
        return analysis.teacher_forced_predictions(
            self.solve_context(), all_inputs_scaled, all_trues_scaled
        )

    @property
    def draw_analyses(self):
        """Whether to emit the evaluation figures and run the post-hoc analyses.

        `--no-analyses` turns this off. The line it draws is regenerability:
        everything gated on it can be recreated later with `nucml plots` from
        the bundle, so skipping it during a hyperparameter sweep costs nothing
        permanent. The training loss curve is deliberately **not** gated — it is
        written by `on_train_end`, which `plots` never reaches, so it is the one
        figure that would be gone for good. Metrics are computed either way.
        """
        return self.cfg.runtime.get("analyses", True)

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
        # Not gated on `draw_analyses`: `nucml plots` cannot regenerate this one,
        # because it never runs `fit`. See `draw_analyses`.
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

        # Array order, not config-dict order — a config listing targets in a
        # different order than they appear among the inputs would otherwise
        # mislabel every per-target metric, figure and CSV.
        target_names = ordered_names(datamodule.target_index_map)
        forcing_names = ordered_names(datamodule.col_index_map)
        logger.info(
            f"Test set: {num_runs} trajectories, {steps} steps, {n_target} targets"
        )

        per_target_metrics, *_ = evaluation.report_per_target(
            trues_unscaled.reshape(-1, n_target),
            ar_preds_unscaled.reshape(-1, n_target),
            target_names,
            self.result_dir,
            self.log,
            steps_per_run=steps,
            figures=self.draw_analyses,
        )

        evaluation.report_mare_comparison(
            trues_unscaled,
            ar_preds_unscaled,
            tf_preds_unscaled,
            target_names,
            self.log,
            per_target_metrics,
        )

        if self.draw_analyses:
            # These re-integrate the whole test set — the Jacobian sweep and the
            # per-step importance are the most expensive things a run does, and
            # the reason `--no-analyses` exists.
            ctx = self.solve_context()
            analysis.jacobian_analysis(
                ctx, all_inputs_scaled, all_trues_scaled, target_names, forcing_names
            )
            analysis.stepwise_importance(
                ctx,
                all_inputs_scaled,
                all_trues_scaled,
                target_names,
                forcing_names,
                self.log,
            )
            if self.matrix_ode:
                analysis.depletion_matrix_analysis(
                    ctx, all_inputs_scaled, all_trues_scaled, target_names
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
            evaluation.report_trajectories(
                self.t_span.cpu().numpy(),
                trues_unscaled,
                ar_preds_unscaled,
                inputs_unscaled[:, :, 0],  # power, the first forcing input
                target_names,
                self.result_dir,
                xlabel="Time",
            )

        self._write_test_metrics(per_target_metrics)

        self._test_preds.clear()
        self._test_trues.clear()
        self._test_input_trajs.clear()

    def _write_test_metrics(self, per_target_metrics):
        """Write the test metrics beside the figures they belong to.

        These are the only run outputs the bundle cannot carry: `write_bundle`
        runs before `trainer.test`, so at bundle time they do not exist yet.
        """
        averages = {
            f"{key}_avg": float(np.mean([m[key] for m in per_target_metrics]))
            for key in ("mae", "rmse", "r2")
        }
        evaluation.write_test_metrics(
            self.result_dir, {**averages, "per_target": per_target_metrics}
        )

    # ── Predict / optimiser ──────────────────────────────────────────────────

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        target_pred, target_true = self._forward_batch(batch)
        return {"pred": target_pred.cpu(), "true": target_true.cpu()}

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
