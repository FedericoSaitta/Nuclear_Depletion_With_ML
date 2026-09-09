import lightning as L
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

from nuclear_surrogates import evaluation
from nuclear_surrogates.datamodule.dataset_helper import ordered_names
from nuclear_surrogates.models.model_architectures import Deep_Neural_Network
from nuclear_surrogates.models.model_helper import get_loss_fn
from nuclear_surrogates.utils import metrics, plot
from nuclear_surrogates.utils.paths import result_dir

# Shuffles per feature in the permutation-importance sweep. It multiplies the
# number of forward passes, and the whole test set goes through each one.
IMPORTANCE_REPEATS = 5


class DNN_Model(L.LightningModule):
    """Single-step surrogate: state(t) -> state(t+1), or the change between them."""

    def __init__(self, config_object):
        super().__init__()
        self.cfg = config_object
        # Embed the config in the checkpoint, as NODE_Model does, so a DNN
        # .ckpt is self-describing instead of needing its original YAML.
        self.save_hyperparameters(
            {"config": OmegaConf.to_container(config_object, resolve=True)}
        )

        self.n_inputs = len(config_object.dataset.inputs)
        self.n_outputs = len(config_object.dataset.targets)
        self.dropout_prob = config_object.model.dropout_probability
        self.NN_layers = config_object.model.layers
        self.activation = config_object.model.activation
        self.output_activation = config_object.model.output_activation
        self.residual_connections = config_object.model.residual_connections

        self.loss_fn = get_loss_fn(config_object.train.loss)

        # Residual connections are skipped where adjacent layer widths differ.
        self.residual_map = [
            in_size == out_size
            for in_size, out_size in zip(
                self.NN_layers[:-1], self.NN_layers[1:], strict=False
            )
        ]
        if self.residual_connections and not any(self.residual_map):
            logger.error(
                "Residual connections requested, but no adjacent layers have "
                "matching input/output dimensions."
            )

        self._init_tracking_variables()
        self.result_dir = result_dir(config_object)

        self.model = Deep_Neural_Network(
            self.n_inputs,
            self.n_outputs,
            self.NN_layers,
            self.dropout_prob,
            self.activation,
            self.output_activation,
            self.residual_connections,
        )

    def _init_tracking_variables(self):
        self.train_losses = []
        self.val_losses = []
        self.test_predictions = []
        self.test_labels = []
        self.test_inputs = []
        self.test_targets_scaled = []
        self.val_preds_epoch = []
        self.val_targets_epoch = []

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
        x, y = batch
        loss = self.loss_fn(self.model(x), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.train_losses.append(self.trainer.callback_metrics["train_loss"].item())

    def on_train_end(self):
        # Not gated on `draw_analyses`: `nucml plots` cannot regenerate this one,
        # because it never runs `fit`. See `draw_analyses`.
        plot.plot_losses(self.train_losses, self.val_losses, self.result_dir)

    # ── Validation ───────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        # R² and MAE are reported in physical units, not model units.
        target_scaler = self.trainer.datamodule.target_scaler
        self.val_preds_epoch.append(
            target_scaler.inverse_transform(y_hat.detach().cpu().numpy())
        )
        self.val_targets_epoch.append(target_scaler.inverse_transform(y.cpu().numpy()))
        return loss

    def on_validation_epoch_end(self):
        self.val_losses.append(self.trainer.callback_metrics["val_loss"].item())

        all_preds = np.concatenate(self.val_preds_epoch, axis=0)
        all_targets = np.concatenate(self.val_targets_epoch, axis=0)

        r2 = float(metrics.r2(all_targets, all_preds).mean())
        self.log("val_r2", r2, prog_bar=True)

        mae = float(metrics.mae(all_targets, all_preds).mean())
        self.log("val_mae", mae, prog_bar=True)

        self.val_preds_epoch = []
        self.val_targets_epoch = []

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        target_scaler = self.trainer.datamodule.target_scaler
        return {
            "labels": torch.from_numpy(
                target_scaler.inverse_transform(y.cpu().numpy())
            ),
            "predictions": torch.from_numpy(
                target_scaler.inverse_transform(self.model(x).cpu().numpy())
            ),
        }

    # ── Test ─────────────────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.test_inputs.append(x.cpu().numpy())
        self.test_targets_scaled.append(y.cpu().numpy())

        res = self.predict_step(batch, batch_idx)
        self.test_predictions.append(res["predictions"].cpu().numpy())
        self.test_labels.append(res["labels"].cpu().numpy())

    def on_test_epoch_end(self):
        datamodule = self.trainer.datamodule
        # Array order, not config-dict order: the prediction columns are laid
        # out by target_index_map, and a config listing targets in a different
        # order than they appear among the inputs would otherwise mislabel
        # every per-target metric and figure.
        target_names = ordered_names(datamodule.target_index_map)

        test_data = self._prepare_test_data()
        y_true_test = test_data["labels"]
        y_pred_test = test_data["predictions"]
        logger.info(
            f"Test set shape - True: {y_true_test.shape}, Pred: {y_pred_test.shape}"
        )

        # Everything below works on (runs, steps, targets) arrays in physical
        # units, which is the shape the shared report functions expect.
        trues, ar_preds, tf_preds = self._build_trajectories(
            test_data, y_pred_test, datamodule, target_names
        )

        # R2/MAE/RMSE are taken on the free-running trajectory in atom/b-cm,
        # flattened over runs and steps — the same call the NODE makes with the
        # same arguments (`neural_ode.on_test_epoch_end`). They used to be taken
        # on `y_pred_test`, the raw one-step output, which under
        # `target_delta_conc` is a *change* per 10-day step: a different
        # quantity in different units from the NODE's, so the two models'
        # headline numbers could not be read side by side.
        n_target = trues.shape[2]
        per_target_metrics, mae_arr, rmse_arr, r2_arr = evaluation.report_per_target(
            trues.reshape(-1, n_target),
            ar_preds.reshape(-1, n_target),
            target_names,
            self.result_dir,
            self.log,
            steps_per_run=datamodule.samples_per_run,
            figures=self.draw_analyses,
        )

        evaluation.report_mare_comparison(
            trues, ar_preds, tf_preds, target_names, self.log, per_target_metrics
        )
        if self.draw_analyses:
            self._compute_feature_importance(
                target_names, datamodule, datamodule.test_dataloader()
            )
            evaluation.report_prediction_comparisons(
                trues, ar_preds, tf_preds, target_names, self.result_dir
            )
            evaluation.report_error_growth(
                trues, ar_preds, tf_preds, target_names, self.result_dir, self.log
            )
            evaluation.report_trajectories(
                # No time axis: the DNN is a one-step map and has no notion of
                # how long a step is. Counting steps is the honest x-axis here.
                np.arange(trues.shape[1]),
                trues,
                ar_preds,
                self._forcing_series(test_data, datamodule),
                target_names,
                self.result_dir,
                xlabel="Time step",
                forcing_name=ordered_names(datamodule.col_index_map)[0],
            )

        self._write_test_metrics(mae_arr, rmse_arr, r2_arr, per_target_metrics)

    def _forcing_series(self, test_data, datamodule):
        """The first input column per run, in physical units — the forcing.

        Mirrors the NODE, which plots `inputs[:, :, 0]`. Column 0 of
        `col_index_map` is the forcing for both models by construction: the
        config lists it before the isotope concentrations.
        """
        samples_per_run = datamodule.samples_per_run
        unscaled = datamodule.input_scaler.inverse_transform(test_data["inputs"])
        forcing_idx = ordered_names(datamodule.col_index_map)[0]
        series = unscaled[:, datamodule.col_index_map[forcing_idx]]
        return series.reshape(len(series) // samples_per_run, samples_per_run)

    def _prepare_test_data(self):
        """Consolidate test data from batches."""
        return {
            "inputs": np.concatenate(self.test_inputs, axis=0),
            "targets_scaled": np.concatenate(self.test_targets_scaled, axis=0),
            "predictions": np.concatenate(self.test_predictions, axis=0),
            "labels": np.concatenate(self.test_labels, axis=0),
        }

    # ── Trajectory assembly ──────────────────────────────────────────────────

    def _build_trajectories(self, test_data, y_pred_test, datamodule, target_names):
        """Roll out autoregressively, then shape everything as (runs, steps, targets).

        Every series comes back as absolute concentrations on the grid
        c(0) … c(N-1), which is the grid the NODE reports on, so the two models'
        metrics and figures line up step for step.

        With `target_delta_conc` the model works in concentration *changes*, so
        each run is integrated from the initial concentration carried in its
        first input row. The teacher-forced series is built one step from the
        *true* state rather than by accumulating predictions, matching
        `analysis.rollout.teacher_forced_predictions` — see
        `evaluation.teacher_forced_from_deltas`.
        """
        X_test = test_data["inputs"]
        samples_per_run = datamodule.samples_per_run
        n_runs = len(X_test) // samples_per_run
        shape = (n_runs, samples_per_run)

        ar_preds_dict, ar_trues_dict = metrics.model_autoregress(
            self.model,
            X_test,
            test_data["targets_scaled"],
            datamodule.input_scaler,
            datamodule.target_scaler,
            samples_per_run,
            datamodule.col_index_map,
            datamodule.target_index_map,
            delta_conc=datamodule.delta_conc,
        )

        trues, ar_preds, tf_preds = [], [], []
        for idx, target_name in enumerate(target_names):
            true_series = ar_trues_dict[target_name].reshape(shape)
            ar_series = ar_preds_dict[target_name].reshape(shape)
            tf_series = y_pred_test[:, idx].reshape(shape)

            # Every series needs this run's true starting concentration: to
            # integrate from under `target_delta_conc`, and to sit on the
            # NODE's grid either way. A target that is not also an input does
            # not have one. That used to fall through to a warning and leave the
            # channel in delta units, which now silently mixes units inside an
            # averaged R2 — so refuse instead. Every shipped config lists all
            # seven targets among its inputs.
            if target_name not in datamodule.col_index_map:
                raise ValueError(
                    f"target {target_name!r} is not also an input, so its "
                    f"initial concentration is unknown and its trajectory "
                    f"cannot be put in absolute units. Add it to "
                    f"`dataset.inputs`."
                )
            initial = self._initial_concentrations(
                X_test, datamodule, target_name, samples_per_run
            )

            if datamodule.delta_conc:
                true_series = evaluation.integrate_deltas(true_series, initial)
                ar_series = evaluation.integrate_deltas(ar_series, initial)
                # Built from the integrated truth, so it is one step from the
                # true state rather than from an accumulated prediction.
                tf_series = evaluation.teacher_forced_from_deltas(
                    tf_series, true_series
                )
            else:
                true_series, ar_series, tf_series = (
                    evaluation.align_to_initial(s, initial)
                    for s in (true_series, ar_series, tf_series)
                )

            trues.append(true_series)
            ar_preds.append(ar_series)
            tf_preds.append(tf_series)

        return tuple(
            np.stack(arrays, axis=-1) for arrays in (trues, ar_preds, tf_preds)
        )

    @staticmethod
    def _initial_concentrations(X_test, datamodule, target_name, samples_per_run):
        """Each run's starting concentration for *target_name*, in physical units."""
        first_rows = X_test[::samples_per_run]
        unscaled = datamodule.input_scaler.inverse_transform(first_rows)
        return unscaled[:, datamodule.col_index_map[target_name]]

    # ── Feature importance ───────────────────────────────────────────────────

    def _compute_feature_importance(self, target_names, datamodule, loader):
        """Permutation feature importance per target, under R² and MSE.

        Computes only. The logging, the CSVs, the markdown table and the bar
        charts are `evaluation.report_feature_importance`'s job, which is what
        puts this on the same footing as the NODE's per-step sweep — that one
        has always written its numbers out beside the figures, and this one used
        to leave nothing behind but a PNG.
        """
        logger.info(f"\n{'=' * 20}")
        logger.info(f"FEATURE IMPORTANCE ANALYSIS (K={IMPORTANCE_REPEATS})")
        logger.info(f"{'=' * 20}")

        feature_names = ordered_names(datamodule.col_index_map)
        # The same Forcing/State split the NODE's importance tables carry: a
        # feature that is also a target is the model's own state fed back in.
        target_set = set(target_names)
        feature_types = [
            "State" if name in target_set else "Forcing" for name in feature_names
        ]

        importances = {}
        for idx, target_name in enumerate(target_names):
            logger.info(f"\nComputing feature importance for: {target_name}")
            importances[target_name] = {
                metric_name: metrics.calculate_feature_importance(
                    self.model,
                    loader,
                    self.device,
                    n_repeats=IMPORTANCE_REPEATS,
                    metric={"name": metric_name, "direction": direction},
                    output_idx=idx,
                )
                for metric_name, direction in (
                    ("r2", "increasing"),
                    ("mse", "decreasing"),
                )
            }

        evaluation.report_feature_importance(
            importances,
            feature_names,
            feature_types,
            target_names,
            self.result_dir,
            self.log,
            n_repeats=IMPORTANCE_REPEATS,
        )

    # ── Bookkeeping ──────────────────────────────────────────────────────────

    def _write_test_metrics(self, mae_arr, rmse_arr, r2_arr, per_target_metrics):
        """Write the test metrics beside the figures they belong to.

        These are the only run outputs the bundle cannot carry: `write_bundle`
        runs before `trainer.test`, so at bundle time they do not exist yet.
        """
        evaluation.write_test_metrics(
            self.result_dir,
            {
                "mae_avg": float(mae_arr.mean()),
                "rmse_avg": float(rmse_arr.mean()),
                "r2_avg": float(r2_arr.mean()),
                "per_target": per_target_metrics,
            },
        )

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
