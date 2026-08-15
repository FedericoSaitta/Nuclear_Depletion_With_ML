import os

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
        self.val_r2_scores = []
        self.val_mae_scores = []
        self.test_predictions = []
        self.test_labels = []
        self.test_inputs = []
        self.test_targets_scaled = []
        self.val_preds_epoch = []
        self.val_targets_epoch = []

    # ── Training ─────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self.model(x), y)
        # Deliberately not logged to the SQLite database: only final results are.
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.train_losses.append(self.trainer.callback_metrics["train_loss"].item())

    def on_train_end(self):
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
        self.val_r2_scores.append(r2)

        mae = float(metrics.mae(all_targets, all_preds).mean())
        self.log("val_mae", mae, prog_bar=True)
        self.val_mae_scores.append(mae)

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

        mae_arr, rmse_arr, r2_arr = self._compute_and_log_overall_metrics(
            y_true_test, y_pred_test
        )

        per_target_metrics = []
        for idx, target_name in enumerate(target_names):
            self._plot_single_output(
                target_name,
                y_true_test[:, idx],
                y_pred_test[:, idx],
                mae_arr[idx],
                rmse_arr[idx],
                r2_arr[idx],
                datamodule.samples_per_run,
            )
            per_target_metrics.append(
                {
                    "name": target_name,
                    "mae": float(mae_arr[idx]),
                    "rmse": float(rmse_arr[idx]),
                    "r2": float(r2_arr[idx]),
                }
            )

        # Everything below works on (runs, steps, targets) arrays in physical
        # units, which is the shape the shared report functions expect.
        trues, ar_preds, tf_preds = self._build_trajectories(
            test_data, y_pred_test, datamodule, target_names
        )

        evaluation.report_mare_comparison(
            trues, ar_preds, tf_preds, target_names, self.log, per_target_metrics
        )
        self._compute_feature_importance(
            target_names, datamodule, datamodule.test_dataloader()
        )
        evaluation.report_prediction_comparisons(
            trues, ar_preds, tf_preds, target_names, self.result_dir
        )
        evaluation.report_error_growth(
            trues, ar_preds, tf_preds, target_names, self.result_dir, self.log
        )

        self._log_to_database(mae_arr, rmse_arr, r2_arr, per_target_metrics)

    def _prepare_test_data(self):
        """Consolidate test data from batches."""
        return {
            "inputs": np.concatenate(self.test_inputs, axis=0),
            "targets_scaled": np.concatenate(self.test_targets_scaled, axis=0),
            "predictions": np.concatenate(self.test_predictions, axis=0),
            "labels": np.concatenate(self.test_labels, axis=0),
        }

    def _compute_and_log_overall_metrics(self, y_true, y_pred):
        """Per-output MAE, RMSE and R²; the averages go to the progress bar."""
        mae_per_output = metrics.mae(y_true, y_pred)
        rmse_per_output = metrics.rmse(y_true, y_pred)
        r2_per_output = metrics.r2(y_true, y_pred)

        self.log("Mean Absolute Error (avg)", float(mae_per_output.mean()))
        self.log("Root Mean Squared Error (avg)", float(rmse_per_output.mean()))
        self.log("R-squared coefficient (avg)", float(r2_per_output.mean()))

        return mae_per_output, rmse_per_output, r2_per_output

    def _plot_single_output(
        self, target_name, y_true, y_pred, mae, rmse, r2, samples_per_run
    ):
        output_dir = os.path.join(self.result_dir, target_name)
        os.makedirs(output_dir, exist_ok=True)
        plot.plot_predictions_vs_actuals(y_true, y_pred, mae, rmse, r2, output_dir)
        plot.plot_residuals_combined(
            y_true, y_pred, output_dir, steps_per_run=samples_per_run
        )

    # ── Trajectory assembly ──────────────────────────────────────────────────

    def _build_trajectories(self, test_data, y_pred_test, datamodule, target_names):
        """Roll out autoregressively, then shape everything as (runs, steps, targets).

        With `target_delta_conc` the model works in concentration *changes*, so
        each run's series is integrated back to absolute concentrations from the
        initial concentration carried in that run's first input row. A target
        that is not also an input has no such initial value, and is left as
        deltas — which is what the comparison then reports.
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
            series = [
                ar_trues_dict[target_name].reshape(shape),
                ar_preds_dict[target_name].reshape(shape),
                y_pred_test[:, idx].reshape(shape),
            ]

            if datamodule.delta_conc:
                if target_name in datamodule.col_index_map:
                    initial = self._initial_concentrations(
                        X_test, datamodule, target_name, samples_per_run
                    )
                    series = [evaluation.deltas_to_absolute(s, initial) for s in series]
                else:
                    logger.warning(
                        f"{target_name} is not an input, so its initial "
                        f"concentration is unknown — reporting deltas instead of "
                        f"absolute concentrations."
                    )

            for collection, values in zip(
                (trues, ar_preds, tf_preds), series, strict=False
            ):
                collection.append(values)

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
        """Permutation feature importance per target, under R² and MSE."""
        logger.info(f"\n{'=' * 20}")
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info(f"{'=' * 20}")

        feature_names = [
            key
            for key, _ in sorted(datamodule.col_index_map.items(), key=lambda x: x[1])
        ]

        for idx, target_name in enumerate(target_names):
            output_dir = os.path.join(self.result_dir, target_name)
            logger.info(f"\nComputing feature importance for: {target_name}")

            for metric_name, direction in (("r2", "increasing"), ("mse", "decreasing")):
                means, stds, baseline = metrics.calculate_feature_importance(
                    self.model,
                    loader,
                    self.device,
                    n_repeats=5,
                    metric={"name": metric_name, "direction": direction},
                    output_idx=idx,
                )
                plot.plot_feature_importance(
                    means,
                    stds,
                    feature_names,
                    baseline,
                    output_dir,
                    f"{metric_name}_score",
                    n_top=20,
                )

            logger.info(f"  Feature importance plots saved to: {output_dir}")

    # ── Bookkeeping ──────────────────────────────────────────────────────────

    def _log_to_database(self, mae_arr, rmse_arr, r2_arr, per_target_metrics):
        if not hasattr(self.trainer.logger, "update_final_results"):
            return

        self.trainer.logger.update_final_results(
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            val_r2_scores=self.val_r2_scores,
            val_mae_scores=self.val_mae_scores,
            test_metrics={
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
