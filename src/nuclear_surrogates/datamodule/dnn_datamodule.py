import lightning as L
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

import nuclear_surrogates.datamodule.data_scalers as data_scalers
import nuclear_surrogates.datamodule.dataset_helper as data_help
import nuclear_surrogates.utils.plot as plot
from nuclear_surrogates.datamodule.preprocessor import (
    Preprocessor,
    require_fitted_scalers,
)
from nuclear_surrogates.utils.paths import result_dir


class DNN_Datamodule(L.LightningDataModule):
    """Serves (state_t -> state_t+1) pairs, split 80/10/10 by whole runs."""

    def __init__(self, cfg_object):
        super().__init__()

        self.path_to_data = cfg_object.dataset.path_to_data
        self.path_to_inference_data = cfg_object.dataset.get(
            "path_to_inference_data", None
        )
        self.fraction_of_data = cfg_object.dataset.fraction_of_data
        self.train_batch_size = cfg_object.dataset.train.batch_size
        self.val_batch_size = cfg_object.dataset.val.batch_size

        self.num_workers = cfg_object.runtime.num_workers
        logger.info(f"Using {self.num_workers} cpus")

        # Seeded here rather than relying on the global RNG, so a datamodule
        # constructed outside main() (tests, packaging) splits identically.
        self.seed = cfg_object.runtime.get("seed", 42)

        # When set, the fitted scalers are LOADED rather than re-derived.
        # Required for inference; during training it is how `nucml plots`
        # replays a run against the scalers that run was trained with.
        self.preprocessor_path = cfg_object.dataset.get("preprocessor_path", None)
        self.preprocessor = None
        self.make_plots = cfg_object.runtime.get("analyses", True)

        # `modes.inference` sets this; setup() then takes _setup_inference.
        self.inference_mode = False

        self.inputs = data_scalers.create_scaler_dict(cfg_object.dataset["inputs"])
        self.target = data_scalers.create_scaler_dict(cfg_object.dataset["targets"])
        # Sequential 80/10/10 unless the config says otherwise.
        self.split = data_help.split_fractions(cfg_object, default=(0.8, 0.1, 0.1))
        self.delta_conc = cfg_object.dataset.target_delta_conc
        self.train_drop_last = cfg_object.train.drop_last

        self.result_dir = result_dir(cfg_object)
        # Guards against setting up twice when training and testing back to back.
        self._has_setup = False

    def setup(self, stage):
        if self._has_setup:
            return
        self._has_setup = True

        logger.info("Setting up the data module...")

        # Inputs first, then any target that is not also an input. Passing these
        # to read_data keeps the rest of the depletion chain off the heap.
        all_columns = list(self.inputs.keys()) + [
            k for k in self.target if k not in self.inputs
        ]

        if self.inference_mode:
            self._setup_inference(all_columns)
        else:
            self._setup_training(all_columns)

    # ── Reading ──────────────────────────────────────────────────────────────

    def _read_pairs(self, path, fraction, all_columns):
        """Read one file into the (state_t -> state_t+1) pairs the DNN consumes.

        `create_timeseries_targets` drops each run's last row, which has no
        successor inside its own run, so no pair ever straddles a run boundary.
        """
        data_df, self.run_length, self.time_array = data_help.read_data(
            path,
            fraction,
            drop_run_label=True,
            columns=all_columns,
        )
        data_help.print_dataset_stats(data_df)
        data_df = data_df.select(all_columns)

        self.input_data_arr, self.col_index_map = data_help.split_df(
            data_df, self.inputs.keys()
        )
        self.target_data_arr, self.target_index_map = data_help.split_df(
            data_df, self.target.keys()
        )

        X, Y = data_help.create_timeseries_targets(
            self.input_data_arr,
            self.target_data_arr,
            self.time_array,
            self.col_index_map,
            self.target_index_map,
            self.delta_conc,
        )

        # One (t -> t+1) pair per timestep except the last of each run, which has
        # no successor inside the run.
        self.samples_per_run = self.run_length - 1
        return X, data_help.ensure_2d(Y)

    # ── Training ─────────────────────────────────────────────────────────────

    def _setup_training(self, all_columns):
        X, Y = self._read_pairs(self.path_to_data, self.fraction_of_data, all_columns)

        # Split by whole runs, sequentially in time.
        train_frac, val_frac, test_frac = self.split
        X_train, X_val, X_test, y_train, y_val, y_test, split_info = (
            data_help.timeseries_train_val_test_split(
                X,
                Y,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                steps_per_run=self.samples_per_run,
                shuffle_within_train=True,
                rng=np.random.default_rng(self.seed),
            )
        )
        self.split_info = split_info

        if self.make_plots:
            self._plot_distributions(X_train, y_train, "Raw")

        # Fit (or load) the scalers, then apply them. Fitting happens on the
        # training split only — val/test are transform-only.
        span = float(self.time_array[: self.run_length - 1][-1] - self.time_array[0])

        if self.preprocessor_path:
            self.preprocessor = Preprocessor.load(self.preprocessor_path)
        else:
            self.preprocessor = Preprocessor.fit(
                self.inputs,
                self.target,
                X_train,
                y_train,
                self.col_index_map,
                self.target_index_map,
                t_days=span,
                t_days_data_span=span,
            )
        self.input_scaler = self.preprocessor.input_scaler
        self.target_scaler = self.preprocessor.target_scaler

        X_train, X_val, X_test = (
            self.input_scaler.transform(X_train),
            self.input_scaler.transform(X_val),
            self.input_scaler.transform(X_test),
        )
        y_train, y_val, y_test = (
            self.target_scaler.transform(y_train),
            self.target_scaler.transform(y_val),
            self.target_scaler.transform(y_test),
        )

        if self.make_plots:
            self._plot_distributions(X_train, y_train, "Scaled")

        self.train_dataset, self.val_dataset, self.test_dataset = (
            data_help.create_tensor_datasets(
                X_train, X_val, X_test, y_train, y_val, y_test
            )
        )
        logger.info(f"Training dataset size: {len(y_train)}")
        logger.info(f"Validation dataset size: {len(y_val)}")
        logger.info(f"Test dataset size: {len(y_test)}")

    # ── Inference ────────────────────────────────────────────────────────────

    def _setup_inference(self, all_columns):
        """Evaluate a frozen model on the whole of `path_to_inference_data`.

        The counterpart of `NODE_Datamodule._setup_inference`, and simpler:
        the DNN is a one-step map rather than an integration, so there is no
        time axis to renormalise. `t_days` never reaches the forward pass —
        `predict_step` and `test_step` only go through `target_scaler` — so the
        NODE's `_training_time_unit` correction has no analogue here and must
        not be copied in.

        What does have to match training is `dataset.target_delta_conc`, since
        it decides whether the targets are concentrations or differences. It
        travels in the bundle's config, so a bundle is consistent by
        construction.
        """
        if not self.path_to_inference_data:
            raise SystemExit(
                "Inference mode needs dataset.path_to_inference_data. "
                "With a bundle, pass --data <file>."
            )

        require_fitted_scalers(self.preprocessor_path)
        logger.info("Inference mode: loading fitted scalers from the bundle")
        self.preprocessor = Preprocessor.load(self.preprocessor_path)
        self.input_scaler = self.preprocessor.input_scaler
        self.target_scaler = self.preprocessor.target_scaler

        # The whole file, not a fraction of it, and no split: every pair is test.
        X, Y = self._read_pairs(self.path_to_inference_data, 1.0, all_columns)
        X = self.input_scaler.transform(X)
        Y = self.target_scaler.transform(Y)

        # Empty but correctly shaped, so Lightning can still build the loaders.
        empty_X = np.zeros((0, X.shape[1]), dtype=X.dtype)
        empty_y = np.zeros((0, Y.shape[1]), dtype=Y.dtype)
        self.train_dataset, self.val_dataset, self.test_dataset = (
            data_help.create_tensor_datasets(empty_X, empty_X, X, empty_y, empty_y, Y)
        )
        n_runs = len(Y) // self.samples_per_run if self.samples_per_run else 0
        logger.info(f"Test dataset size: {len(Y)} pairs from {n_runs} runs")

    def _plot_distributions(self, inputs, targets, prefix):
        plot.plot_data_distributions(
            inputs,
            self.col_index_map,
            save_dir=self.result_dir,
            name=f"{prefix}_Inputs",
        )
        plot.plot_data_distributions(
            targets,
            self.target_index_map,
            save_dir=self.result_dir,
            name=f"{prefix}_Targets",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # persistent_workers=True is a hard error when num_workers == 0.
            persistent_workers=self.num_workers > 0,
            drop_last=self.train_drop_last,
            pin_memory=False,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
