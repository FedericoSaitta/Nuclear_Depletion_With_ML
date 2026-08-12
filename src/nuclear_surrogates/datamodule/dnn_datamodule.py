from loguru import logger
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

# Local Imports
import nuclear_surrogates.datamodule.dataset_helper as data_help
import nuclear_surrogates.utils.plot as plot
import nuclear_surrogates.datamodule.data_scalers as data_scalers
from nuclear_surrogates.datamodule.preprocessor import Preprocessor
from nuclear_surrogates.utils.paths import result_dir


class DNN_Datamodule(L.LightningDataModule):
    def __init__(self, cfg_object):
        super().__init__()

        self.path_to_data = cfg_object.dataset.path_to_data
        self.fraction_of_data = cfg_object.dataset.fraction_of_data
        self.train_batch_size = cfg_object.dataset.train.batch_size
        self.val_batch_size = cfg_object.dataset.val.batch_size

        self.num_workers = cfg_object.runtime.num_workers
        logger.info(f"Using {self.num_workers} cpus")

        # Seeded here rather than relying on the global RNG, so a datamodule
        # constructed outside main() (tests, packaging) splits identically.
        self.seed = cfg_object.runtime.get("seed", 42)

        # When set, the fitted scalers are LOADED rather than re-derived.
        self.preprocessor_path = cfg_object.dataset.get("preprocessor_path", None)
        self.preprocessor = None
        self.make_plots = cfg_object.runtime.get("plots", True)

        # `modes.inference` sets this; the DNN has no inference branch, so it
        # must fail loudly rather than quietly testing on the training file's
        # own split and reporting it as a cross-dataset number (AUDIT.md §A3).
        self.inference_mode = False

        # Get the inputs and target dictionaries that include their respective scaling
        self.inputs = data_scalers.create_scaler_dict(cfg_object.dataset["inputs"])
        self.target = data_scalers.create_scaler_dict(cfg_object.dataset["targets"])
        self.delta_conc = cfg_object.dataset.target_delta_conc
        self.train_drop_last = cfg_object.train.drop_last

        # === Result output directory === #
        self.result_dir = result_dir(cfg_object)

        # Private variable to ensure set up is not done twice when calling training and test scripts back to back
        self._has_setup = False

    def setup(self, stage):
        if self._has_setup:
            return
        self._has_setup = True

        if self.inference_mode:
            raise NotImplementedError(
                "DNN inference mode is not implemented. `modes.inference` sets "
                "inference_mode, but this datamodule has no branch for it, so it "
                "would silently evaluate the TRAINING file's own test split — a "
                "cross-dataset claim made that way would be false (AUDIT.md §A3). "
                "Use runtime.model=NODE for inference, or implement the branch "
                "mirroring NODE_Datamodule._setup_inference."
            )

        logger.info("Setting up the data module...")

        # Obtain the df, the run length and the actualy time data such that we can plot
        data_df, self.run_length, self.time_array = data_help.read_data(
            self.path_to_data, self.fraction_of_data, drop_run_label=True
        )
        data_help.print_dataset_stats(data_df)

        # Preserve order: inputs first, then targets (no duplicates)
        all_columns = list(self.inputs.keys()) + [
            k for k in self.target.keys() if k not in self.inputs.keys()
        ]

        # Get the columns we are interested in this analysis
        data_df = data_help.filter_columns(data_df, all_columns)

        # Get the array and dictionary for inputs and output
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

        # Split data 80/10/10 -- Train/validation/Test, this is Time aware
        X_train, X_val, X_test, y_train, y_val, y_test, split_info = (
            data_help.timeseries_train_val_test_split(
                X,
                Y,
                train_frac=0.8,
                val_frac=0.1,
                test_frac=0.1,
                steps_per_run=100,
                shuffle_within_train=True,
                rng=np.random.default_rng(self.seed),
            )
        )
        self.split_info = split_info
        data_help.write_split_indices(self.result_dir, split_info, self.seed)

        if self.make_plots:
            plot.plot_data_distributions(
                X_train, self.col_index_map, save_dir=self.result_dir, name="Raw_Inputs"
            )
            plot.plot_data_distributions(
                y_train,
                self.target_index_map,
                save_dir=self.result_dir,
                name="Raw_Targets",
            )

        # Fit (or load) the scalers, then apply them. Fitting happens on the
        # training split only — val/test are transform-only.
        y_train, y_val, y_test = (
            data_help.ensure_2d(y_train),
            data_help.ensure_2d(y_val),
            data_help.ensure_2d(y_test),
        )
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

        self.X_test = X_test
        self.Y_test = y_test

        if self.make_plots:
            plot.plot_data_distributions(
                X_train,
                self.col_index_map,
                save_dir=self.result_dir,
                name="Scaled_Inputs",
            )
            plot.plot_data_distributions(
                y_train,
                self.target_index_map,
                save_dir=self.result_dir,
                name="Scaled_Targets",
            )

        # Create tensor datasets and log their sizes
        self.train_dataset, self.val_dataset, self.test_dataset = (
            data_help.create_tensor_datasets(
                X_train, X_val, X_test, y_train, y_val, y_test
            )
        )
        logger.info(f"Training dataset size: {len(y_train)}")
        logger.info(f"Validation dataset size: {len(y_val)}")
        logger.info(f"Test dataset size: {len(y_test)}")

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
