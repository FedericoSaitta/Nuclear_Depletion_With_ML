from loguru import logger
import lightning as L
from torch.utils.data import DataLoader
import os

# Local Imports
import ML.datamodule.dataset_helper as data_help
import ML.utils.plot as plot
import ML.datamodule.data_scalers as data_scalers


class DNN_Datamodule(L.LightningDataModule):
    def __init__(self, cfg_object):
        super().__init__()

        self.path_to_data = cfg_object.dataset.path_to_data
        self.fraction_of_data = cfg_object.dataset.fraction_of_data
        self.train_batch_size = cfg_object.dataset.train.batch_size
        self.val_batch_size = cfg_object.dataset.val.batch_size

        self.num_workers = cfg_object.runtime.num_workers
        logger.info(f"Using {self.num_workers} cpus")

        # Get the inputs and target dictionaries that include their respective scaling
        self.inputs = data_scalers.create_scaler_dict(cfg_object.dataset["inputs"])
        self.target = data_scalers.create_scaler_dict(cfg_object.dataset["targets"])
        self.delta_conc = cfg_object.dataset.target_delta_conc
        self.train_drop_last = cfg_object.train.drop_last

        # === Result output directory === #
        self.result_dir = f"results/{cfg_object.model.name}/"
        os.makedirs(self.result_dir, exist_ok=True)

        # Private variable to ensure set up is not done twice when calling training and test scripts back to back
        self._has_setup = False

    def setup(self, stage):
        if self._has_setup:
            return
        self._has_setup = True

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

        # Create Column Transformers for inputs and get the scaler for the targets
        self.input_scaler = data_scalers.create_column_transformer(
            self.inputs, self.col_index_map
        )
        self.target_scaler = data_scalers.create_column_transformer(
            self.target, self.target_index_map
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
        X_train, X_val, X_test, y_train, y_val, y_test = (
            data_help.timeseries_train_val_test_split(
                X,
                Y,
                train_frac=0.8,
                val_frac=0.1,
                test_frac=0.1,
                steps_per_run=100,
                shuffle_within_train=True,
            )
        )

        plot.plot_data_distributions(
            X_train, self.col_index_map, save_dir=self.result_dir, name="Raw_Inputs"
        )
        plot.plot_data_distributions(
            y_train, self.target_index_map, save_dir=self.result_dir, name="Raw_Targets"
        )

        # Scale all datasets
        X_train, X_val, X_test, y_train, y_val, y_test = data_help.scale_datasets(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            self.input_scaler,
            self.target_scaler,
        )

        self.X_test = X_test
        self.Y_test = y_test

        plot.plot_data_distributions(
            X_train, self.col_index_map, save_dir=self.result_dir, name="Scaled_Inputs"
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
            persistent_workers=True,
            drop_last=self.train_drop_last,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
