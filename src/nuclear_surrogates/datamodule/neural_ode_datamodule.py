# ML/datamodule/node_datamodule.py
from dataclasses import dataclass
from typing import Any

from loguru import logger
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

import nuclear_surrogates.datamodule.dataset_helper as data_help
import nuclear_surrogates.datamodule.data_scalers as data_scalers
import nuclear_surrogates.utils.plot as plot
from nuclear_surrogates.datamodule.preprocessor import Preprocessor
from nuclear_surrogates.utils.paths import result_dir

# The physical span the NODE's normalised time axis is measured against.
# Historically hardcoded; the data's true span is 990 days, and correcting the
# figures that consume this is AUDIT.md §B1 — deliberately not done here, so
# that this change moves no published number.
DEFAULT_TRAINING_T_DAYS = 1000.0


@dataclass
class _Trajectories:
    """One file, read and reshaped into per-run trajectories."""

    input_flat: Any
    target_flat: Any
    input_trajs: Any
    target_trajs: Any
    col_index_map: dict
    target_index_map: dict
    steps_per_run: int
    actual_steps: int
    num_runs: int
    n_input_features: int
    n_target_features: int
    time_array: Any


def _data_span(traj: _Trajectories) -> float:
    """Physical span of one run, in days, as the data actually records it."""
    raw_t = traj.time_array[: traj.actual_steps]
    return float(raw_t[-1] - raw_t[0])


class NODE_Datamodule(L.LightningDataModule):
    """
    DataModule for Neural ODE trajectory training.

    Key difference from the standard datamodule: instead of serving
    (input, target) pairs, this serves FULL TRAJECTORIES.
    Each item in the dataset is one complete run of shape (steps, features).
    A batch is (batch_size, steps, features).

    Uses separate input_scaler and target_scaler, matching the DNN datamodule.
    """

    def __init__(self, cfg_object):
        super().__init__()
        self.path_to_inference_data = getattr(
            cfg_object.dataset, "path_to_inference_data", None
        )
        self.inference_mode = False
        self.path_to_data = cfg_object.dataset.path_to_data
        self.fraction_of_data = cfg_object.dataset.fraction_of_data

        # When set, the fitted scalers are LOADED rather than re-derived from
        # path_to_data. This is what lets a checkpoint be evaluated without the
        # training dataset, and what removes the train/serve skew in inference
        # mode (AUDIT.md §3.3).
        self.preprocessor_path = cfg_object.dataset.get("preprocessor_path", None)
        self.preprocessor = None
        self.make_plots = cfg_object.runtime.get("plots", True)

        self.train_batch_size = cfg_object.dataset.train.batch_size
        self.val_batch_size = cfg_object.dataset.val.batch_size

        self.num_workers = cfg_object.runtime.num_workers

        # Seeded here rather than relying on the global RNG, so a datamodule
        # constructed outside main() (tests, packaging) splits identically.
        self.seed = cfg_object.runtime.get("seed", 42)

        # Separate scaler configs for inputs and targets (same as DNN)
        self.inputs = data_scalers.create_scaler_dict(cfg_object.dataset["inputs"])
        self.target = data_scalers.create_scaler_dict(cfg_object.dataset["targets"])

        self.result_dir = result_dir(cfg_object)

        self._has_setup = False

    def _read_trajectories(self, path, fraction, all_columns):
        """Read one file into (runs, steps, features) arrays.

        The last row of every run is dropped — it is the NaN end-of-run marker
        the depletion writer emits. This logic used to exist twice, once per
        branch of setup() (AUDIT.md Pass 1 §4).
        """
        df, steps_per_run, time_array = data_help.read_data(
            path, fraction, drop_run_label=True
        )
        data_help.print_dataset_stats(df)
        df = data_help.filter_columns(df, all_columns)

        input_arr, col_index_map = data_help.split_df(df, self.inputs.keys())
        target_arr, target_index_map = data_help.split_df(df, self.target.keys())

        num_total = input_arr.shape[0]
        num_runs = num_total // steps_per_run
        assert (
            num_runs * steps_per_run == num_total
        ), f"Data length {num_total} not divisible by steps_per_run {steps_per_run}"

        keep = np.ones(num_total, dtype=bool)
        keep[steps_per_run - 1 :: steps_per_run] = False
        input_arr, target_arr = input_arr[keep], target_arr[keep]

        actual_steps = steps_per_run - 1
        logger.info(f"Dropped {num_runs} NaN rows, {input_arr.shape[0]} samples remain")
        assert not np.isnan(
            input_arr
        ).any(), "NaNs in input data after dropping last rows!"
        assert not np.isnan(
            target_arr
        ).any(), "NaNs in target data after dropping last rows!"

        n_in, n_tgt = input_arr.shape[1], target_arr.shape[1]
        logger.info(f"Total trajectories: {num_runs}, each {actual_steps} steps")
        logger.info(f"Input features: {n_in}, Target features: {n_tgt}")

        return _Trajectories(
            input_flat=input_arr,
            target_flat=target_arr,
            input_trajs=input_arr.reshape(num_runs, actual_steps, n_in),
            target_trajs=target_arr.reshape(num_runs, actual_steps, n_tgt),
            col_index_map=col_index_map,
            target_index_map=target_index_map,
            steps_per_run=steps_per_run,
            actual_steps=actual_steps,
            num_runs=num_runs,
            n_input_features=n_in,
            n_target_features=n_tgt,
            time_array=time_array,
        )

    def _setup_inference(self, all_columns):
        """Evaluate a frozen model on `path_to_inference_data`.

        With `dataset.preprocessor_path` set, the fitted scalers are loaded and
        `path_to_data` is never opened — which is what lets a checkpoint be
        evaluated without shipping the training dataset, and what removes the
        train/serve skew (AUDIT.md §3.3). Without it, the historical behaviour
        is kept: scalers are re-fit on `path_to_data`.
        """
        if self.preprocessor_path:
            logger.info("Inference mode: loading fitted scalers from the bundle")
            self.preprocessor = Preprocessor.load(self.preprocessor_path)
            training_T = self.preprocessor.t_days
        else:
            logger.warning(
                "Inference mode with no dataset.preprocessor_path: re-fitting "
                "scalers on path_to_data. These are NOT the scalers the "
                "checkpoint was trained with — see AUDIT.md §3.3."
            )
            fit_traj = self._read_trajectories(
                self.path_to_data, self.fraction_of_data, all_columns
            )
            self.preprocessor = Preprocessor.fit(
                self.inputs,
                self.target,
                fit_traj.input_flat,
                fit_traj.target_flat,
                fit_traj.col_index_map,
                fit_traj.target_index_map,
                t_days=DEFAULT_TRAINING_T_DAYS,
                t_days_data_span=_data_span(fit_traj),
            )
            training_T = DEFAULT_TRAINING_T_DAYS

        self.input_scaler = self.preprocessor.input_scaler
        self.target_scaler = self.preprocessor.target_scaler

        traj = self._read_trajectories(self.path_to_inference_data, 1.0, all_columns)
        self.steps_per_run = traj.steps_per_run
        self.actual_steps = traj.actual_steps
        self.time_array = traj.time_array
        self.col_index_map = traj.col_index_map
        self.target_index_map = traj.target_index_map
        self.n_input_features = traj.n_input_features
        self.n_target_features = traj.n_target_features

        input_scaled = self.input_scaler.transform(traj.input_flat).reshape(
            traj.num_runs, traj.actual_steps, traj.n_input_features
        )
        target_scaled = self.target_scaler.transform(traj.target_flat).reshape(
            traj.num_runs, traj.actual_steps, traj.n_target_features
        )

        combined = np.concatenate([input_scaled, target_scaled], axis=-1)
        test_trajs = torch.tensor(combined, dtype=torch.float32)
        assert not torch.isnan(test_trajs).any(), "NaNs in scaled inference data!"

        # Dummy train/val (empty but valid shape)
        dummy = torch.zeros(
            0, traj.actual_steps, traj.n_input_features + traj.n_target_features
        )
        self.train_dataset = TensorDataset(dummy)
        self.val_dataset = TensorDataset(dummy)
        self.test_dataset = TensorDataset(test_trajs)
        self.test_trajs = test_trajs

        logger.info(f"Test dataset size: {traj.num_runs} trajectories")

        # Time span from inference data, normalised by the TRAINING time range
        raw_t_inf = traj.time_array[: traj.actual_steps]
        self.t_span = torch.tensor(
            (raw_t_inf - raw_t_inf[0]) / training_T,
            dtype=torch.float32,
        )
        logger.info(
            f"Training time span: {training_T:.4f}, "
            f"Inference time span: {raw_t_inf[-1] - raw_t_inf[0]:.4f}"
        )
        logger.info(f"Normalised inference t_span: [0, {self.t_span[-1]:.6f}]")

    def setup(self, stage=None):
        if self._has_setup:
            return
        self._has_setup = True

        logger.info("Setting up NODE trajectory data module...")

        all_columns = list(self.inputs.keys()) + [
            k for k in self.target.keys() if k not in self.inputs.keys()
        ]

        if self.inference_mode:
            self._setup_inference(all_columns)
        else:
            traj = self._read_trajectories(
                self.path_to_data, self.fraction_of_data, all_columns
            )
            self.steps_per_run = traj.steps_per_run
            self.time_array = traj.time_array
            self.actual_steps = traj.actual_steps
            self.col_index_map = traj.col_index_map
            self.target_index_map = traj.target_index_map
            num_runs = traj.num_runs
            n_input_features = traj.n_input_features
            n_target_features = traj.n_target_features
            input_trajs = traj.input_trajs
            target_trajs = traj.target_trajs

            if self.make_plots:
                plot.plot_data_distributions(
                    traj.input_flat,
                    self.col_index_map,
                    save_dir=self.result_dir,
                    name="Raw_Inputs",
                )
                plot.plot_data_distributions(
                    traj.target_flat,
                    self.target_index_map,
                    save_dir=self.result_dir,
                    name="Raw_Targets",
                )

            # ══════════════════════════════════════════════════════════════
            # TRAINING: split into train/val/test, scalers fitted on train only
            # ══════════════════════════════════════════════════════════════

            # Re-fit scalers on TRAINING split only (overwrite the full-data fit above)
            perm = np.random.default_rng(self.seed).permutation(num_runs)
            input_trajs = input_trajs[perm]
            target_trajs = target_trajs[perm]

            n_train = int(num_runs * 0.6)
            n_val = int(num_runs * 0.2)

            # perm[i] is the original run index now sitting at position i, so the
            # split is recorded in terms of runs as they appear in the source file.
            self.split_info = {
                "strategy": "random_by_run",
                "n_runs": num_runs,
                "steps_per_run": self.actual_steps,
                "train": perm[:n_train].tolist(),
                "val": perm[n_train : n_train + n_val].tolist(),
                "test": perm[n_train + n_val :].tolist(),
            }
            data_help.write_split_indices(self.result_dir, self.split_info, self.seed)

            train_input_raw = input_trajs[:n_train]
            val_input_raw = input_trajs[n_train : n_train + n_val]
            test_input_raw = input_trajs[n_train + n_val :]

            train_target_raw = target_trajs[:n_train]
            val_target_raw = target_trajs[n_train : n_train + n_val]
            test_target_raw = target_trajs[n_train + n_val :]

            logger.info(
                f"Train: {len(train_input_raw)}, Val: {len(val_input_raw)}, Test: {len(test_input_raw)} runs"
            )

            # Fit scalers on the training split only (no data leakage)
            train_input_flat = train_input_raw.reshape(-1, n_input_features)
            train_target_flat = train_target_raw.reshape(-1, n_target_features)

            self.preprocessor = Preprocessor.fit(
                self.inputs,
                self.target,
                train_input_flat,
                train_target_flat,
                self.col_index_map,
                self.target_index_map,
                t_days=DEFAULT_TRAINING_T_DAYS,
                t_days_data_span=_data_span(traj),
            )
            self.input_scaler = self.preprocessor.input_scaler
            self.target_scaler = self.preprocessor.target_scaler

            # Scale all splits
            def scale_split(input_raw, target_raw):
                n = len(input_raw)
                input_scaled = self.input_scaler.transform(
                    input_raw.reshape(-1, n_input_features)
                ).reshape(n, self.actual_steps, n_input_features)
                target_scaled = self.target_scaler.transform(
                    target_raw.reshape(-1, n_target_features)
                ).reshape(n, self.actual_steps, n_target_features)
                return input_scaled, target_scaled

            train_input_scaled, train_target_scaled = scale_split(
                train_input_raw, train_target_raw
            )
            val_input_scaled, val_target_scaled = scale_split(
                val_input_raw, val_target_raw
            )
            test_input_scaled, test_target_scaled = scale_split(
                test_input_raw, test_target_raw
            )

            if self.make_plots:
                plot.plot_data_distributions(
                    train_input_scaled.reshape(-1, n_input_features),
                    self.col_index_map,
                    save_dir=self.result_dir,
                    name="Scaled_Inputs",
                )
                plot.plot_data_distributions(
                    train_target_scaled.reshape(-1, n_target_features),
                    self.target_index_map,
                    save_dir=self.result_dir,
                    name="Scaled_Targets",
                )

            def combine_to_tensor(input_scaled, target_scaled):
                combined = np.concatenate([input_scaled, target_scaled], axis=-1)
                return torch.tensor(combined, dtype=torch.float32)

            train_trajs = combine_to_tensor(train_input_scaled, train_target_scaled)
            val_trajs = combine_to_tensor(val_input_scaled, val_target_scaled)
            test_trajs = combine_to_tensor(test_input_scaled, test_target_scaled)

            for split_name, trajs in [
                ("train", train_trajs),
                ("val", val_trajs),
                ("test", test_trajs),
            ]:
                assert (
                    not torch.isnan(trajs).any()
                ), f"NaNs in {split_name} trajectories: {torch.isnan(trajs).sum()}"

            raw_t = self.time_array[: self.actual_steps]
            self.t_span = torch.tensor(
                (raw_t - raw_t[0]) / (raw_t[-1] - raw_t[0]),
                dtype=torch.float32,
            )

            self.train_dataset = TensorDataset(train_trajs)
            self.val_dataset = TensorDataset(val_trajs)
            self.test_dataset = TensorDataset(test_trajs)
            self.test_trajs = test_trajs

            logger.info(f"Training dataset size: {len(train_trajs)} trajectories")
            logger.info(f"Validation dataset size: {len(val_trajs)} trajectories")
            logger.info(f"Test dataset size: {len(test_trajs)} trajectories")

            self.n_input_features = n_input_features
            self.n_target_features = n_target_features

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
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
