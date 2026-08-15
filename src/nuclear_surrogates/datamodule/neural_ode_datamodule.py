from dataclasses import dataclass
from typing import Any

import lightning as L
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

import nuclear_surrogates.datamodule.data_scalers as data_scalers
import nuclear_surrogates.datamodule.dataset_helper as data_help
import nuclear_surrogates.utils.plot as plot
from nuclear_surrogates.datamodule.preprocessor import (
    Preprocessor,
    require_fitted_scalers,
)
from nuclear_surrogates.utils.paths import result_dir

# The nominal training span recorded in a bundle's `t_days`. Historically
# hardcoded at 1000 while the data actually spans 990 days; it is kept at 1000
# so existing bundles and the depletion-matrix figure's unit conversion are
# unchanged (AUDIT.md P1). It is NOT the number the time axis is normalised by
# — see `_training_time_unit`.
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


def _training_time_unit(preprocessor) -> float:
    """The physical span that training's normalised time t=1 corresponds to.

    Training normalises its own grid to exactly [0, 1]
    (`(raw_t - raw_t[0]) / (raw_t[-1] - raw_t[0])`), so the unit is the span of
    the data it was fitted on — `t_days_data_span`, 990 days for the CASL runs.

    Inference used to divide by `t_days` instead, which is the nominal 1000, so
    the same 990-day grid came out as [0, 0.99]: every step was integrated over
    a 1 % shorter interval than the model was trained on, and a frozen
    checkpoint did not reproduce its own training-mode trajectories. That was
    AUDIT.md P14. Bundles written before `t_days_data_span` existed fall back to
    `t_days` and keep their old behaviour.
    """
    span = getattr(preprocessor, "t_days_data_span", None)
    if span is None:
        logger.warning(
            "preprocessor has no t_days_data_span (bundle predates it) — "
            "normalising inference time by the nominal t_days instead"
        )
        return float(preprocessor.t_days)
    return float(span)


def _to_trajectory_tensor(input_scaled, target_scaled):
    """Concatenate forcing and target features into one (runs, steps, features) tensor."""
    combined = np.concatenate([input_scaled, target_scaled], axis=-1)
    return torch.tensor(combined, dtype=torch.float32)


class NODE_Datamodule(L.LightningDataModule):
    """Serves whole trajectories rather than (input, target) pairs.

    Each item is one complete run of shape (steps, features); a batch is
    (batch_size, steps, features). Inputs and targets get separate scalers, as
    in the DNN datamodule.
    """

    def __init__(self, cfg_object):
        super().__init__()
        self.path_to_data = cfg_object.dataset.path_to_data
        self.path_to_inference_data = getattr(
            cfg_object.dataset, "path_to_inference_data", None
        )
        self.inference_mode = False
        self.fraction_of_data = cfg_object.dataset.fraction_of_data

        # When set, the fitted scalers are LOADED rather than re-derived from
        # path_to_data. This is what lets a checkpoint be evaluated without the
        # training dataset, and what removes the train/serve skew.
        self.preprocessor_path = cfg_object.dataset.get("preprocessor_path", None)
        self.preprocessor = None
        self.make_plots = cfg_object.runtime.get("plots", True)

        self.train_batch_size = cfg_object.dataset.train.batch_size
        self.val_batch_size = cfg_object.dataset.val.batch_size
        self.num_workers = cfg_object.runtime.num_workers

        # Seeded here rather than relying on the global RNG, so a datamodule
        # constructed outside main() (tests, packaging) splits identically.
        self.seed = cfg_object.runtime.get("seed", 42)

        # Random-by-run 60/20/20 unless the config says otherwise.
        self.split = data_help.split_fractions(cfg_object, default=(0.6, 0.2, 0.2))

        self.inputs = data_scalers.create_scaler_dict(cfg_object.dataset["inputs"])
        self.target = data_scalers.create_scaler_dict(cfg_object.dataset["targets"])

        self.result_dir = result_dir(cfg_object)
        self._has_setup = False

    def setup(self, stage=None):
        if self._has_setup:
            return
        self._has_setup = True

        logger.info("Setting up NODE trajectory data module...")

        all_columns = list(self.inputs.keys()) + [
            k for k in self.target if k not in self.inputs
        ]

        if self.inference_mode:
            self._setup_inference(all_columns)
        else:
            self._setup_training(all_columns)

    # ── Reading ──────────────────────────────────────────────────────────────

    def _read_trajectories(self, path, fraction, all_columns):
        """Read one file into (runs, steps, features) arrays.

        The last row of every run is dropped — it is the NaN end-of-run marker
        the depletion writer emits.
        """
        df, steps_per_run, time_array = data_help.read_data(
            path, fraction, drop_run_label=True, columns=all_columns
        )
        data_help.print_dataset_stats(df)
        df = df.select(all_columns)

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
        assert not np.isnan(input_arr).any(), "NaNs in input data after dropping!"
        assert not np.isnan(target_arr).any(), "NaNs in target data after dropping!"

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

    def _adopt(self, traj: _Trajectories):
        """Copy a file's shape and column layout onto the datamodule."""
        self.steps_per_run = traj.steps_per_run
        self.actual_steps = traj.actual_steps
        self.time_array = traj.time_array
        self.col_index_map = traj.col_index_map
        self.target_index_map = traj.target_index_map
        self.n_input_features = traj.n_input_features
        self.n_target_features = traj.n_target_features

    def _scale(self, traj: _Trajectories, input_raw, target_raw):
        """Apply the fitted scalers, preserving the (runs, steps, features) shape."""
        n = len(input_raw)
        input_scaled = self.input_scaler.transform(
            input_raw.reshape(-1, traj.n_input_features)
        ).reshape(n, traj.actual_steps, traj.n_input_features)
        target_scaled = self.target_scaler.transform(
            target_raw.reshape(-1, traj.n_target_features)
        ).reshape(n, traj.actual_steps, traj.n_target_features)
        return input_scaled, target_scaled

    # ── Training ─────────────────────────────────────────────────────────────

    def _setup_training(self, all_columns):
        """Split by run, fit the scalers on the training split, build the datasets."""
        traj = self._read_trajectories(
            self.path_to_data, self.fraction_of_data, all_columns
        )
        self._adopt(traj)

        if self.make_plots:
            self._plot_distributions(traj.input_flat, traj.target_flat, "Raw")

        num_runs = traj.num_runs
        perm = np.random.default_rng(self.seed).permutation(num_runs)
        input_trajs = traj.input_trajs[perm]
        target_trajs = traj.target_trajs[perm]

        train_frac, val_frac, _ = self.split
        n_train = int(num_runs * train_frac)
        n_val = int(num_runs * val_frac)

        # perm[i] is the original run index now sitting at position i, so the
        # split is recorded in terms of runs as they appear in the source file.
        self.split_info = {
            "strategy": "random_by_run",
            "fractions": list(self.split),
            "n_runs": num_runs,
            "steps_per_run": self.actual_steps,
            "train": perm[:n_train].tolist(),
            "val": perm[n_train : n_train + n_val].tolist(),
            "test": perm[n_train + n_val :].tolist(),
        }

        splits = {
            "train": (input_trajs[:n_train], target_trajs[:n_train]),
            "val": (
                input_trajs[n_train : n_train + n_val],
                target_trajs[n_train : n_train + n_val],
            ),
            "test": (input_trajs[n_train + n_val :], target_trajs[n_train + n_val :]),
        }
        logger.info(
            f"Train: {n_train}, Val: {n_val}, Test: {num_runs - n_train - n_val} runs"
        )

        # Fitted on the training split only — no leakage from val/test. The
        # load branch is `regenerate_plots` replaying a finished run against
        # the scalers that run was trained with, rather than a fresh fit that
        # would only coincidentally agree with them.
        train_input_raw, train_target_raw = splits["train"]
        if self.preprocessor_path:
            logger.info("Loading fitted scalers from the bundle rather than fitting")
            self.preprocessor = Preprocessor.load(self.preprocessor_path)
        else:
            self.preprocessor = Preprocessor.fit(
                self.inputs,
                self.target,
                train_input_raw.reshape(-1, traj.n_input_features),
                train_target_raw.reshape(-1, traj.n_target_features),
                self.col_index_map,
                self.target_index_map,
                t_days=DEFAULT_TRAINING_T_DAYS,
                t_days_data_span=_data_span(traj),
            )
        self.input_scaler = self.preprocessor.input_scaler
        self.target_scaler = self.preprocessor.target_scaler

        scaled = {
            name: self._scale(traj, *raw_pair) for name, raw_pair in splits.items()
        }

        if self.make_plots:
            train_input_scaled, train_target_scaled = scaled["train"]
            self._plot_distributions(
                train_input_scaled.reshape(-1, traj.n_input_features),
                train_target_scaled.reshape(-1, traj.n_target_features),
                "Scaled",
            )

        datasets = {}
        for name, (input_scaled, target_scaled) in scaled.items():
            trajs = _to_trajectory_tensor(input_scaled, target_scaled)
            assert not torch.isnan(trajs).any(), f"NaNs in {name} trajectories"
            datasets[name] = TensorDataset(trajs)
            logger.info(f"{name.capitalize()} dataset size: {len(trajs)} trajectories")

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["val"]
        self.test_dataset = datasets["test"]
        self.test_trajs = self.test_dataset.tensors[0]

        raw_t = self.time_array[: self.actual_steps]
        self.t_span = torch.tensor(
            (raw_t - raw_t[0]) / (raw_t[-1] - raw_t[0]), dtype=torch.float32
        )

    # ── Inference ────────────────────────────────────────────────────────────

    def _setup_inference(self, all_columns):
        """Evaluate a frozen model on `path_to_inference_data`.

        The fitted scalers come from `dataset.preprocessor_path` and nowhere
        else, so `path_to_data` is never opened — which is what lets a
        checkpoint be evaluated without shipping the training dataset.
        """
        require_fitted_scalers(self.preprocessor_path)
        logger.info("Inference mode: loading fitted scalers from the bundle")
        self.preprocessor = Preprocessor.load(self.preprocessor_path)
        training_T = _training_time_unit(self.preprocessor)

        self.input_scaler = self.preprocessor.input_scaler
        self.target_scaler = self.preprocessor.target_scaler

        traj = self._read_trajectories(self.path_to_inference_data, 1.0, all_columns)
        self._adopt(traj)

        test_trajs = _to_trajectory_tensor(
            *self._scale(traj, traj.input_trajs, traj.target_trajs)
        )
        assert not torch.isnan(test_trajs).any(), "NaNs in scaled inference data!"

        # Empty but correctly shaped, so Lightning can still build the loaders.
        dummy = torch.zeros(
            0, traj.actual_steps, traj.n_input_features + traj.n_target_features
        )
        self.train_dataset = TensorDataset(dummy)
        self.val_dataset = TensorDataset(dummy)
        self.test_dataset = TensorDataset(test_trajs)
        self.test_trajs = test_trajs

        logger.info(f"Test dataset size: {traj.num_runs} trajectories")

        # Inference time is normalised by the TRAINING span, not its own, so the
        # model sees the same time units it was fitted in.
        raw_t_inf = traj.time_array[: traj.actual_steps]
        self.t_span = torch.tensor(
            (raw_t_inf - raw_t_inf[0]) / training_T, dtype=torch.float32
        )
        logger.info(
            f"Training time span: {training_T:.4f}, "
            f"Inference time span: {raw_t_inf[-1] - raw_t_inf[0]:.4f}"
        )
        logger.info(f"Normalised inference t_span: [0, {self.t_span[-1]:.6f}]")

    def _plot_distributions(self, input_flat, target_flat, prefix):
        plot.plot_data_distributions(
            input_flat,
            self.col_index_map,
            save_dir=self.result_dir,
            name=f"{prefix}_Inputs",
        )
        plot.plot_data_distributions(
            target_flat,
            self.target_index_map,
            save_dir=self.result_dir,
            name=f"{prefix}_Targets",
        )

    # ── Dataloaders ──────────────────────────────────────────────────────────

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
