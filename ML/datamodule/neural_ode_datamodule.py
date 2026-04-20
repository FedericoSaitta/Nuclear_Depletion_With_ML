# ML/datamodule/node_datamodule.py
from loguru import logger
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import os

import ML.datamodule.dataset_helper as data_help
import ML.datamodule.data_scalers as data_scalers
import ML.utils.plot as plot


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
    self.path_to_inference_data = getattr(cfg_object.dataset, 'path_to_inference_data', None)
    self.inference_mode = False
    self.path_to_data = cfg_object.dataset.path_to_data
    self.fraction_of_data = cfg_object.dataset.fraction_of_data

    self.train_batch_size = cfg_object.dataset.train.batch_size
    self.val_batch_size = cfg_object.dataset.val.batch_size

    self.num_workers = cfg_object.runtime.num_workers

    # Separate scaler configs for inputs and targets (same as DNN)
    self.inputs = data_scalers.create_scaler_dict(cfg_object.dataset['inputs'])
    self.target = data_scalers.create_scaler_dict(cfg_object.dataset['targets'])

    self.result_dir = f'results/{cfg_object.model.name}/'
    os.makedirs(self.result_dir, exist_ok=True)

    self._has_setup = False

  def setup(self, stage=None):


    if self._has_setup:
        return
    self._has_setup = True

    logger.info("Setting up NODE trajectory data module...")

    # ── Load primary data (training data, or scaler-fitting data in inference mode) ──
    data_df, self.steps_per_run, self.time_array = data_help.read_data(
        self.path_to_data, self.fraction_of_data, drop_run_label=True,
    )
    data_help.print_dataset_stats(data_df)

    all_columns = list(self.inputs.keys()) + [k for k in self.target.keys() if k not in self.inputs.keys()]
    data_df = data_help.filter_columns(data_df, all_columns)

    input_data_arr, self.col_index_map = data_help.split_df(data_df, self.inputs.keys())
    target_data_arr, self.target_index_map = data_help.split_df(data_df, self.target.keys())

    # ── Drop last row of each run ──
    num_total = input_data_arr.shape[0]
    num_runs = num_total // self.steps_per_run
    assert num_runs * self.steps_per_run == num_total, \
        f"Data length {num_total} not divisible by steps_per_run {self.steps_per_run}"

    keep_mask = np.ones(num_total, dtype=bool)
    for r in range(num_runs):
        keep_mask[r * self.steps_per_run + (self.steps_per_run - 1)] = False
    input_data_arr = input_data_arr[keep_mask]
    target_data_arr = target_data_arr[keep_mask]

    self.actual_steps = self.steps_per_run - 1
    logger.info(f"Dropped {num_runs} NaN rows, {input_data_arr.shape[0]} samples remain")
    assert not np.isnan(input_data_arr).any(), "NaNs in input data after dropping last rows!"
    assert not np.isnan(target_data_arr).any(), "NaNs in target data after dropping last rows!"

    n_input_features = input_data_arr.shape[1]
    n_target_features = target_data_arr.shape[1]

    input_trajs = input_data_arr.reshape(num_runs, self.actual_steps, n_input_features)
    target_trajs = target_data_arr.reshape(num_runs, self.actual_steps, n_target_features)

    logger.info(f"Total trajectories: {num_runs}, each {self.actual_steps} steps")
    logger.info(f"Input features: {n_input_features}, Target features: {n_target_features}")

    # ── Fit scalers on primary data (used by both paths) ──
    train_input_flat = input_data_arr
    train_target_flat = target_data_arr

    plot.plot_data_distributions(train_input_flat, self.col_index_map, save_dir=self.result_dir, name='Raw_Inputs')
    plot.plot_data_distributions(train_target_flat, self.target_index_map, save_dir=self.result_dir, name='Raw_Targets')

    self.input_scaler = data_scalers.create_column_transformer(self.inputs, self.col_index_map)
    self.target_scaler = data_scalers.create_column_transformer(self.target, self.target_index_map)
    self.input_scaler.fit(train_input_flat)
    self.target_scaler.fit(train_target_flat)

    if self.inference_mode:
        # ══════════════════════════════════════════════════════════════
        # INFERENCE: scalers fitted on path_to_data, test on path_to_inference_data
        # ══════════════════════════════════════════════════════════════
        logger.info("Inference mode: scalers fitted on training data, loading inference data...")

        inf_df, inf_steps_per_run, inf_time_array = data_help.read_data(
            self.path_to_inference_data, 1.0, drop_run_label=True,
        )
        inf_df = data_help.filter_columns(inf_df, all_columns)
        inf_input_arr, _ = data_help.split_df(inf_df, self.inputs.keys())
        inf_target_arr, _ = data_help.split_df(inf_df, self.target.keys())

        # Drop last row of each run
        inf_total = inf_input_arr.shape[0]
        inf_num_runs = inf_total // inf_steps_per_run
        assert inf_num_runs * inf_steps_per_run == inf_total, \
            f"Inference data length {inf_total} not divisible by steps_per_run {inf_steps_per_run}"

        inf_keep = np.ones(inf_total, dtype=bool)
        for r in range(inf_num_runs):
            inf_keep[r * inf_steps_per_run + (inf_steps_per_run - 1)] = False
        inf_input_arr = inf_input_arr[inf_keep]
        inf_target_arr = inf_target_arr[inf_keep]

        inf_actual_steps = inf_steps_per_run - 1
        self.actual_steps = inf_actual_steps
        self.time_array = inf_time_array

        assert not np.isnan(inf_input_arr).any(), "NaNs in inference input data!"
        assert not np.isnan(inf_target_arr).any(), "NaNs in inference target data!"

        inf_input_trajs = inf_input_arr.reshape(inf_num_runs, inf_actual_steps, n_input_features)
        inf_target_trajs = inf_target_arr.reshape(inf_num_runs, inf_actual_steps, n_target_features)

        logger.info(f"Inference data: {inf_num_runs} trajectories, {inf_actual_steps} steps")

        # Scale inference data using training-fitted scalers
        inf_input_scaled = self.input_scaler.transform(
            inf_input_arr
        ).reshape(inf_num_runs, inf_actual_steps, n_input_features)

        inf_target_scaled = self.target_scaler.transform(
            inf_target_arr
        ).reshape(inf_num_runs, inf_actual_steps, n_target_features)

        combined = np.concatenate([inf_input_scaled, inf_target_scaled], axis=-1)
        test_trajs = torch.tensor(combined, dtype=torch.float32)

        assert not torch.isnan(test_trajs).any(), "NaNs in scaled inference data!"

        # Dummy train/val (empty but valid shape)
        dummy = torch.zeros(0, inf_actual_steps, n_input_features + n_target_features)
        self.train_dataset = TensorDataset(dummy)
        self.val_dataset = TensorDataset(dummy)
        self.test_dataset = TensorDataset(test_trajs)
        self.test_trajs = test_trajs

        logger.info(f"Test dataset size: {inf_num_runs} trajectories")

        # Time span from inference data, scaled using TRAINING time range
        raw_t_inf = inf_time_array[:inf_actual_steps]
        training_T = 1000  # physical training data spans 1000 days
        self.t_span = torch.tensor(
            (raw_t_inf - raw_t_inf[0]) / training_T,
            dtype=torch.float32,
        )
        logger.info(f"Training time span: {training_T:.4f}, Inference time span: {raw_t_inf[-1] - raw_t_inf[0]:.4f}")
        logger.info(f"Normalised inference t_span: [0, {self.t_span[-1]:.6f}]")

    else:
        # ══════════════════════════════════════════════════════════════
        # TRAINING: split into train/val/test, scalers fitted on train only
        # ══════════════════════════════════════════════════════════════

        # Re-fit scalers on TRAINING split only (overwrite the full-data fit above)
        perm = np.random.permutation(num_runs)
        input_trajs = input_trajs[perm]
        target_trajs = target_trajs[perm]

        n_train = int(num_runs * 0.6)
        n_val = int(num_runs * 0.2)

        train_input_raw = input_trajs[:n_train]
        val_input_raw = input_trajs[n_train:n_train + n_val]
        test_input_raw = input_trajs[n_train + n_val:]

        train_target_raw = target_trajs[:n_train]
        val_target_raw = target_trajs[n_train:n_train + n_val]
        test_target_raw = target_trajs[n_train + n_val:]

        logger.info(f"Train: {len(train_input_raw)}, Val: {len(val_input_raw)}, Test: {len(test_input_raw)} runs")

        # Re-fit scalers on training split only (no data leakage)
        train_input_flat = train_input_raw.reshape(-1, n_input_features)
        train_target_flat = train_target_raw.reshape(-1, n_target_features)

        self.input_scaler = data_scalers.create_column_transformer(self.inputs, self.col_index_map)
        self.target_scaler = data_scalers.create_column_transformer(self.target, self.target_index_map)
        self.input_scaler.fit(train_input_flat)
        self.target_scaler.fit(train_target_flat)

        # Scale all splits
        def scale_split(input_raw, target_raw):
            n = len(input_raw)
            input_scaled = self.input_scaler.transform(input_raw.reshape(-1, n_input_features)).reshape(n, self.actual_steps, n_input_features)
            target_scaled = self.target_scaler.transform(target_raw.reshape(-1, n_target_features)).reshape(n, self.actual_steps, n_target_features)
            return input_scaled, target_scaled

        train_input_scaled, train_target_scaled = scale_split(train_input_raw, train_target_raw)
        val_input_scaled, val_target_scaled = scale_split(val_input_raw, val_target_raw)
        test_input_scaled, test_target_scaled = scale_split(test_input_raw, test_target_raw)

        plot.plot_data_distributions(train_input_scaled.reshape(-1, n_input_features), self.col_index_map, save_dir=self.result_dir, name='Scaled_Inputs')
        plot.plot_data_distributions(train_target_scaled.reshape(-1, n_target_features), self.target_index_map, save_dir=self.result_dir, name='Scaled_Targets')

        def combine_to_tensor(input_scaled, target_scaled):
            combined = np.concatenate([input_scaled, target_scaled], axis=-1)
            return torch.tensor(combined, dtype=torch.float32)

        train_trajs = combine_to_tensor(train_input_scaled, train_target_scaled)
        val_trajs = combine_to_tensor(val_input_scaled, val_target_scaled)
        test_trajs = combine_to_tensor(test_input_scaled, test_target_scaled)

        for split_name, trajs in [('train', train_trajs), ('val', val_trajs), ('test', test_trajs)]:
            assert not torch.isnan(trajs).any(), \
                f"NaNs in {split_name} trajectories: {torch.isnan(trajs).sum()}"

        raw_t = self.time_array[:self.actual_steps]
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

    # ── Shared: feature counts ──
    self.n_input_features = n_input_features
    self.n_target_features = n_target_features

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset, batch_size=self.train_batch_size, shuffle=True,
        num_workers=self.num_workers, persistent_workers=self.num_workers > 0, pin_memory=True,
    )

  def val_dataloader(self):
    return DataLoader(
        self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
        num_workers=self.num_workers, persistent_workers=self.num_workers > 0, pin_memory=True,
    )

  def test_dataloader(self):
    return DataLoader(
        self.test_dataset, batch_size=self.val_batch_size, shuffle=False,
        num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
    )

  def predict_dataloader(self):
    return self.test_dataloader()