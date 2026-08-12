## User Guide on Pytorch Guide:

This library was created by keeping in mind usability and future use so its use is tailored for machine learning analysis of nuclear fuel as it depletes over time but tweaks and different models can be added simply by adding further pytorch modules. \
This library comes with a DNN module which is the module used to produce the results in our project reports.\
The training and testing of the model is controlled by a .yaml file which the user can edit, the current parameters and their values are as follow under each of the .yaml keywords.

# Configuration Guide

## Dataset Configuration

### `dataset`
- **`path_to_data`**: `string`
  - Path to the training HDF5 file. Relative values are resolved against the
    **config file's own directory**, never the working directory.
  
- **`fraction_of_data`**: `float` (0.0 to 1.0)
  - Fraction of the dataset to use for training/validation
  - `1.0` = use entire dataset, `0.5` = use 50% of data

- **`inputs`**: `dictionary`
  - Contains the columns to include as the inputs to the model, and their specific scaling
  - The possible scalers are: MinMax, Standard, Robust, MaxAbs, Normalizer,
    Quantile, Power, and None if no scaling should be applied. Any other value
    raises, rather than silently leaving the column unscaled.
  ```yaml
    power_W_g: "MinMax"
    U238: "robust"
    ```

- **`targets`**: `dictionary`
  - Similalry to the inputs these columns will be included as the targets and outputs of the model. Notably the inputs are at time `t` while the targets are at time `t+1`. Hence you can have a column be both a target and an input, common in many time series predictions tasks.

- **`target_delta_conc`**: `boolean`
  - **Options**:
    - `True`: model predicts concentration difference for each step
    - `False`: model predicts absolute values

### Data Loaders

#### `train`
- **`batch_size`**: `integer`
  - Number of samples per training batch
  - Larger batches = more stable gradients, higher memory usage but worse generalization

#### `val`
- **`batch_size`**: `integer`
  - Number of samples per validation batch
  - Should be as large as possible on the available hardware as gradients are not computed

---

## Model Configuration

### `model`
- **`name`**: `string`
  - Model identifier used for saving results and logging

- **`layers`**: `list[integer]`
  - Architecture of hidden layers
  - Example: `[64, 64]` = two hidden layers with 64 neurons each
  - Example: `[128, 64, 32]` = three hidden layers with decreasing sizes

- **`dropout_probability`**: `float` (0.0 to 1.0)
  - Dropout rate for regularization
  - `0.0` = no dropout, `0.1` = 10% of neurons dropped

- **`activation`**: `string`
  - Activation function for hidden layers
  - **Options**: `"relu"`, `"tanh"`, `"sigmoid"`, `"leaky_relu"`, `"elu"`,
    `"gelu"`, `"selu"`, `"softplus"`, `"none"`
  - Any other value raises, rather than silently substituting a default

- **`output_activation`**: `string`
  - Activation function for output layer, from the same set as `activation`
  - Use `"none"` for regression tasks

- **`residual_connections`**: `boolean`
  - **Options**:
    - `True`: Enable skip connections between layers (only works if two adjacent layers match in the number of inputs and outputs)
    - `False`: Standard feedforward connections

---

## Training Configuration

### `train`
- **`loss`**: `string`
  - Loss function for optimization
  - **Options**: `"mse"` (Mean Squared Error), `"mae"` (Mean Absolute Error), `"huber"`, `"smooth_l1"`

- **`learning_rate`**: `float`
  - Initial learning rate for optimizer
  - Typical range: `1e-5` to `1e-2`

- **`weight_decay`**: `float`
  - L2 regularization penalty
  - `0.0` = no regularization, typical values: `1e-5` to `1e-3`

- **`num_epochs`**: `integer`
  - Maximum number of training epochs

- **`lr_scheduler_patience`**: `integer`
  - Number of epochs without improvement before reducing learning rate
  - Used with ReduceLROnPlateau scheduler

- **`early_stopping_patience`**: `integer`
  - Number of epochs without improvement before stopping training
  - Prevents overfitting

- **`dropout_probability`**: `float` (0.0 to 1.0)
  - Dropout rate during training (can override model dropout)

---

## Runtime Configuration

### `runtime`
- **`mode`**: `string`
  - Execution mode
  - **Options**:
    - `"train"`: Train from scratch
    - `"train_from_ckp"`: Resume training from checkpoint
    - `"inference"`: Load a checkpoint and test on `dataset.path_to_inference_data`

- **`ckp_path`**: `string`
  - Path to the checkpoint to load. Resolved against the config file's directory.
  - Used by `mode = "train_from_ckp"` and `mode = "inference"`

- **`output_dir`**: `string`
  - Root for run outputs; each run writes to `<output_dir>/<model.name>/`.
  - Resolved against the config file's directory, so the output location does not
    depend on where the command was launched from.

- **`model_database`**: `string`
  - SQLite file the experiment logger appends a row to. Resolved like the paths above.

- **`device`**: `string`
  - Computation device
  - **Options**: `"cuda"` (GPU), `"cpu"`

- **`seed`**: `integer`
  - Seeds the whole run. `main.py` calls `L.seed_everything(seed, workers=True)` before
    anything is constructed, which covers weight initialisation and the DataLoader
    shuffle order.
  - The two run-splitting permutations take an explicit
    `np.random.default_rng(seed)` rather than the global RNG
    (`neural_ode_datamodule.py`, `dataset_helper.timeseries_train_val_test_split`), so a
    datamodule built outside `main()` — as the tests and `nucml-package` do — splits
    identically.
  - Every run writes its partition to
    `<output_dir>/<model.name>/split_indices.json`, so which runs were held out is
    recoverable after the fact.
  - Permutation feature importance (`metrics.calculate_feature_importance`) takes a
    `seed` argument, default `0`.
  - **Caveat:** this gives run-to-run reproducibility on a fixed machine and library
    set. Bitwise equality across GPUs additionally needs
    `torch.use_deterministic_algorithms(True)` and TF32 disabled — `main.py` sets
    `torch.set_float32_matmul_precision("high")`, which permits TF32 matmuls.

- **`drop_last`**: `boolean`
  - **Options**:
    - `True`: Drop incomplete final batch
    - `False`: Keep all data including incomplete batch

- **`num_workers`**: `integer`
  - Number of parallel workers for data loading
  - `0` = load in main process, `>0` = use multiprocessing
---