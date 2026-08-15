# Configuration reference

Every key a run config may contain, what reads it, and what values are valid.

A run is one YAML file plus optional command-line overrides:

```bash
uv run nucml --config configs/main_config.yaml train.num_epochs=5 runtime.device=cpu
```

Two conventions hold throughout:

- **Paths are resolved relative to the config file itself**, never to the
  working directory (`utils/paths.py:resolve_config_paths`), so a run behaves
  the same whichever directory it is launched from.
- **Unrecognised enumerated values raise.** A typo in a scaler, activation or
  loss name fails at construction rather than silently substituting a default;
  `tests/test_configs.py` catches it earlier still, before a config is
  committed.

Keys marked **NODE** are read only when `runtime.model: NODE`; keys marked
**DNN** only by the DNN path. Everything else is shared.

---

## `dataset`

| Key | Type | Notes |
|---|---|---|
| `path_to_data` | path | Training HDF5 (or CSV). Fits the scalers. |
| `path_to_inference_data` | path | The file evaluated by `runtime.mode: inference`. |
| `preprocessor_path` | path | Fitted scalers from a bundle. **Required** for `inference`. |
| `fraction_of_data` | float in (0, 1] | See the semantics note below. |
| `split` | mapping | Optional `{train, val, test}` fractions; must sum to 1.0. |
| `inputs` | mapping | `column: scaler`, the model's inputs. |
| `targets` | mapping | `column: scaler`, the model's outputs. |
| `target_delta_conc` | bool | Predict the change per step rather than the absolute value. |
| `train.batch_size` | int | Training dataloader batch size. |
| `val.batch_size` | int | Validation and test batch size (no gradients, so it can be larger). |

**`fraction_of_data` is a prefix, not a sample.** It keeps the *first*
`fraction × total` whole runs of the file (`dataset_helper.read_data`), so
`0.1` means the first 10 % of runs, not a random 10 %.

**`split`** partitions whole runs, never mid-run. When the key is absent each
model falls back to its historical default, which is why the two differ:

| | default | strategy |
|---|---|---|
| DNN | 0.8 / 0.1 / 0.1 | sequential by run, training runs shuffled |
| NODE | 0.6 / 0.2 / 0.2 | seeded random permutation of runs |

The partition actually used is written to
`<output_dir>/<model.name>/model-bundle/split_indices.json`. That the two
models split differently is a known limitation of the head-to-head comparison
— see `AUDIT.md` P5.

**Scalers** (case-insensitive): `MinMax`, `Standard`, `Robust`, `MaxAbs`,
`Normalizer`, `Quantile`, `Power`, `none`. Each is fitted **per column** and on
the **training split only**. Fitted state is persisted to `preprocessor.json`,
which is what lets a checkpoint be served without the training dataset.

**`preprocessor_path`** points at a bundle directory or a `preprocessor.json`.
Like every other path key it is resolved relative to the config file.

Inference **requires** it, and there is no fallback. A checkpoint without its
scalers is half a model: re-fitting on whatever data is to hand produces scalers
the checkpoint never saw, and every number computed through them is then wrong
by however much the two fits disagree — silently, since the output still looks
reasonable. Inference used to do exactly that behind a warning; it now exits.

Set during training, it makes the run *load* scalers instead of fitting them.
That is how `regenerate_plots` replays a finished run against its own scalers.

You should rarely need the key by hand: `nucml --bundle` sets it.

**`target_delta_conc: true`** makes the target `c(t+1) − c(t)`. This matters
for isotopes like U238 that change by only a few percent over the full history:
asked for `c(t+1)` directly, a network scores R² ≈ 1 by copying its input.
Absolute concentrations are recovered by cumulative summation at evaluation
time (`evaluation.deltas_to_absolute`).

---

## `model`

| Key | Type | Notes |
|---|---|---|
| `name` | str | Names the run's output directory and its database row. |
| `layers` | list[int] | Hidden widths, e.g. `[128, 128]`. |
| `dropout_probability` | float in [0, 1] | `0.0` disables dropout. |
| `activation` | str | Hidden-layer activation. |
| `output_activation` | str | Use `none` for regression. |
| `residual_connections` | bool | Skip connections, applied only where adjacent widths match. |
| `matrix_ode` | bool | **NODE** — use the constrained depletion matrix. |
| `matrix_zero_entries` | list[[int, int]] | **NODE** — matrix entries forced to zero. |

**Activations**: `relu`, `tanh`, `sigmoid`, `leaky_relu`, `elu`, `gelu`,
`selu`, `softplus`, `none`.

**`residual_connections`** is applied per layer, only where the input and
output widths are equal; if none match, the model logs an error rather than
silently doing nothing.

**`matrix_ode: true`** switches the right-hand side from `dy/dt = net(u, y)` to
`dy/dt = A(u, y) · y`, where `A` is produced by the network and then
constrained (`ODEFuncMatrix.build_matrix`):

- entries listed in `matrix_zero_entries` are forced to zero;
- diagonal entries pass through `−softplus` (a nuclide can only lose itself);
- off-diagonal entries through `+softplus` (a transition only feeds forward).

The network outputs only the surviving entries, so no capacity is spent on ones
that are masked away. Indices are `[row, column]` into the **target** list, in
the order the targets appear in the data — `tests/test_configs.py` checks they
are in range.

---

## `train`

| Key | Type | Notes |
|---|---|---|
| `loss` | str | `mse`, `mae`, `huber`, `smooth_l1`. |
| `learning_rate` | float | AdamW initial learning rate. |
| `weight_decay` | float | AdamW L2 penalty. |
| `lr_scheduler_patience` | int | Stale epochs before `ReduceLROnPlateau` halves the rate. |
| `num_epochs` | int | Maximum epochs. |
| `early_stopping_patience` | int | Optional. Omit to disable early stopping. |
| `grad_clip` | float | Gradient-norm clip; `0.0` disables. |
| `drop_last` | bool | **DNN** — drop an incomplete final training batch. |

### Solver settings (**NODE** only)

| Key | Type | Notes |
|---|---|---|
| `solver` | str | Any `torchdiffeq` method, e.g. `dopri5`, `rk4`. |
| `rtol` / `atol` | float | Adaptive-solver tolerances. |
| `step_size` | float | **Fixed-step solvers only** (`rk4`). Ignored by `dopri5`. |
| `solver_options` | mapping | Optional. Extra `torchdiffeq` options the flat keys cannot express. |
| `use_adjoint` | bool | Optional. Gradients from a backward adjoint solve. |
| `adjoint_method` | str | Solver for the backward pass. |
| `adjoint_rtol` / `adjoint_atol` | float | Backward-pass tolerances. |
| `adjoint_solver_options` | mapping | Optional. Kept separate because a forward-only key (rk4's `step_size`) is invalid for the adjoint. |

**`use_adjoint`** recovers gradients by solving the adjoint system backwards
instead of storing the forward graph: O(1) memory in the number of function
evaluations rather than O(NFE), at roughly twice the wall time per batch. It is
what makes a large batch fit once the learned dynamics turn stiff — see
`configs/NODE_adjoint.yaml`, which documents the measured trade.

**Tolerances are not free.** The number of function evaluations per epoch is
logged as `nfe` and drawn on `training_loss_log.png`; watch it when changing
`rtol`/`atol`. Note also that the states are float32 (eps ≈ 1.2e-7), so an
`atol` at or below that is asking for more than the arithmetic delivers
(`AUDIT.md` P4).

---

## `runtime`

| Key | Type | Notes |
|---|---|---|
| `mode` | str | `train`, `train_from_ckp`, `inference`, `regenerate_plots`. |
| `model` | str | `DNN` or `NODE`. |
| `ckp_path` | path | Checkpoint for `train_from_ckp` / `inference` / `regenerate_plots`. |
| `bundle_path` | path | Set by `--bundle`; where `regenerate_plots` reads `split_indices.json`. |
| `device` | str | `cpu`, `cuda`, `auto`, `gpu`, `mps`. |
| `seed` | int | Seeds the whole run — see below. |
| `num_workers` | int | Dataloader worker processes; `0` loads in the main process. |
| `output_dir` | path | Root for run outputs: `<output_dir>/<model.name>/`. |
| `model_database` | path | SQLite file the experiment logger appends a row to. |
| `plots` | bool | Optional, default `true`. Set `false` to skip the data-distribution figures. |

**Modes.**

- `inference` loads `ckp_path` and evaluates it on
  `dataset.path_to_inference_data`, for both the DNN and the NODE. It requires
  `dataset.preprocessor_path` and never opens `dataset.path_to_data`.
- `regenerate_plots` redraws a *finished* run's figures: it rebuilds that run's
  own train/val/test split over `dataset.path_to_data`, loads the bundle's
  scalers, and runs the evaluation epoch on the test share alone. Nothing is
  trained and no checkpoint is written. Unlike `inference` it does need the
  training dataset, because the split is recorded as indices into it.

  The rebuilt split is checked against the bundle's `split_indices.json` and the
  run **exits on a mismatch** — a different dataset, `fraction_of_data` or seed
  would otherwise yield plausible figures labelled as the published run's.

In practice you do not write either config by hand. `nucml --bundle` sets `mode`,
`ckp_path`, `bundle_path` and `preprocessor_path` from the bundle's contents:

```bash
# evaluate on new data
uv run nucml --bundle results/<model_name>/model-bundle \
             --data   datasets/new_runs.h5 \
             --out    predictions/

# redraw the run's own figures
uv run nucml --bundle results/<model_name>/model-bundle --regenerate-plots \
             --data   datasets/casl_3305_runs_inter.h5 \
             --out    figures/
```

**`seed`.** `main.py` calls `L.seed_everything(seed, workers=True)` before
anything is constructed, covering weight initialisation and dataloader
shuffling. The two run-splitting permutations and the permutation-importance
shuffle take explicit `numpy` generators rather than the global RNG, so a
datamodule built outside `main()` — as the tests and `nucml-package` do —
splits identically.

*Caveat:* this gives run-to-run reproducibility on a fixed machine and library
set. Bitwise equality across different GPUs additionally requires
`torch.use_deterministic_algorithms(True)` and TF32 disabled; `main.py` sets
`torch.set_float32_matmul_precision("high")`, which permits TF32 matmuls.

**`num_workers`.** For the NODE, `0` is often faster: the dataset is already
tensors in RAM, so workers add process handover and a per-process copy for no
gain.

---

## What a run writes

```
<output_dir>/<model.name>/
├── best-<name>-epoch=NN.ckpt      best-validation-loss checkpoint
├── training_loss_log.png          loss curves (plus NFE for the NODE)
├── model-bundle/                  the run's record
│   ├── weights.ckpt
│   ├── preprocessor.json          fitted scalers, plain text, source of truth
│   ├── preprocessor.joblib        convenience copy
│   ├── config.resolved.yaml       the fully-merged config, post-override
│   ├── split_indices.json         which runs were train / val / test
│   └── metadata.json              git SHA, seed, dataset SHA-256, versions
└── <target>/                      per-isotope figures and CSVs, one dir each
```

The bundle is the unit of publication: a checkpoint alone is half a model,
because the other half is the fitted scalers. `docs/training_pipeline.md`
describes what each figure shows.
