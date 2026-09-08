# Configuration reference

Every key a run config may contain, what reads it, and what values are valid —
followed by the [command-line flags](#command-line) that supply everything a
config deliberately does not.

A run is one YAML file, one subcommand, and optional overrides:

```bash
uv run nucml train --config configs/node.yaml --data datasets/casl_3305_runs_inter.h5 \
                   --device cpu train.num_epochs=5
```

Two conventions hold throughout:

- **A config describes the model, not the run.** There is no path, device,
  seed or mode key in any config under `configs/`; those are flags, so the same
  file runs unedited on a laptop and on the cluster. `tests/test_configs.py`
  enforces the absence.
- **Unrecognised enumerated values raise.** A typo in a scaler, activation or
  loss name fails at construction rather than silently substituting a default;
  `tests/test_configs.py` catches it earlier still, before a config is
  committed.

Keys marked **NODE** are read only when `model.kind: NODE`; keys marked
**DNN** only by the DNN path. Everything else is shared.

---

## `dataset`

| Key | Type | Notes |
|---|---|---|
| `fraction_of_data` | float in (0, 1] | See the semantics note below. |
| `split` | mapping | Optional `{train, val, test}` fractions; must sum to 1.0. |
| `inputs` | mapping | `column: scaler`, the model's inputs. |
| `targets` | mapping | `column: scaler`, the model's outputs. |
| `target_delta_conc` | bool | Predict the change per step rather than the absolute value. |
| `train.batch_size` | int | Training dataloader batch size. |
| `val.batch_size` | int | Validation and test batch size (no gradients, so it can be larger). |

Three further `dataset` keys exist but are **written by the CLI, never by you**:
`path_to_data` (from `--data` on `train` / `finetune` / `plots`),
`path_to_inference_data` (from `--data` on `infer`) and `preprocessor_path`
(from `--bundle`). They appear in a bundle's `config.resolved.yaml` because that
file records what actually ran; a config in `configs/` that declared one would
be describing a machine rather than a model, and the test suite rejects it.

**`fraction_of_data` is a prefix, not a sample.** It keeps the *first*
`fraction × total` whole runs of the file (`dataset_helper.read_data`), so
`0.1` means the first 10 % of runs, not a random 10 %.

**`split`** partitions whole runs, never mid-run. It carries the three
fractions plus `strategy`, and both models support both strategies:

| `strategy` | partition |
|---|---|
| `sequential_by_run` | contiguous slices in file order, so the test set is the tail of the file; the training runs are then shuffled |
| `random_by_run` | a seeded permutation of whole runs (`dataset_helper.run_permutation`), partitioned at the same boundaries by both models |

Under `random_by_run`, **one seed and one run count put both models on the same
test runs**, which is what makes a DNN-vs-NODE comparison paired rather than
merely matched. All three shipped model configs set it, and
`tests/test_configs.py` asserts they agree.

When a key is absent each model falls back to its historical default —
`sequential_by_run` at 0.8 / 0.1 / 0.1 for the DNN, `random_by_run` at
0.6 / 0.2 / 0.2 for the NODE — so a config written before these keys existed,
including the `config.resolved.yaml` archived inside an older bundle, still
rebuilds the partition it was trained with. An unrecognised strategy raises
rather than falling back.

`random_by_run` needs a seed; it comes from `--seed`, and the partition actually
used is written to `<--out>/<model.name>/model-bundle/split_indices.json`.

**Scalers** (case-insensitive): `MinMax`, `Standard`, `Robust`, `MaxAbs`,
`Normalizer`, `Quantile`, `Power`, `none`. Each is fitted **per column** and on
the **training split only**. Fitted state is persisted to `preprocessor.json`,
which is what lets a checkpoint be served without the training dataset.

**`preprocessor_path`** points at the `preprocessor.json` inside a bundle, and
`--bundle` is what sets it. `infer` **requires** it, and there is no fallback. A
checkpoint without its scalers is half a model: re-fitting on whatever data is
to hand produces scalers the checkpoint never saw, and every number computed
through them is then wrong by however much the two fits disagree — silently,
since the output still looks reasonable. Inference used to do exactly that
behind a warning; it now exits.

Present during a training-shaped run, it makes the run *load* scalers instead of
fitting them. That is how `plots` replays a finished run against its own
scalers.

**`target_delta_conc: true`** makes the target `c(t+1) − c(t)`. This matters
for isotopes like U238 that change by only a few percent over the full history:
asked for `c(t+1)` directly, a network scores R² ≈ 1 by copying its input.

Absolute concentrations are recovered at evaluation time, and *how* depends on
the rollout: `evaluation.integrate_deltas` sums the predictions for the
free-running trajectory, while `evaluation.teacher_forced_from_deltas` adds each
prediction to the true concentration for the one-step one. Both start at the
true c(0), so they land on the grid the NODE reports on — see
`docs/training_pipeline.md` §5. Every target must therefore also be an input, so
that its initial concentration is known; a config that breaks this is refused.

---

## `model`

| Key | Type | Notes |
|---|---|---|
| `kind` | str | `DNN` or `NODE`. Chooses the model class *and* its datamodule. |
| `name` | str | Names the run's output directory and its database row. |
| `layers` | list[int] | Hidden widths, e.g. `[128, 128]`. |
| `dropout_probability` | float in [0, 1] | `0.0` disables dropout. |
| `activation` | str | Hidden-layer activation. |
| `output_activation` | str | Optional. Use `none` for regression; omit under `matrix_ode`. |
| `residual_connections` | bool | Skip connections, applied only where adjacent widths match. |
| `matrix_ode` | bool | **NODE** — use the constrained depletion matrix. |
| `matrix_zero_entries` | list[[int, int]] | **NODE** — matrix entries forced to zero. |

**`kind`** is the one key with no default: `main.py` looks it up directly, so a
config that omits it fails immediately rather than training the wrong model.

**Activations**: `relu`, `tanh`, `sigmoid`, `leaky_relu`, `elu`, `gelu`,
`selu`, `softplus`, `none`.

**`output_activation` under `matrix_ode`.** `ODEFuncMatrix` hardcodes `"none"`
and applies its own sign constraints instead, so a value set here is silently
discarded. `configs/node.yaml` therefore omits the key, and
`tests/test_configs.py` checks that every matrix-ODE config does — a key that
reads as a knob and is not one is worse than an absent key.

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
| `use_adjoint` | bool | Optional. Gradients from a backward adjoint solve. |
| `adjoint_method` | str | Solver for the backward pass. |
| `adjoint_rtol` / `adjoint_atol` | float | Backward-pass tolerances. |

**`use_adjoint`** recovers gradients by solving the adjoint system backwards
instead of storing the forward graph: O(1) memory in the number of function
evaluations rather than O(NFE), at roughly twice the wall time per batch. It is
what makes a large batch fit once the learned dynamics turn stiff. Neither
shipped config enables it; turn it on with an override
(`train.use_adjoint=true`) when a batch no longer fits.

**Tolerances are not free.** The number of function evaluations per epoch is
logged as `nfe` and drawn on `training_loss_log.png`; watch it when changing
`rtol`/`atol`. Note also that the states are float32 (eps ≈ 1.2e-7), so an
`atol` at or below that is asking for more than the arithmetic delivers
(`docs/training_pipeline.md` §8).

---

## Command line

What the config no longer holds. Four subcommands, each taking one source —
a config to build a new model, or a bundle to reuse a trained one:

| Command | Source | Does |
|---|---|---|
| `nucml train` | `--config CFG` | Fits scalers on the training split, trains, tests, writes a bundle. |
| `nucml finetune` | `--bundle DIR` | Loads the bundle's **weights only**, then trains as above into a new bundle. |
| `nucml infer` | `--bundle DIR` | Evaluates the frozen model on `--data`. Never opens the training file. |
| `nucml plots` | `--bundle DIR` | Redraws a finished run's figures. Nothing trained, no checkpoint written. |

Every verb takes the same machine flags, all optional except `--data`:

| Flag | Default | Sets |
|---|---|---|
| `--data FILE` | **required** | `dataset.path_to_data`, or `path_to_inference_data` under `infer`. |
| `--out DIR` | `results` | Root for run outputs: `<--out>/<model.name>/`. |
| `--device D` | `auto` | `auto`, `cpu`, `cuda`, `gpu`, `mps`. |
| `--workers N` | `0` | Dataloader worker processes; `0` loads in the main process. |
| `--seed N` | `42`, or the bundle's | Seeds the whole run — see below. |
| `--no-analyses` | off | Skip every figure and post-hoc analysis; metrics are still written. |
| `key=value …` | — | Trailing OmegaConf overrides, applied last. |

These become the `runtime` block of the resolved config, which is what the
bundle records. Nothing reads a `runtime` section out of a hand-written YAML;
`read_bundle` strips the recorded one back off, since the machine reading a
bundle is rarely the one that wrote it.

**`finetune` is a warm start, not a resume.** Only the weights are restored —
the optimizer, the LR schedule and the epoch counter all start fresh. That is
the useful behaviour for adapting a trained model to new data or restarting
with a different learning rate, but it means the loss curve will not simply
continue where the previous run's left off.

**`plots` needs the training dataset**, unlike `infer`, because the test split
is recorded as indices into it. The rebuilt split is checked against the
bundle's `split_indices.json` and the run **exits on a mismatch** — a different
dataset, `fraction_of_data` or seed would otherwise yield plausible figures
labelled as the published run's.

**`--seed`.** `main.py` calls `L.seed_everything(seed, workers=True)` before
anything is constructed, covering weight initialisation and dataloader
shuffling. The two run-splitting permutations and the permutation-importance
shuffle take explicit `numpy` generators rather than the global RNG, so a
datamodule built outside `main()` — as the tests do — splits identically.

For the three bundle verbs the default is not `42` but *the seed the bundle
recorded*, so `plots` reproduces the published split without being told. Passing
`--seed` there overrides it, which is precisely what makes the split differ.

*Caveat:* this gives run-to-run reproducibility on a fixed machine and library
set. Bitwise equality across different GPUs additionally requires
`torch.use_deterministic_algorithms(True)` and TF32 disabled; `main.py` sets
`torch.set_float32_matmul_precision("high")`, which permits TF32 matmuls.

**`--workers`.** For the NODE, `0` is often faster: the dataset is already
tensors in RAM, so workers add process handover and a per-process copy for no
gain.

---

## What a run writes

```
<--out>/<model.name>/
├── best-<name>-epoch=NN.ckpt      best-validation-loss checkpoint
├── training_loss_log.png          loss curves (plus NFE for the NODE)
├── test_metrics.json              per-isotope MAE / RMSE / R² / MARE
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
