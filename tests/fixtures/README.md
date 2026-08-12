# Golden fixtures

These files are the safety net. They pin what the published models compute, so a
refactor that changes a number fails a test instead of quietly changing a paper
figure. Everything here is committed and small (~2.4 MB total) — no Git LFS, and
no dependency on the datasets, which is what lets the golden tests run in CI.

## What each file is

| File | Size | What it pins |
|---|---|---|
| `golden_node_eval.yaml` | 2 KB | The configuration these fixtures were generated from — architecture, and the `rtol 1e-6 / atol 1e-8` the trajectories were integrated at. Frozen: editing it invalidates every file below. |
| `best-matrix_ode_7x7_breeding_chain-epoch=2367.ckpt` | 241 KB | The paper's NODE. Retrieved from cluster scratch — **this repo is now one of its few copies**. |
| `mini_casl_10runs.h5` | 1.9 MB | First 10 runs (1010 rows) sliced deterministically from `casl_3305_runs_inter.h5`. |
| `preprocessor.json` | 8 KB | The fitted scalers. See "Why this exists" below. |
| `golden_node_preds.npy` | 27 KB | Autoregressive trajectories, scaled units. |
| `golden_node_trues.npy` | 27 KB | Ground truth for the above. |
| `golden_node_tf_preds.npy` | 27 KB | Teacher-forced rollout — a separate model computation. |
| `golden_node_metrics.json` | <1 KB | Per-target R², MAE, MARE. |
| `golden_node_eval.json` | 88 KB | Error-growth curves and the TF-vs-AR MARE comparison — the paper's figures. |
| `golden_matrix_A.npy` | <1 KB | The learned depletion matrix at 3 fixed (state, forcing) probes. |

## Why `preprocessor.json` exists

A checkpoint is half a model; the fitted scalers are the other half. They used
to be re-derived from the 542 MB training file on every load, which meant the
golden tests could only run on a machine that had it — so in CI they skipped,
silently, and the safety net did not actually exist.

`preprocessor.json` holds those scaler parameters as plain text. It is generated
from `datasets/casl_3305_runs_inter.h5` **once**, by `make_golden.py`, and
committed. Nothing in the test suite reads the training dataset;
`test_golden.py::test_inference_does_not_read_the_training_dataset` asserts that
and will fail if a read is reintroduced.

**It reproduces the historical inference path, deliberately.** The parameters are
fit on `path_to_data` under the config's `fraction_of_data`, which is what
inference mode did before bundles existed. That keeps the golden values
byte-identical across the change. It is *not* the "correct" train-split
preprocessor — the epoch-2367 model's original split was an unseeded permutation
and is unrecoverable. These fixtures are regression anchors,
not a reproduction of the original training run.

Note `preprocessor.json` records `t_days: 1000.0` alongside
`t_days_data_span: 990.0`. The data really spans 990 days; 1000 is the constant
the published runs used. Both travel so the discrepancy is visible — correcting
what consumes it is `AUDIT.md` P1 and has not been done.

## Why the config lives here

These fixtures pin the output of a *specific* configuration, so the config is
part of the fixture. It used to be `configs/main_config.yaml`, which is a
working file — the moment its solver tolerances were loosened from
`rtol 1e-6 / atol 1e-8` to `1e-5 / 1e-7`, dopri5 took different steps and five
of the nine golden tests failed on trajectories differing by ~2%. Nothing was
wrong with the code; the anchor had simply moved.

Note the tolerance recorded here is what the fixtures were *generated* with. The
checkpoint was *trained* at `1e-5 / 1e-7` — see `AUDIT.md` P4. Reconciling the
two means regenerating.

## Regenerating

```
uv run --extra ml python tests/make_golden.py     # needs datasets/casl_3305_runs_inter.h5
```

**This erases the safety net if you use it to make a failing test pass.** A red
golden test means either you broke something or you changed something
deliberately. Only in the second case do you regenerate, and then:

1. `git diff tests/fixtures/` and confirm every moved number is one you meant to move.
2. Say why in the commit message.

## Tolerances

Defined once in `tests/golden_setup.py`, not scattered through the assertions.

| Comparison | Tolerance | Why |
|---|---|---|
| Trajectories (`TRAJECTORY_*`) | `rtol=5e-3, atol=1e-6` | **Not portable at a tighter bound.** dopri5 picks its steps from an error estimate, so a last-bit arithmetic difference can tip it into a different step sequence and the whole trajectory inherits that. These fixtures were generated with the CUDA build of torch on Windows; CI installs the CPU wheel on Linux, and the two disagree by up to ~2e-3 relative. Still a real anchor: every genuine regression seen so far moved trajectories by >= 2e-2, four times this bound. |
| Matrix probes (`MATRIX_*`) | `rtol=1e-5, atol=1e-7` | No ODE solve, so no adaptive amplification — stays tight, and does pass cross-platform. |
| Scalar metrics | `rtol=1e-4` | Accumulated over ~1000 points. |
| Preprocessor round-trip | `atol=1e-12` | Pure serialisation — must be exact. |

Across a dependency upgrade (torch, sklearn), loosen trajectory tolerances
consciously and record it. Never loosen one silently to clear a red build.

## Not pinned yet

- The DNN. Its paper checkpoint is at
  `results/legacy_ML/Best_Chain_Result/best-Best_Chain_Result-epoch=176.ckpt`
  (236 KB) and its config is `configs/BEST_7_Isotope_DNN.yaml`; the DNN test path
  uses a deterministic sequential split, so it would need no extra fixtures
  beyond the checkpoint.
- The scaled→physical depletion-matrix conversion (`_get_unscaling_matrix`).
  `golden_matrix_A.npy` pins the matrix the network builds, not the unit
  conversion applied before it is plotted — and that conversion has known issues
  (`AUDIT.md` P1-P2).
- Jacobian sensitivity and stepwise importance.
