# The training pipeline, end to end

How data becomes a trained surrogate, what the two models actually optimise, and
how to read the numbers that come out. Written against the code in
`src/nuclear_surrogates/`; every claim here is traceable to a named function.

The last section answers a specific question that the `BEST_7_Isotope_DNN`
results raise: **why does autoregressive rollout score better than teacher
forcing?** The short answer is that the two are not competing training methods —
they are two ways of *rolling out* the same trained model — and one of them
inherits the stability of the underlying physics while the other does not.

---

## 1. What the data is

A **run** is one simulated fuel pin, depleted by OpenMC under a randomly drawn
operating history: 100 depletion steps of 10 days each, so 101 sampled time
points spanning 990 days. Every step draws fresh power, fuel/moderator/clad
temperatures, moderator density and boron concentration from configured ranges
(`data_generation/datagen.py`, configured by
`data_generation/configs/casl_pincell.yaml`).

`casl_3305_runs_inter.h5` holds 3,305 such runs — 333,805 rows × 238 columns,
one column per nuclide in the depletion chain plus the operating state.

The CASL and BEAVRS pipelines share one pin model — fuel, helium gap,
Zircaloy-4 cladding, water — so a surrogate trained on one can be held against
the other. See `data_generation/README.md`, "One model, two histories".

The physics being learned is the **breeding chain**:

```
U238 --(n,γ)--> U239 --β⁻--> Np239 --β⁻--> Pu239 --(n,γ)--> Pu240 --(n,γ)--> Pu241 --(n,γ)--> Pu242
      (capture)      23.45 min      2.356 d        (capture)       (capture)        (capture)
```

Two timescales matter enormously later. U239 (half-life 23.45 min) and Np239
(2.356 d) are *fast*: within one 10-day step they turn over ~614 and ~4.2
half-lives respectively. They are not free variables — they sit at **secular
equilibrium**, algebraically slaved to the neutron flux. The Pu isotopes are
*slow*: they accumulate over the full 990 days.

---

## 2. Shared preprocessing

Both models go through the same path (`datamodule/dataset_helper.py`):

1. **Read.** Only the configured columns are pulled out of the HDF5.
2. **Detect run boundaries.** `detect_run_length` finds where `time_days` resets.
3. **Drop the end-of-run marker.** The depletion writer emits a NaN row at the
   end of every run; it is removed, leaving 100 usable points per run.
4. **Split by whole runs** — never mid-run, so no trajectory straddles the
   boundary.
5. **Fit scalers on the training split only**, then transform all three splits.
   The fitted state is persisted as a `Preprocessor` (`preprocessor.json`) so a
   checkpoint can be served without the training file.

**How the runs are dealt out** is `dataset.split.strategy`, and both models
support both values:

- `sequential_by_run` — contiguous slices in file order, so the test set is the
  tail of the file. The DNN's historical behaviour, and still the default when
  the key is absent, so an archived bundle rebuilds the partition it was
  trained with.
- `random_by_run` — a seeded permutation of whole runs
  (`dataset_helper.run_permutation`), partitioned at the same two boundaries by
  both datamodules. **Given one seed and one run count, the DNN and the NODE
  therefore hold out the same runs**, which is what makes the head-to-head
  comparison paired rather than merely matched.

All three shipped configs set `random_by_run` and 80 / 10 / 10; a test asserts
they agree (`tests/test_configs.py`), because the moment they do not, the two
models are being scored on different data. The defaults below apply only to a
config that omits the keys, and the scalers are what each config sets rather
than a property of either model:

| | default split | default strategy | scalers (as shipped) |
|---|---|---|---|
| DNN | 80 / 10 / 10 | sequential by run | per-column mix (quantile, robust, standard, MinMax) |
| NODE | 60 / 20 / 20 | random by run, seeded | MinMax throughout |

---

## 3. The DNN — a learned one-step map

### What it represents

Let `c(t) ∈ ℝ⁷` be the isotope concentrations and `u(t) ∈ ℝ⁶` the operating
conditions (power, three temperatures, moderator density, boron). The DNN learns
the **one-step transition map**

```
    Δĉ(t) = f_θ( c(t), u(t) )              with  ĉ(t+1) = c(t) + Δĉ(t)
```

The network is an MLP: 13 inputs → [128, 128] → 7 outputs, tanh activations,
residual connections where widths match (`models/model_architectures.py`,
`Deep_Neural_Network`).

### Why it predicts Δc and not c(t+1)

`target_delta_conc: true` makes the target the *change* over a step
(`dataset_helper.create_timeseries_targets`):

```python
y = target_data[idx + 1] - target_data[idx]      # delta mode
```

This matters because U238 changes by only a few percent across 990 days. Asked
to predict `c(t+1)` directly, a network trivially scores R² ≈ 1 by copying its
input — the interesting signal is a tiny residual on top of a large constant.
Predicting Δc removes the constant and forces the network to model the part that
is actually dynamics. The cost is that absolute concentration has to be
reconstructed before the DNN can be compared with the NODE at all, and *how* it
is reconstructed decides what the resulting number means — see §5 and §6.

### The training objective

One step, supervised, on **true** inputs:

```
    L(θ) = (1/N) Σ_i  Huber( f_θ(c_i, u_i) ,  Δc_i )
```

computed in *scaled* units, over samples drawn independently from anywhere in
any training run. There is no notion of a trajectory in the loss — every
`(state, next-state)` pair is an isolated supervised example. In sequence-model
language this is a **teacher-forced objective**: the network is only ever asked
to take one step from ground truth.

Optimiser AdamW (lr 9.134e-4, weight decay 1.338e-4), `ReduceLROnPlateau` on
validation loss, early stopping after 20 stale epochs out of a 500-epoch
budget. Batch 512, full dataset.

---

## 4. The NODE — a learned continuous-time generator

### What it represents

Instead of a discrete step, the NODE learns the **right-hand side of an ODE** and
obtains states by integrating it:

```
    dy/dt = A( u(t), y(t) ) · y(t)          y(0) = true initial state
```

with `y` the seven scaled concentrations and `u(t)` the power, held
piecewise-constant between grid points (zero-order hold,
`ForcedODEFunc.interpolate_forcing`).

`A` is not a free 7×7 matrix. It is produced by an MLP and then **constrained**
(`ODEFuncMatrix.build_matrix`):

- **Sparsity.** 36 of the 49 entries are forced to zero by
  `model.matrix_zero_entries`, leaving only the physically allowed transitions —
  the diagonal and the sub-diagonal feeds of the breeding chain.
- **Sign structure.** Diagonal entries pass through `−softplus` (a nuclide can
  only lose itself); off-diagonal entries through `+softplus` (a transition can
  only feed forward).

The sign structure makes `A` a **Metzler matrix**, which guarantees the scaled
concentrations can never go negative — a physical invariant enforced by
construction rather than learned. The network only outputs the 13 live entries,
so no capacity is spent on entries that are masked away.

### The training objective

Here is the structural difference from the DNN. `_forward_batch` takes **only**
`y(0)` and integrates the entire 100-step trajectory with `dopri5`, then scores
every point along it:

```
    L(θ) = (1/(B·T)) Σ_b Σ_t  ‖ ŷ_θ(t_b) − y(t_b) ‖²        ŷ_θ = odeint(f_θ, y(0), t)
```

The model never sees ground truth after `t = 0`. Gradients propagate back
*through the solver*, across all ~150 function evaluations. **The NODE is
therefore trained on its own rollout** — the regime the user's question is
reaching for — while the DNN is trained one step at a time.

Optimiser AdamW (lr 0.004), dopri5 at `rtol 1e-5 / atol 1e-7`, batch 64,
early stopping after 75 stale epochs. Only power is used as forcing, over the
whole dataset (`fraction_of_data: 1.0`).

### Consequence

| | training horizon | sees own errors during training? | cost |
|---|---|---|---|
| DNN | 1 step | no | seconds/epoch |
| NODE | 100 steps | yes | ~4 s/epoch, thousands of epochs |

---

## 5. How the models are evaluated

After training, `trainer.test` reloads the best checkpoint and produces every
figure and number. Two rollout modes are compared, and both are defined
identically for the two models — which is the point, since the paper puts their
numbers side by side.

**Teacher forcing (TF).** One step from the **true** state, every step:

```
    DNN     ĉ_TF(t+1) = c(t) + f_θ( c(t), u(t) )          c(t) is ground truth
    NODE    ĉ_TF(t+1) = odeint( f_θ, c(t), [t, t+1] )     c(t) is ground truth
```

Both copy `ĉ_TF(0) = c(0)`, which has no predecessor.
(`evaluation.teacher_forced_from_deltas`, `analysis.rollout.teacher_forced_predictions`.)

**Autoregressive (AR).** Free-running from the initial condition; the model is
given **its own** evolving state:

```
    DNN     ĉ_AR(t+1) = ĉ_AR(t) + f_θ( ĉ_AR(t), u(t) )    (metrics.model_autoregress)
    NODE    ĉ_AR(·)   = odeint( f_θ, c(0), t_span )       one solve, whole trajectory
```

For the DNN, only the isotope columns are fed back; power, temperatures and
boron keep their true values, because those are prescribed inputs, not
predictions.

**Both models are reported on the same 100 points.** The DNN's targets are one
step ahead of its inputs, so every series it produces naturally runs from c(1)
to c(N) — a window shifted one 10-day step later than the NODE's, at both ends.
`evaluation.align_to_initial` prepends the true c(0) and drops that last step,
which puts both models on c(0) … c(N-1), days 0–990. Without it their MARE
denominators are drawn from different windows and their error-growth curves are
indexed off by one step.

### Which metric is computed on what

The same table now applies to **both** models:

| metric | computed on | meaning |
|---|---|---|
| R², MAE, RMSE | absolute c, autoregressive | 990-day forecast accuracy |
| MARE (TF) | absolute c, one step from truth | one-step accuracy |
| MARE (AR) | absolute c, autoregressive | 990-day forecast accuracy |
| MALE growth curves | absolute c, both rollouts | how error evolves along a run |

Everything is in physical units (atom/b-cm), flattened over runs × steps, and
averaged unweighted across the seven nuclides.

`metrics.mare` is `mean|error| / max|truth|` — normalised by a single global
maximum, not per sample. Despite the name it is not mean absolute *relative*
error (see §8). Because its denominator comes from the truths in the test
window, it is only comparable between models that share a test set — which,
under `random_by_run`, they do.

**A caveat on R², with the floor measured.** Every run in the CASL set starts
from the *same* fresh-fuel composition, so the variance R² normalises by —
pooled over runs × steps — is dominated by the depletion curve all 3305 runs
share, not by their differing response to the power history. Which is the part a
surrogate exists to predict.

Quantify it with a no-model baseline: emit one ensemble-mean trajectory for
every run, ignoring the power history entirely.

| baseline | U238 | U239 | Np239 | Pu239 | Pu240 | Pu241 | Pu242 |
|---|---|---|---|---|---|---|---|
| R² | 0.977 | 0.048 | 0.053 | 0.987 | 0.979 | 0.974 | 0.947 |
| MARE | 5.5e-4 | 1.9e-1 | 1.8e-1 | 2.4e-2 | 2.9e-2 | 3.1e-2 | 2.7e-2 |

So R² = 0.99 on the actinide chain sits just above a floor of 0.95–0.99, while
R² = 0.99 on U239 or Np239 is a long way above 0.05. The two do not mean the
same thing. Quote the chain nuclides against that floor rather than against
zero; U239 and Np239 are where the models demonstrably track the forcing, and
they are also the two the fast-decay caveat in §8 applies to.

The familiar "R² ≈ 1 by copying the input" worry is a *one-step* problem, and is
what `target_delta_conc` exists to avoid (§3). It does not bite here: the AR
rollout is never handed c(t), and freezing each run at its true c(0) scores
between −0.8 and −5.8.

---

## 6. Why the teacher-forced curve is built from the true state

This is a methodological note, not a result. It records a construction that was
tried, produced a striking-looking finding, and was rejected — because the
finding was an artefact of the construction.

### The rejected version

The DNN predicts Δc, so an absolute-concentration curve has to be reconstructed
somehow. The obvious route is to integrate:

```
    c_TF(t) = c(0) + Σ_{s<t} Δ̂( c(s), u(s) )        ← rejected
```

Each Δ̂ is teacher-forced, but the *curve* is not: it is an accumulation of 100
separately-predicted deltas. Under that definition, MARE(AR) beat MARE(TF) on
six of seven nuclides, by 6.7× on U239 and 6.4× on Np239 — an eye-catching
result, with a clean mechanism behind it.

### The mechanism, which is real

Write the true one-step map as `c(t+1) = F(c(t), u(t))` and the model as
`F̂ = F + ε`. For the trajectory error `e(t) = ĉ(t) − c(t)`:

```
    AR        :  e(t+1) = J·e(t) + ε      →  ‖e‖ ≤ ‖ε‖ / (1 − ρ)   if ρ = ‖J‖ < 1
    TF-cumsum :  e(t+1) =   e(t) + ε      →  e(T) = Σ ε(t)          unbounded
```

TF-cumsum is an open-loop integrator: its effective Jacobian is the identity, so
every one-step error is banked permanently and nothing in the loop can notice
the drift, because the model is always handed the true state. AR is a closed
loop — if the rolled-out concentration drifts above truth the model sees an
above-equilibrium state and predicts a more negative Δ — so its error reaches a
bounded fixed point. Depletion is dissipative and its fast modes are strongly
attracting, so here the feedback is *negative* feedback and it stabilises. The
isotope ordering matched: U239 and Np239 relax to secular equilibrium almost
instantly (`ρ ≈ 0`, the largest gap); Pu242 is terminal, essentially a pure
integrator with `J ≈ I`, and was the one nuclide where AR stopped helping.

### Why it was rejected anyway

The comparison was between two different things. The NODE's teacher-forced
number is a genuine one-step quantity — each point is a fresh solve seeded from
truth (`rollout.py`) — so putting it next to a DNN number that accumulated 100
deltas compared one-step accuracy against 990-day drift and called the
difference an architecture difference.

`evaluation.teacher_forced_from_deltas` now adds each predicted delta to the
**true** concentration, matching the NODE step for step. Under that definition
neither metric is measuring the other's horizon, and the pair answers the two
questions it was meant to:

- MARE(TF): *how good is one step?*
- MARE(AR): *how good is a 990-day forecast?*

The contraction argument above still holds and is still the reason AR does not
blow up on this system — that is a genuine property of depletion, and worth
stating. What it is not is evidence that AR beats teacher forcing. Do not
generalise the sign either way: on a system with `ρ > 1` the same algebra gives
the familiar compounding blow-up.

---

## 7. What a run leaves behind

`<--out>/<model.name>/` after `nucml train --config ... --data ...`
(`--out` defaults to `results/`):

```
best-<name>-epoch=NN.ckpt        best-validation-loss checkpoint
model-bundle/                    the run's record — see below
test_metrics.json                per-isotope MAE / RMSE / R² / MARE, averaged too
<target>/                        per-isotope figures, one directory each
  predictions_vs_actual.png      scatter against truth
  residuals_combined*.png        residual structure, linear and log-log
  <t>_prediction_comparison.png  truth vs TF vs AR for one run
  <t>_{MAE,MALE}_growth_linear.png  error against timestep, TF and AR
  test_traj_N.png                power history over truth vs prediction, with R²
  test_all_trajectories.png      every test run overlaid, with mean |residual|
  {r2,mse}_score_importance.png  permutation importance (DNN)
  permutation_importance.csv     the numbers behind those two charts (DNN)
  jacobian_*                     sensitivity analysis (NODE)
  stepwise_importance*           per-step permutation importance (NODE)
depletion_matrix_{mean,evolution}.png   the learned A, in physical units (NODE)
permutation_importance.md        importance tables, ready to paste (DNN)
stepwise_importance.md           importance tables, ready to paste (NODE)
training_loss_log.png            loss curves, plus NFE for the NODE
```

Both models emit the same figures except where the physics differs: permutation
feature importance is a DNN output, and the Jacobian sweep, the depletion matrix
and the per-step importance need the ODE's right-hand side so they exist only for
the NODE. Each model's importance leaves the same three things behind — a
per-feature metric through `self.log`, a CSV per target, and one markdown table
at the root — so a results directory can be read back without re-running
anything. The DNN's `Type` column splits its inputs the way the NODE's does:
`State` for a feature that is also a target, `Forcing` for everything else.

The percentages in both tables are a feature's share of the total, with negative
importances (a shuffled column that happened to score *better*) clipped to zero
first. For the DNN the R² and MSE columns rank identically by construction —
R² = 1 − MSE/Var at fixed truth, so one is an affine map of the other — and the
two are kept because the absolute numbers are read in different units.

Everything else — the scatter, the residuals, the growth curves, the
trajectories — is drawn by the same code, fed the same quantities on the same
100-point window, so the head-to-head comparison of §5 is like-for-like.

`--no-analyses` skips all of it, figures and analyses alike, while still writing
`test_metrics.json` and the bundle. That is the flag for a hyperparameter sweep,
where the Jacobian and importance passes re-integrate the whole test set for
output nobody reads.

The one thing it does **not** skip is `training_loss_log.png`, and the reason is
the rule the flag follows: it suppresses exactly what `nucml plots` can put back.
`plots` replays `trainer.test` against a bundle and never runs `fit`, so the loss
curve is the single figure that could not be recovered without retraining.

The **bundle** is the unit of publication — a checkpoint alone is half a model,
because the other half is the fitted scalers:

```
model-bundle/
├── weights.ckpt              the checkpoint
├── preprocessor.json         fitted scalers, plain text, source of truth
├── preprocessor.joblib       convenience copy
├── config.resolved.yaml      the fully-merged config
├── split_indices.json        which runs were train / val / test
└── metadata.json             git SHA, seed, dataset SHA-256, library versions
```

`metadata.json` records the *validation* metrics, because the bundle is written
before `trainer.test` runs. The test metrics — the per-isotope MAE, RMSE, R² and
the TF/AR MARE pair — land in `test_metrics.json` beside the figures instead.

To run the model again, point one of the three bundle verbs at the directory:

```bash
uv run nucml infer --bundle results/<model_name>/model-bundle \
                   --data   datasets/new_runs.h5 \
                   --out    predictions/
```

To redraw this run's own figures instead of evaluating new data, use `plots` and
pass the dataset it was trained on. That rebuilds the run's train/val/test
split, keeps only the test share, and replays it with the bundle's weights and
scalers — the same runs the published figures came from, verified against
`split_indices.json` before anything is drawn.

To keep training from these weights, `finetune` warm-starts a fresh run from
them: weights only, so the optimizer and LR schedule restart.

`read_bundle` repoints the bundle's config at the bundle's own weights and
scalers and clears `dataset.path_to_data`, so the run cannot fall back on a
training file the bundle does not ship. The config is needed alongside the
weights because both models build their architecture from it — and for the NODE
the solver and its tolerances change the numbers, not just the runtime — which
is why `config.resolved.yaml` is in the bundle rather than assumed.

---

## 8. Known caveats

These bear directly on how the results above should be read.

The four that made the DNN/NODE numbers incomparable are now fixed, and the fix
moved every published DNN number — the results predating it cannot be mixed with
the results after it. For the record, they were: the two models split
differently (§2); the DNN's R²/MAE/RMSE were on one-step Δc while the NODE's
were on absolute concentration from the full rollout, which are different
quantities in different units (§5); the DNN's MARE(TF) accumulated 100 deltas
while the NODE's was a genuine one step (§6); and the two were reported on
100-point windows offset from each other by one 10-day step (§5).

What remains:

- **The comparison is level on data and metrics, not on everything.** The two
  models still differ by design in ways worth stating rather than removing: the
  objectives (Huber on scaled Δc vs MSE over whole scaled trajectories), the
  target scalers, and the epoch budgets (500 / patience 20 vs 5000 / patience
  75). Both select on `val_loss`, but those are different losses, so checkpoint
  selection is not on a common criterion. The NODE logs no `val_r2`/`val_mae` at
  all, which is why its `metadata.json` records infinities — they are absent
  values, not results.
- **`configs/dnn.yaml` sees five inputs the NODE does not** — fuel, moderator
  and clad temperature, moderator density, boron. `configs/dnn_no_state.yaml` is
  the matched-input variant, and the one to compare against the NODE; the other
  is the ablation. BEAVRS has none of those five columns, so only the matched
  variant can be evaluated there at all.
- **No spread on any number.** Every result is a single training run on a single
  split. There are no repeats and no error bars on the model comparison.
- **`mare` is misnamed.** It is `mean|error| / max|truth|` — normalised by one
  global maximum, not per sample — so it is not mean absolute *relative* error.
  Published numbers depend on the current definition, so it must not drift. Its
  denominator is drawn from the truths in the test window, so it is only
  comparable across models that share a test set.
- **R² on absolute concentration has a high floor.** All runs share one initial
  composition, so most of the pooled variance is the depletion curve they have
  in common. A no-model baseline already reaches 0.95–0.99 on the actinide chain
  (§5 tabulates it). R² is the right quantity for *comparing* the two models,
  since it is now the same quantity for both, but on its own it overstates how
  much either has learned about the power history.
- **A run's numbers depend on the machine it was produced on.** `evaluation.py`
  notes that Windows and Linux `log10` disagree by 1–3 ULPs, which is
  percent-level on MALE. Both models should be regenerated on one machine at one
  commit before their numbers are tabulated together.
- **The depletion-matrix figure's unit conversion is approximate.** It hardcodes
  a 1000-day span against 990 days of data (a 1% bias) and drops the MinMax
  offset, which leaves the U238 column uninterpretable as a rate. The matrix the
  network builds is unaffected; only the conversion applied before plotting is.
- **The solver tolerance is below what the arithmetic delivers.** States are
  float32 (eps ≈ 1.2e-7), so the `atol` the goldens were generated at asks for
  more precision than the representation carries.
- **Fast-decay chain entries are unidentifiable at 10-day sampling.** U239 and
  Np239 are at equilibrium at every sampled point, so their concentrations are
  slaved to the local capture rate rather than to the trajectory's history.
  `tests/test_golden_eval.py` states this as an assertion rather than prose.
