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
(`data_generation/datagen.py`).

`casl_3305_runs_inter.h5` holds 3,305 such runs — 333,805 rows × 238 columns,
one column per nuclide in the depletion chain plus the operating state.

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

The two models split differently, which is a known wart (`AUDIT.md` P5):

| | split | strategy | scalers |
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
is actually dynamics. The cost is that absolute concentration must be recovered
by summing, which is exactly where §6 becomes interesting.

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
validation loss, early stopping after 20 stale epochs. The published model
stopped at epoch 47 of a 500-epoch budget. Batch 512, full dataset.

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
`ForcedODEFunc._interpolate_forcing`).

`A` is not a free 7×7 matrix. It is produced by an MLP and then **constrained**
(`ODEFuncMatrix._build_matrix`):

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
early stopping after 75 stale epochs; the published model stopped at epoch 2367.
Only power is used as forcing, and only 10% of the dataset.

### Consequence

| | training horizon | sees own errors during training? | cost |
|---|---|---|---|
| DNN | 1 step | no | seconds/epoch |
| NODE | 100 steps | yes | ~4 s/epoch, thousands of epochs |

---

## 5. How the models are evaluated

After training, `trainer.test` reloads the best checkpoint and produces every
figure and number. Two rollout modes are compared, and it is essential to know
precisely what each one is.

**Teacher forcing (TF).** At every step the model is given the **true** state and
predicts one step ahead:

```
    Δĉ_TF(t) = f_θ( c(t), u(t) )            ← c(t) is ground truth
```

**Autoregressive (AR).** The model is given **its own** evolving state
(`metrics.model_autoregress`):

```
    Δĉ_AR(t) = f_θ( ĉ_AR(t), u(t) )          ĉ_AR(t+1) = ĉ_AR(t) + Δĉ_AR(t)
```

Only the isotope columns are fed back; power, temperatures and boron keep their
true values throughout, because those are prescribed inputs, not predictions.

**The step that decides everything.** To compare against measured
*concentrations*, both prediction streams are converted from deltas to absolute
values by cumulative summation from the same true initial concentration
(`evaluation.deltas_to_absolute`, applied in `dnn_model._build_trajectories`):

```
    c_TF(t) = c(0) + Σ_{s<t} Δĉ_TF(s)
    c_AR(t) = c(0) + Σ_{s<t} Δĉ_AR(s)
```

So the "teacher-forced" curve is **not** a one-step quantity. It is an
accumulation of 100 separately-predicted deltas. That is the crux of §6.

### Which metric is computed on what

A frequent source of confusion, worth stating plainly:

| metric | computed on | meaning |
|---|---|---|
| R², MAE, RMSE | Δc, teacher-forced | one-step accuracy — what the loss optimised |
| MARE (TF and AR) | absolute c, after cumsum | trajectory accuracy over 990 days |
| MALE growth curves | absolute c, after cumsum | how error evolves along a run |

`metrics.mare` is `mean|error| / max|truth|` — normalised by a single global
maximum, not per sample. Despite the name it is not mean absolute *relative*
error (`AUDIT.md` P7).

---

## 6. Why autoregressive beats teacher forcing here

### The observation

From the `BEST_7_Isotope_DNN` run (331 test runs, R²_avg = 0.9883):

| isotope | R² (one-step Δc) | MARE TF | MARE AR | TF / AR |
|---|---|---|---|---|
| U238 | 0.9888 | 0.000050 | 0.000049 | 1.02 |
| **U239** | 0.9944 | 0.118618 | **0.017700** | **6.70** |
| **Np239** | 0.9942 | 0.109115 | **0.017132** | **6.37** |
| Pu239 | 0.9835 | 0.006296 | 0.005498 | 1.15 |
| Pu240 | 0.9892 | 0.004759 | 0.003038 | 1.57 |
| Pu241 | 0.9682 | 0.004126 | 0.003460 | 1.19 |
| Pu242 | 0.9994 | 0.002418 | **0.002496** | **0.97** |

AR wins on six of seven, dramatically on the two fast isotopes — and *loses*,
narrowly, on Pu242. That pattern is not noise; it is the mechanism.

### The two error recursions

Write the true one-step map as `c(t+1) = F(c(t), u(t))` and the model as
`F̂ = F + ε`, where `ε` is the model's one-step error. Define the trajectory
error `e(t) = ĉ(t) − c(t)`.

**Autoregressive.** The model is evaluated at its own state, so the true map's
sensitivity enters:

```
    e(t+1) = F̂(ĉ(t)) − F(c(t))
           = [F(ĉ(t)) − F(c(t))] + ε(ĉ(t))
           ≈ J · e(t) + ε              where J = ∂F/∂c
```

**Teacher-forced, then accumulated.** The model is evaluated at the *true* state,
so its output does not depend on the accumulated error at all:

```
    ĉ_TF(t+1) = ĉ_TF(t) + Δ̂(c(t))
    e(t+1)    = e(t) + ε(t)            — i.e. J is replaced by the identity
```

### What that implies

These are the same recursion with different Jacobians, and the difference is
decisive:

```
    AR :  e(t+1) = J·e(t) + ε     →   ‖e‖ ≤ ‖ε‖ / (1 − ρ)     if ρ = ‖J‖ < 1
    TF :  e(t+1) =   e(t) + ε     →   e(T) = Σ ε(t)            unbounded
```

- **TF-cumsum is an open-loop integrator.** Its effective Jacobian is the
  identity — marginally unstable. Every one-step error is banked permanently.
  Zero-mean errors accumulate as a random walk (`~σ√T`); any systematic bias
  accumulates linearly (`~bT`). Nothing in the loop can ever notice, let alone
  correct, the drift, because the model is always handed the true state.
- **AR is a closed loop.** If the rolled-out concentration drifts above truth,
  the model — which has learned the real physics — sees an above-equilibrium
  state and predicts a more negative Δ, pulling it back. The error reaches a
  **bounded fixed point** set by the one-step error and the contraction rate.

So AR wins **exactly when the underlying physics is contracting** (`ρ < 1`) and
the model is accurate enough to inherit that contraction. Feeding a model its own
output is usually described as a liability — error compounding — and it is, for
chaotic or neutrally-stable systems. Depletion is neither: it is a dissipative
system whose fast modes are strongly attracting. Here the feedback is *negative*
feedback, and it stabilises.

### The evidence matches, isotope by isotope

The predicted ordering is that the AR advantage should track how strongly each
isotope is attracted back to its own equilibrium:

- **U239 and Np239 (6.7× and 6.4×).** Turning over 614 and 4.2 half-lives per
  step, these relax to secular equilibrium almost instantly — `ρ ≈ 0`, the
  strongest possible contraction. `U239_MAE_growth_linear.png` shows it exactly:
  the TF curve climbs without bound to ~3.6e-9 while **the AR curve rises once
  and then runs flat for all 100 steps**. That plateau is `‖ε‖/(1−ρ)`.
- **Pu239, Pu240, Pu241 (1.2–1.6×).** Intermediate — governed by a balance of
  capture and decay, so partially self-correcting.
- **U238 (1.02×).** Depletes only a few percent; both methods are accurate to
  5e-5 and there is nothing to separate them.
- **Pu242 (0.97× — AR slightly worse).** The decisive counter-example. Pu242 is
  the **terminal** nuclide: it is produced by Pu241 capture and has essentially
  no loss channel, so it only ever accumulates. Its dynamics *are* a pure
  integrator, `J ≈ I`. The theory therefore predicts AR should have no restoring
  force and should degenerate to TF's behaviour — and
  `Pu242_MAE_growth_linear.png` shows precisely that: the two curves lie on top
  of one another, both growing without bound, neither saturating.

One isotope where the physics offers no contraction is the one isotope where
autoregressive rollout stops helping. That is a strong confirmation.

### What this does and does not say about training

A precise correction, because it matters for how the result is written up:

- **AR and TF are not two training methods here.** Both columns come from *one*
  model, trained *one* way — the DNN's one-step supervised objective. They differ
  only in how that fixed model is rolled out at test time.
- **The right conclusion** is about deployment and about metrics: for this
  system, autoregressive rollout is the better way to *use* the model, and the
  TF-cumsum curve is a poor proxy for trajectory accuracy. Reporting only the
  teacher-forced number would understate the surrogate — the opposite of the
  usual worry.
- **The underlying intuition is still sound**, and the repository already acts on
  it: the NODE *is* trained on its own rollout (§4), which is why its loss is
  computed over whole trajectories. If you want a genuine training-method
  comparison, that is the axis — one-step objective (DNN) versus multi-step
  objective (NODE) — not the AR/TF columns of a single model.
- **Do not generalise the sign of the effect.** AR beats TF-cumsum here because
  depletion is contracting. On a system with `ρ > 1`, the same algebra predicts
  the familiar compounding blow-up.

### A caveat on the comparison itself

TF and AR are compared on absolute concentration, which both reach by
accumulation. If instead you compare them on the quantity the model actually
predicts — Δc — teacher forcing wins trivially, since it is handed the true
inputs. That comparison is the reported R² (0.9883). The two numbers answer
different questions, and both belong in a write-up:

- R² on Δc: *how good is one step?*
- MARE on AR trajectories: *how good is a 990-day forecast?*

---

## 7. What a run leaves behind

`results/<model.name>/` after `nucml --config ...`:

```
best-<name>-epoch=NN.ckpt        best-validation-loss checkpoint
model-bundle/                    the run's record — see below
<target>/                        per-isotope figures, one directory each
  predictions_vs_actual.png      scatter against truth
  residuals_combined*.png        residual structure, linear and log-log
  <t>_prediction_comparison.png  truth vs TF vs AR for one run
  <t>_{MAE,MALE}_growth_*.png    error against timestep — the §6 evidence
  {r2,mse}_score_importance.png  permutation importance (DNN)
  jacobian_*                     sensitivity analysis (NODE)
  stepwise_importance*           per-step permutation importance (NODE)
depletion_matrix_{mean,evolution}.png   the learned A, in physical units (NODE)
training_loss_log.png            loss curves, plus NFE for the NODE
```

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

A row in `Chain_Model.db` records the run's identity and results and points at
this directory via `bundle_path`; the configuration itself is read from
`config.resolved.yaml` rather than duplicated into columns.

---

## 8. Known caveats

Recorded in full in `AUDIT.md`; the ones that bear on reading these results:

- **P5** — the DNN and NODE use different split protocols and different amounts
  of data, so their headline numbers are not a like-for-like comparison.
- **P6** — every number is a single training run on a single split; there is no
  spread.
- **P7** — `mare` is normalised by a global maximum, not per sample.
- **P1/P2** — the depletion-matrix figure's conversion to physical units carries
  a 1% time-span bias and drops the MinMax offset, which makes the U238 column
  uninterpretable as a rate.
