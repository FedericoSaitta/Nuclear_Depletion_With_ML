# Data generation

OpenMC depletion pipelines. Runs in the **simulation** environment (`.venv-sim`),
which shares no imports with the ML half — see the repository README for how the
two environments are set up.

**The config describes the simulation; the flags describe the machine.** Every
physics parameter lives in a YAML under `configs/`; `-n/-c/-t/-s/--out` say how
many workers on how many cores, and nothing else.

## The three pipelines

| Entry point | Config | What it does |
|---|---|---|
| `datagen.py` | `configs/casl_pincell.yaml` | Randomised operating histories on a full pin cell. Produced the CASL datasets both published models are trained on. |
| `quarter_datagen.py` | `configs/beavrs_quarterpin.yaml` | BEAVRS Cycle 1's *measured* power history on a quarter pin cell, with flux/fission/capture tallies. One `integrate()` for the whole cycle. |
| `zoom_datagen.py` | `configs/beavrs_zoom.yaml` | Re-runs a window of that cycle at hourly resolution, starting from a daily run's depleted compositions. One `integrate()` per step, so low-power steps can drop to reduced fidelity. |

All three build the **same pin cell**, from `pin_sim.py`: fuel, helium gap,
Zircaloy cladding, borated water, reflective on all sides. What differs is the
*history* — sampled per step versus measured and fixed — and `model.symmetry`.

```bash
# generate
uv run --extra sim python data_generation/datagen.py \
    --config data_generation/configs/casl_pincell.yaml -n 8 -c 16 -s 42

# merge the per-worker files into one training set
uv run --extra sim python data_generation/merge_runs.py \
    data_generation/data datasets/casl_runs.h5 --drop-empty
```

`--help` works on all of them without OpenMC installed: argv is parsed before
the import, which is also what lets `OMP_NUM_THREADS` be set in time.

SLURM wrappers for the cluster are in `scripts/`: `casl_datagen.sh`,
`beavrs_datagen.sh`, `zoom_datagen.sh`.

## What a sweep writes

Each worker writes `data/worker_<id>.h5` directly in the layout the ML pipeline
reads — there is no CSV step. Alongside them goes `run_manifest.json`:

```
config          the resolved simulation config
workers         every worker id and its Monte-Carlo seed
chain_file      name, path and SHA-256 of the depletion chain
cross_sections  the cross-section library in use
openmc_version, git_sha, git_dirty, python, platform, created_utc
```

That file is the point of the exercise. A published `.h5` without it is a table
of numbers nobody can regenerate: the chain, the library and the per-worker
seeds all live outside the data.

Given the same master seed, `config.create_worker_configs` draws the same worker
seeds, so `-s S` makes a sweep reproducible. `common.run_sweep` offsets the
master seed per round — without that, every round redrew *identical* histories
and the dataset filled with duplicates that then straddled the train/test split.

## Module layout

Split along one line: **does it need OpenMC?** Everything that does not is
importable — and tested — in the ML environment, which is what gives this half
any CI coverage at all (`tests/test_datagen_io.py`).

| Needs OpenMC | Pure Python |
|---|---|
| `common.py` — paths, the sweep driver, one depletion step, results extraction | `config.py` — load/validate a config, derive worker configs |
| `pin_sim.py` — the OpenMC model, for every pipeline | `cli.py` — the `-n/-c/-t/-s/--out` flags the sweeps share |
| `tally_io.read_statepoint_tallies` (imports openmc locally) | `dataset_io.py` — HDF5 writer, merge, provenance manifest |
| | `power_history.py` — parse and resample the BEAVRS history |
| | `nuclides.py` — which nuclides are tallied, and why |
| | `tally_io.py` — the tally column schema and the power-fraction maths |

## One model, two histories

Every pipeline builds its model from `pin_sim.py`, and the only thing that
distinguishes one pin cell from another is `model.symmetry`:

| | `symmetry: full` | `symmetry: quarter` |
|---|---|---|
| what is modelled | the whole cell, boundaries at ±pitch/2 | one quadrant, boundaries at 0 and +pitch/2 |
| regions | fuel, He gap, Zr-4 clad, water — identical | identical |
| areas | π r², pitch² | a quarter of each |
| cost | 4× the tracking | — |

Both describe the same infinite lattice of identical pins. A geometry difference
between two pipelines leaves no trace in the generated columns, so nothing
downstream could catch one; `tests/test_datagen_io.py`'s
`test_every_shipped_config_models_the_same_pin` holds the shared values equal
instead, on the configs themselves.

Symmetry is a config key rather than a property of the script, so **an entry
point is chosen by the history it applies, not by the geometry**:
`datagen.py --config <a quarter-pin config>` is a sampled history on a quarter
pin, and it works.

What still differs between the pipelines is the **operating history**, and that
is deliberate:

| | CASL (`datagen.py`) | BEAVRS (`quarter_datagen.py`, `zoom_datagen.py`) |
|---|---|---|
| power | sampled per step from a range | measured, from `beavrs_cycle1_power.csv` |
| temperatures, moderator density | sampled per step | fixed for the run, under `model` |
| boron | sampled per step, 0–1000 ppm | none — pure H₂O |
| tallies | none | flux, per-nuclide fission and (n,γ) |

## Inputs not in this repository

- **Nuclear data** — cross sections (~7 GB) and depletion chains, from
  <https://openmc.org/official-data-libraries/>, expected in `../data/`.
- `beavrs_cycle1_power.csv` **is** committed: 527 measured `day,percent` points
  over the 575-day cycle. `power_history.parse_power_history` also reads the
  table in its original release form, banner and header included, so a fresh
  download can be dropped in unedited.
