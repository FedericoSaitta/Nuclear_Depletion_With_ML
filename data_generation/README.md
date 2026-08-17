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
| `common.py` — paths, the sweep driver, results extraction | `config.py` — load/validate a config, derive worker configs |
| `quarter_sim.py`, `reactor_sim.py` — the two OpenMC models | `dataset_io.py` — HDF5 writer, merge, provenance manifest |
| `tally_io.read_statepoint_tallies` (imports openmc locally) | `power_history.py` — parse and resample the BEAVRS history |
| | `nuclides.py` — which nuclides are tallied, and why |

The two model modules prefix every function `quarterpin_` / `pincell_`. They
used to export `create_materials` and `create_settings` under the same names
while returning different numbers of materials, which is a trap worth closing.

Note the two models genuinely differ beyond geometry: the quarter-pin model has
a helium gap and the pin-cell model does not, and only the pin-cell model
carries boron in the water. That predates this code and is not a refactoring
artefact.

## Inputs not in this repository

- **Nuclear data** — cross sections (~7 GB) and depletion chains, from
  <https://openmc.org/official-data-libraries/>, expected in `../data/`.
- `beavrs_cycle1_power.csv` **is** committed: 527 measured `day,percent` points
  over the 575-day cycle. `power_history.parse_power_history` also reads the
  table in its original release form, banner and header included, so a fresh
  download can be dropped in unedited.
