#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 7-0

#SBATCH -J casl_datagen
#SBATCH -o casl_datagen_%j.out
#SBATCH -e casl_datagen_%j.err
# No --mail-user: SLURM mails the submitting user, which is the right person on
# any cluster account. Override with `sbatch --mail-user=<addr>` if needed.
#SBATCH --mail-type=ALL

module purge
# No `module load python` and no conda: uv provisions CPython 3.12 from
# pyproject.toml's `requires-python`, and .venv-sim carries the locked set.
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$SLURM_SUBMIT_DIR/.venv-sim"
cd "$SLURM_SUBMIT_DIR"

# HDF5 file locking breaks on most parallel filesystems. `common.setup_paths`
# sets this per-worker; exporting here covers the OpenMC executable too.
export HDF5_USE_FILE_LOCKING=FALSE

# The randomised-history pipeline behind the CASL datasets. Every worker is a
# separate process holding its own in-memory nuclide data, so RAM rather than
# cores is the binding constraint — 16 single-threaded workers fit the
# allocation above. `-s` makes the sweep reproducible; drop it for fresh data.
uv run --extra sim --no-dev --locked python data_generation/datagen.py \
    --config "${DATAGEN_CONFIG:-data_generation/configs/casl_pincell.yaml}" \
    -n "${DATAGEN_RUNS:-8}" \
    -c 16 \
    -t 1 \
    -s "${DATAGEN_SEED:-42}"

# Merge the per-worker files into one dataset for training.
uv run --extra sim --no-dev --locked python data_generation/merge_runs.py \
    data_generation/data "datasets/casl_${SLURM_JOB_ID}.h5"
