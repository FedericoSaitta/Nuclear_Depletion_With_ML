#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -t 7-0

#SBATCH -J beavrs_datagen
#SBATCH -o beavrs_datagen_%j.out
#SBATCH -e beavrs_datagen_%j.err
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

# One worker on all 10 allocated cores. Workers x threads must equal `#SBATCH -c`
# above: this script used to ask for 3 cores and then run 10 OpenMP threads,
# which oversubscribes the allocation by more than 3x and slows the run down.
uv run --extra sim --no-dev --locked python data_generation/quarter_datagen.py \
    --config "${DATAGEN_CONFIG:-data_generation/configs/beavrs_quarterpin.yaml}" \
    -n 1 \
    -c 1 \
    -t 10

# Merge the per-worker files into one dataset for training.
uv run --extra sim --no-dev --locked python data_generation/merge_runs.py \
    data_generation/data "datasets/beavrs_cycle1_${SLURM_JOB_ID}.h5"
