#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 3
#SBATCH -t 7-0

#SBATCH -J quarter_datagen
#SBATCH -o datagen_%j.out
#SBATCH -e datagen_%j.err
# No --mail-user: SLURM mails the submitting user, which is the right person on
# any cluster account. Override with `sbatch --mail-user=<addr>` if needed.
#SBATCH --mail-type=ALL

module purge
# No `module load python` and no conda: uv provisions CPython 3.12 from
# .python-version, and .venv-sim carries the locked dependency set.
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$SLURM_SUBMIT_DIR/.venv-sim"
cd "$SLURM_SUBMIT_DIR"

# HDF5 file locking breaks on most parallel filesystems; datagen.py:65 sets
# this per-worker, but exporting here covers the OpenMC executable too.
export HDF5_USE_FILE_LOCKING=FALSE

uv run --extra sim --no-dev --locked python data_generation/quarter_datagen.py \
    -p "${POWER_HISTORY:-data_generation/data_beavers.txt}" \
    -n 1 \
    -c 1 \
    -t 10 \
    --particles 500 \
    --batches 60 \
    --inactive 15 \
    --dt 1 \
    -f chain_endfb71_pwr.xml