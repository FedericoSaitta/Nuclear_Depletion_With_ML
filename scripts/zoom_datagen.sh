#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 80
#SBATCH -t 7-0
#SBATCH -J zoom_datagen
#SBATCH -o zoom_datagen_%j.out
#SBATCH -e zoom_datagen_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abel.castanedarodriguez@student.manchester.ac.uk

module purge
# No `module load python` and no conda: uv provisions CPython 3.12 from
# .python-version, and .venv-sim carries the locked dependency set.
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$SLURM_SUBMIT_DIR/.venv-sim"
cd "$SLURM_SUBMIT_DIR"

# HDF5 file locking breaks on most parallel filesystems; datagen.py:65 sets
# this per-worker, but exporting here covers the OpenMC executable too.
export HDF5_USE_FILE_LOCKING=FALSE

uv run --extra sim --no-dev --locked python data_generation/zoom_datagen.py \
    --daily-results data_generation/results/worker_1_9f530216/depletion_results.h5 \
    --start-day 140 \
    --end-day 170 \
    -p data_generation/data_beavers.txt \
    -t 80 \
    -s 42 \
    --particles 50000 \
    --batches 60 \
    --inactive 15 \
    --dt 0.0416667 \
    -f chain_endfb71_pwr.xml
