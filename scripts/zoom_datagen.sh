#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 80
#SBATCH -t 7-0
#SBATCH -J zoom_datagen
#SBATCH -o zoom_datagen_%j.out
#SBATCH -e zoom_datagen_%j.err
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

# The daily run to zoom into is worker- and machine-specific (the directory name
# carries a random worker hash), so it has to be supplied at submission time:
#   DAILY_RESULTS=data_generation/results/worker_1_<hash>/depletion_results.h5 \
#       sbatch scripts/zoom_datagen.sh
: "${DAILY_RESULTS:?set DAILY_RESULTS=<path to a daily depletion_results.h5>}"

# Single process on all 80 allocated cores: this pipeline steps one integrate()
# call at a time, so the parallelism is OpenMP inside the transport solve rather
# than across workers.
uv run --extra sim --no-dev --locked python data_generation/zoom_datagen.py \
    --config "${DATAGEN_CONFIG:-data_generation/configs/beavrs_zoom.yaml}" \
    --daily-results "$DAILY_RESULTS" \
    -t 80 \
    -s 42
