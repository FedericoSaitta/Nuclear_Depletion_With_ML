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
module load python/3.13
conda activate nuclear-ml

# Zoom into days 140-170 at hourly resolution (720 steps)
# Uses depleted fuel from the daily run as initial conditions
# Estimated runtime: ~6 hours with 40 threads

python data_generation/zoom_datagen.py \
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
