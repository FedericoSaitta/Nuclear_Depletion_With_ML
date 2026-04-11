#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -t 0-4

#SBATCH -J quarter_datagen
#SBATCH -o datagen_%j.out
#SBATCH -e datagen_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abel.castanedarodriguez@student.manchester.ac.uk

module purge
module load python/3.13
conda activate nuclear-ml

# Small test run: 10 workers x 1 thread = 10 CPUs (matches SBATCH -c 10)
# Low-fidelity settings just to check the pipeline works end-to-end
# always better to use more cores than threads to match the requested sbatch cpus

python data_generation/quarter_datagen.py \
    -p data_generation/example_power_history.csv \
    -n 1 \
    -c 10 \
    -t 1 \
    --particles 1000 \
    --batches 20 \
    --inactive 5 \
    -f chain_endfb71_pwr.xml