"""Write and (optionally) submit a SLURM GPU job that runs `nucml`.

Every site-specific value is an environment variable with a default, so the
script carries no one machine's account details:

    SLURM_PARTITION=gpuA SLURM_WALLTIME=0-4 python scripts/gpu_job_runner.py
    NUCML_CONFIG=configs/dnn.yaml NUCML_DATA=datasets/other.h5 python scripts/gpu_job_runner.py
"""

import os
import subprocess
from pathlib import Path

CPUS = int(os.environ.get("SLURM_CPUS", 12))
GPUS = int(os.environ.get("SLURM_GPUS", 1))
PARTITION = os.environ.get("SLURM_PARTITION", "gpuL")  # e.g. gpuA, gpuA40GB, gpuL
WALLTIME = os.environ.get("SLURM_WALLTIME", "0-1")  # days-hours
JOB_NAME = os.environ.get("SLURM_JOB_NAME", "gpu_job")
# Unset by default: SLURM then mails the submitting user, which is correct on
# any account. Set SLURM_MAIL_USER to override.
EMAIL = os.environ.get("SLURM_MAIL_USER")
CONFIG = os.environ.get("NUCML_CONFIG", "configs/node.yaml")
DATA = os.environ.get("NUCML_DATA", "datasets/casl_3305_runs_inter.h5")


def create_slurm_script(output_file="submit_gpu_job.sh"):
    """Generate a Slurm job submission script."""

    mail_user = f"\n#SBATCH --mail-user={EMAIL}" if EMAIL else ""

    slurm_script = f"""#!/bin/bash --login
#SBATCH -p {PARTITION}                    # GPU partition
#SBATCH --gres=gpu:{GPUS}                 # Request {GPUS} GPU(s)
#SBATCH -n 1                              # One task: a single training process
#SBATCH -c {CPUS}                         # CPU cores for that task (dataloader workers)
#SBATCH -t {WALLTIME}                     # Wallclock time limit
#SBATCH --mail-type=ALL                   # Email notifications{mail_user}
#SBATCH -J {JOB_NAME}                     # Job name
#SBATCH -o logs/job_%j.out                # Standard output log
#SBATCH -e logs/job_%j.err                # Standard error log

export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"

# torch wheels bundle their own CUDA runtime, so no `module load cuda` is needed.
# --device and --workers come from the allocation above rather than from the
# config, which is why one config now serves both the laptop and the cluster.
uv run --extra ml --no-dev --locked nucml train \\
    --config {CONFIG} --data {DATA} --device cuda --workers {CPUS}
"""

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Write the script to file
    with open(output_file, "w") as f:
        f.write(slurm_script)

    # Make the script executable
    os.chmod(output_file, 0o755)

    print(f"Slurm script created: {output_file}")
    return output_file


def submit_job(script_file):
    """Submit the job to Slurm."""
    try:
        result = subprocess.run(
            ["sbatch", script_file], capture_output=True, text=True, check=True
        )
        print("Job submitted successfully!")
        print(result.stdout)

        # Extract job ID from output
        job_id = result.stdout.strip().split()[-1]
        print(f"Job ID: {job_id}")
        return job_id

    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        print(f"Error output: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: sbatch command not found. Are you on a Slurm system?")
        return None


def main():
    print("Configuring job with:")
    print(f"  CPUs: {CPUS}")
    print(f"  GPUs: {GPUS}")
    print(f"  Partition: {PARTITION}")
    print(f"  Email: {EMAIL or 'submitting user (SLURM default)'}")
    print(f"  Config: {CONFIG}")
    print(f"  Data: {DATA}")
    print()

    # Create the Slurm script
    script_file = create_slurm_script()

    # Ask for confirmation before submitting
    response = input("Do you want to submit this job? (yes/no): ").lower()

    if response in ["yes", "y"]:
        submit_job(script_file)
    else:
        print(f"Job not submitted. You can manually submit with: sbatch {script_file}")


if __name__ == "__main__":
    main()
