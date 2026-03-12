import os
import subprocess
from pathlib import Path

# Set the global variables needed for the job submission
CPUS = 12
GPUS = 1
EMAIL = 'federico.saitta@student.manchester.ac.uk'

# federico.saitta@student.manchester.ac.uk
# abel.castanedarodriguez@student.manchester.ac.uk

# Additional configuration
PARTITION = 'gpuL' # Options: 'gpuA', 'gpuA40GB', 'gpuL'
WALLTIME = '0-12'   # 1 day (format: days-hours)
JOB_NAME = 'gpu_job'
CUDA_VERSION = '11.8.0'  # Update this to an available version
PYTHON_SCRIPT = 'main.py'

def create_slurm_script(output_file='submit_gpu_job.sh'):
  """Generate a Slurm job submission script."""
    
  slurm_script = f"""#!/bin/bash --login
#SBATCH -p {PARTITION}                    # GPU partition
#SBATCH --gres=gpu:{GPUS}                 # Request {GPUS} GPU(s)
#SBATCH --ntasks-per-node={CPUS}          # Number of tasks per node
#SBATCH -t {WALLTIME}                     # Wallclock time limit
#SBATCH --mail-type=ALL                   # Email notifications
#SBATCH --mail-user={EMAIL}
#SBATCH -J {JOB_NAME}                     # Job name
#SBATCH -o logs/job_%j.out                # Standard output log
#SBATCH -e logs/job_%j.err                # Standard error log

conda activate nuclear-ml

# Launch nvidia-smi in background, logging every 10 seconds
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 10 > logs/gpu_usage.log &

# Run your Python script
python {PYTHON_SCRIPT}

kill %1
"""
    
  # Create logs directory if it doesn't exist
  Path('logs').mkdir(exist_ok=True)
  
  # Write the script to file
  with open(output_file, 'w') as f:
    f.write(slurm_script)
  
  # Make the script executable
  os.chmod(output_file, 0o755)
  
  print(f"Slurm script created: {output_file}")
  return output_file


def submit_job(script_file):
  """Submit the job to Slurm."""
  try:
    result = subprocess.run(
      ['sbatch', script_file],
      capture_output=True,
      text=True,
      check=True
    )
    print(f"Job submitted successfully!")
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
  print(f"Configuring job with:")
  print(f"  CPUs: {CPUS}")
  print(f"  GPUs: {GPUS}")
  print(f"  Partition: {PARTITION}")
  print(f"  Email: {EMAIL}")
  print(f"  Python script: {PYTHON_SCRIPT}")
  print()
  
  # Create the Slurm script
  script_file = create_slurm_script()
  
  # Ask for confirmation before submitting
  response = input("Do you want to submit this job? (yes/no): ").lower()
  
  if response in ['yes', 'y']:
    submit_job(script_file)
  else:
    print(f"Job not submitted. You can manually submit with: sbatch {script_file}")


if __name__ == '__main__':
  main()