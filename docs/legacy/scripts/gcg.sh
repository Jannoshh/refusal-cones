#!/bin/bash

#SBATCH --job-name=gcg_abl       # Job name
#SBATCH --output=./logs/gcg_abl_%j.out   # Output file (stdout and stderr)
#SBATCH --time=0-10           # Time limit (DD-HH:MM)
#SBATCH --partition=gpu_h100      # Partition name
#SBATCH --gres=gpu:h100:1         # Request 1 GPU of type a100
#SBATCH --mem=16G                 # Memory request
#SBATCH --qos=default             # Quality of Service
#SBATCH --array=0-3             # Array job with 6 tasks (for 6 GPUs)

# Initialize Conda and activate the environment
conda init
conda activate act

echo "Running task $SLURM_ARRAY_TASK_ID"

# Run the Python script directly
python run_gcg.py --batch_size 512 --chunk_id $SLURM_ARRAY_TASK_ID --total_chunks 4 --loss_weight 10




