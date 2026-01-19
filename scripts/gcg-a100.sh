#!/bin/bash

#SBATCH --job-name=gcg_llama       # Job name
#SBATCH --output=./logs/gcg_llama_%j.out   # Output file (stdout and stderr)
#SBATCH --time=0-10           # Time limit (DD-HH:MM)
#SBATCH --partition=gpu_a100      # Partition name
#SBATCH --gres=gpu:a100:1         # Request 1 GPU of type a100
#SBATCH --qos=default             # Quality of Service
#SBATCH --mem=16G
#SBATCH --array=0-6               # Array job with 6 tasks (for 6 GPUs)

# Check if a model argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Model argument is required"
    exit 1
fi

MODEL=$1

# Initialize Conda and activate the environment
conda init
conda activate act

echo "Running task $SLURM_ARRAY_TASK_ID with model $MODEL"

# Run the Python script directly
python run_gcg.py --model $MODEL --batch_size 256 --chunk_id $SLURM_ARRAY_TASK_ID --total_chunks $SLURM_ARRAY_TASK_COUNT --loss_weight 0




