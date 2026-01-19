#!/bin/bash

#SBATCH --job-name=train_orthogonal       # Job name
#SBATCH --output=./logs/orthogonal/train_orthogonal_%j.out   # Output file (stdout and stderr)
#SBATCH --time=0-1           # Time limit (DD-HH:MM)
#SBATCH --partition=gpu_a100      # Partition name
#SBATCH --gres=gpu:a100:1         # Request 1 GPU of type a100
#SBATCH --qos=deadline             # Quality of Service


# Initialize Conda and activate the environment
conda init
conda activate act

# Run the Python script directly
python directopt.py --model meta-llama/Meta-Llama-3-8B-Instruct --train_orthogonal