#!/bin/bash

#SBATCH --job-name=refusal_vector_ind       # Job name
#SBATCH --output=./logs/refusal_vectors/refusal_vector_%j.out   # Output file (stdout and stderr)
#SBATCH --time=0-8           # Time limit (DD-HH:MM)
#SBATCH --partition=gpu_a100      # Partition name
#SBATCH --gres=gpu:a100:1         # Request 1 GPU of type a100
#SBATCH --qos=default             # Quality of Service

# Check if a model argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Model argument is required"
    exit 1
fi
if [ $# -eq 1 ]; then
    echo "N inits argument is required"
    exit 1
fi

MODEL=$1
N_INITS=$2

# Initialize Conda and activate the environment
conda init
conda activate act

echo "Running task with model $MODEL"

# Run the Python script directly
python directopt.py --model $MODEL --n_inits $N_INITS --train_direction

