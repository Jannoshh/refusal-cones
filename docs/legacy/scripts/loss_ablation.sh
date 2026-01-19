#!/bin/bash

#SBATCH --job-name=loss_ablation       # Job name
#SBATCH --output=./logs/make_loss_ablation/refusal_vector_%A_%a.out   # Output file with array job ID and task ID
#SBATCH --time=0-4           # Time limit (DD-HH:MM)
#SBATCH --partition=gpu_a100      # Partition name
#SBATCH --gres=gpu:a100:1         # Request 1 GPU of type a100
#SBATCH --qos=deadline             # Quality of Service
#SBATCH --array=1-11              # Run jobs 1 through 11 in parallel

python directopt.py --model meta-llama/Meta-Llama-3-8B-Instruct --n_inits 10 --job_number $SLURM_ARRAY_TASK_ID --make_loss_ablation 

