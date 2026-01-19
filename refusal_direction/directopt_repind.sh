#!/bin/bash

# Split models into small and large groups
small_models=(
    "google/gemma-2-2b-it"
)

min_dim=2
max_dim=5

# Initialize Conda and activate the environment
conda activate grid

# Function to submit jobs
submit_eval_jobs() {
    local MODEL=$1
    local MODEL_ID=$(basename $MODEL)
    
    for dim in $(seq $min_dim $max_dim); do
        sbatch --job-name=subspace_basis_eval \
               --output=./logs/repind_cone_basis_eval_fixed/${MODEL_ID}/dim_${dim}_%j.out \
               --time=0-1 \
               --partition=${2} \
               --gres=gpu:${3}:1 \
               --qos=default \
               --wrap="python -m pipeline.run_eval --model_path $MODEL --wandb_group repind_subspace_retain1_repind200_fixed_sum_no_dim_${MODEL_ID} --wandb_run dim_${dim} --wandb_project robust_refusal_subspace --lowest_loss_vector"
    done
}

# Process small models on A100
for MODEL in "${small_models[@]}"; do
    submit_eval_jobs "$MODEL" "gpu_a100" "a100"
done