#!/bin/bash

# Split models into small and large groups
small_models=(
    # "google/gemma-2-2b-it"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
)
big_models=(
    # "google/gemma-2-9b-it"
    # "Qwen/Qwen2.5-14B-Instruct"
)

min_dim=8
max_dim=8

# Initialize Conda and activate the environment
conda activate grid

# Function to submit jobs
submit_eval_jobs() {
    local MODEL=$1
    local MODEL_ID=$(basename $MODEL)
    
    for dim in $(seq $min_dim $max_dim); do
        sbatch --job-name=subspace_basis_eval \
               --output=./logs/subspace_basis_eval/${MODEL_ID}/dim_${dim}_%j.out \
               --time=0-2 \
               --partition=${2} \
               --gres=gpu:${3}:1 \
               --qos=default \
               --wrap="python -m pipeline.run_eval --model_path $MODEL --wandb_group join_subspace_${MODEL_ID} --wandb_run dim_${dim} --wandb_project robust_refusal_subspace"
    done
}

# Process small models on A100
for MODEL in "${small_models[@]}"; do
    submit_eval_jobs "$MODEL" "gpu_a100" "a100"
done

# Process big models on H100 (adjust partition and GPU counts as needed)
for MODEL in "${big_models[@]}"; do
    submit_eval_jobs "$MODEL" "gpu_h100" "h100"
done