#!/bin/bash

# Split models into small and large groups
small_models=(
    # "google/gemma-2-2b-it"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-0.5B-Instruct"
)
big_models=(
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "google/gemma-2-9b-it"
    'meta-llama/Meta-Llama-3-8B-Instruct'
)

conda activate ref
# Function to submit jobs
submit_train_job() {
    local MODEL=$1
    local MODEL_ID=$(basename $MODEL)
    
    sbatch --job-name=directopt_sub_train \
           --output=./logs/refusal_subspaces/${MODEL_ID}/%j.out \
           --time=0-12 \
           --partition=${2} \
           --gres=gpu:${3}:1 \
           --qos=default \
           --wrap="python directopt.py --model $MODEL --n_inits 1 --train_subspace"
}

# Process small models on A100
for MODEL in "${small_models[@]}"; do
    submit_train_job "$MODEL" "gpu_a100" "a100"
done

# Process big models on H100 (adjust partition and GPU counts as needed)
for MODEL in "${big_models[@]}"; do
    submit_train_job "$MODEL" "gpu_h100" "h100"
done

