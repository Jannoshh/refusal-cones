#!/bin/bash

# Split models into small and large groups
small_models=(
    "google/gemma-2-2b-it"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "google/gemma-2-9b-it"
)
big_models=(
    # "google/gemma-2-9b-it"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "google/gemma-2-9b-it"
)

min_dim=4

# Model-specific maximum dimensions
declare -A max_dims=(
    ["google/gemma-2-2b-it"]=4
    ["meta-llama/Meta-Llama-3-8B-Instruct"]=6
    ["Qwen/Qwen2.5-1.5B-Instruct"]=8
    ["Qwen/Qwen2.5-3B-Instruct"]=8
    ["Qwen/Qwen2.5-7B-Instruct"]=8
    ["google/gemma-2-9b-it"]=4
    ["Qwen/Qwen2.5-14B-Instruct"]=7
)

# Model-specific samples per process
declare -A samples_per_process=(
    ["google/gemma-2-2b-it"]=32
    ["meta-llama/Meta-Llama-3-8B-Instruct"]=64
    ["Qwen/Qwen2.5-1.5B-Instruct"]=128
    ["Qwen/Qwen2.5-3B-Instruct"]=64
    ["Qwen/Qwen2.5-7B-Instruct"]=128
    ["google/gemma-2-9b-it"]=64
    ["Qwen/Qwen2.5-14B-Instruct"]=64
)
max_samples=63

# Initialize Conda and activate the environment
conda activate grid

# Function to submit jobs
submit_eval_jobs() {
    local MODEL=$1
    local MODEL_ID=$(basename $MODEL)
    local BATCH_SIZE=${samples_per_process[$MODEL]}
    local MAX_DIM=${max_dims[$MODEL]}
    
    for dim in $(seq $min_dim $MAX_DIM); do
        for sample_start_idx in $(seq 0 $BATCH_SIZE $max_samples); do
        sbatch --job-name=subspace_sample_eval \
               --output=./logs/subspace_sample_eval/${MODEL_ID}/dim_${dim}_%j.out \
               --time=0-8 \
               --partition=${2} \
               --gres=gpu:${3}:1 \
               --qos=default \
               --wrap="python -m pipeline.eval_samples --model_path $MODEL --wandb_group repind_subspace_retain0.1_${MODEL_ID} --wandb_run dim_${dim} --wandb_project robust_refusal_subspace --sample_start_idx $sample_start_idx --sample_end_idx $(($sample_start_idx + $BATCH_SIZE))"
        done
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