#!/bin/bash

# Split models into small and large groups
small_models=(
    # "google/gemma-2-2b-it"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)
big_models=(
    "google/gemma-2-9b-it"
    "Qwen/Qwen2.5-14B-Instruct"
)

# Initialize Conda and activate the environment
conda activate grid

# Function to submit jobs
submit_eval_jobs() {
    local MODEL=$1
    local MODEL_ID=$(basename $MODEL)
    
    run_name="run_5"
    if [[ "$MODEL_ID" == "Qwen2.5-3B-Instruct" ]]; then
        run_name="run_4"
    fi
        
    sbatch --job-name=rdo_eval \
            --output=./logs/rdo_eval/${MODEL_ID}/${run_name}_%j.out \
            --time=0-2 \
            --partition=${2} \
            --gres=gpu:${3}:1 \
            --qos=deadline \
            --wrap="python -m pipeline.run_eval --model_path $MODEL --wandb_group sb_data_with_retain_${MODEL_ID} --wandb_run $run_name"
}

# Process small models on A100
for MODEL in "${small_models[@]}"; do
    submit_eval_jobs "$MODEL" "gpu_a100" "a100"
done

# Process big models on H100 (adjust partition and GPU counts as needed)
for MODEL in "${big_models[@]}"; do
    submit_eval_jobs "$MODEL" "gpu_h100" "h100"
done

