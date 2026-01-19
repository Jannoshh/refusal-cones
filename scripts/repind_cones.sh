#!/bin/bash

# Split models into small and large groups
small_models=(
    # "google/gemma-2-2b-it"
    "meta-llama/Meta-Llama-3-8B-Instruct"
)

conda activate ref
# Function to submit jobs
submit_train_job() {
    local MODEL=$1
    local MODEL_ID=$(basename $MODEL)
    
    sbatch --job-name=directopt_sub_train \
           --output=./logs/repind_cones/${MODEL_ID}/%j.out \
           --time=0-8 \
           --partition=${2} \
           --gres=gpu:${3}:1 \
           --qos=default \
           --wrap="python directopt.py --model $MODEL --n_inits 1 --train_repind_cones"
}

# Process small models on A100
for MODEL in "${small_models[@]}"; do
    submit_train_job "$MODEL" "gpu_h100" "h100"
done