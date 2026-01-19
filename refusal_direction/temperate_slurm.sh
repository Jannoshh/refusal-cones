#!/bin/bash

# Split models into small and large groups
conda activate grid
model="google/gemma-2-2b-it"
n_completions=64
temperatures=(0.2 0.5 1 1.2)

for temperature in ${temperatures[@]}; do
    MODEL_ID=$(basename $model)
    sbatch --job-name=temperature_completions \
            --output=./logs/temperature_completions/${MODEL_ID}/temperature_${temperature}_%j.out \
            --time=0-2 \
            --partition=gpu_a100 \
            --gres=gpu:a100:1 \
            --qos=default \
            --wrap="python -m pipeline.run_temperature --temperature ${temperature} --n_completions ${n_completions}"
    done