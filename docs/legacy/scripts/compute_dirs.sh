#!/bin/bash

small_models=(
    # "google/gemma-2b-it"
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

for model in "${small_models[@]}"; do
    sbatch directopt_ind.sh "$model" 5 
done

for model in "${big_models[@]}"; do
    sbatch directopt_indh100.sh "$model" 5
done