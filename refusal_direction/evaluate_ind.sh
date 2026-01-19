#!/bin/bash

# Define models
small_models=(
    "google/gemma-2-2b-it"
)

# Define lambda and cutoff values to sweep over
lambdas=(100 500)
cutoffs=(1)
retains=(0 0.1 1)
diffs=("True" "False")

# Function to submit jobs
submit_eval_jobs() {
    local MODEL=$1
    local MODEL_ID=$(basename $MODEL)
    
    for lambda in "${lambdas[@]}"; do
        for cutoff in "${cutoffs[@]}"; do
            for retain in "${retains[@]}"; do
                for diff in "${diffs[@]}"; do
                    local config="repind${lambda}_cutoff${cutoff}_retain${retain}_diff${diff}"
                    
                    sbatch --job-name=directopt_ind_eval \
                           --output=./logs/directopt_ind_eval/${MODEL_ID}/${config}_%j.out \
                           --time=0-1 \
                           --partition=gpu_a100 \
                           --gres=gpu:a100:1 \
                           --dependency=afterok:1085436 \
                           --qos=default \
                           --wrap="python -m pipeline.run_eval --model_path $MODEL --wandb_group repind_symmetric_${MODEL_ID} --wandb_run ${config}"
                done
            done
        done
    done
}

# Process models on A100
for MODEL in "${small_models[@]}"; do
    submit_eval_jobs "$MODEL"
done