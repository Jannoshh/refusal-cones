#!/bin/bash

# Split models into small and large groups
small_models=(
    "google/gemma-2-2b-it"
)

min_dim=2
max_dim=5

# Model-specific samples per process
samples_per_process=256
max_samples=256

# Initialize Conda and activate the environment
conda activate grid

# Function to submit jobs
submit_eval_jobs() {
    local MODEL=$1
    local MODEL_ID=$(basename $MODEL)
    
    for dim in $(seq $min_dim $max_dim); do
        for sample_start_idx in $(seq 0 $samples_per_process $((max_samples-1))); do
            sample_end_idx=$((sample_start_idx + samples_per_process))
            sbatch --job-name=repind_cone_sample_eval \
                   --output=./logs/repind_cone_sample_eval_fixed/${MODEL_ID}/dim_${dim}_%j.out \
                   --time=0-8 \
                   --partition=${2} \
                   --gres=gpu:${3}:1 \
                   --qos=default \
                   --dependency=afterok:1142194 \
                   --wrap="python -m pipeline.eval_samples --model_path $MODEL --wandb_group repind_subspace_retain1_repind200_fixed_sum_no_dim_${MODEL_ID} --wandb_run dim_${dim} --wandb_project robust_refusal_subspace --sample_start_idx $sample_start_idx --sample_end_idx $sample_end_idx"
        done
    done
}

# Process small models on A100
for MODEL in "${small_models[@]}"; do
    submit_eval_jobs "$MODEL" "gpu_a100" "a100"
done