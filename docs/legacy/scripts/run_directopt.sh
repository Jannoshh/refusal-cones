#!/bin/bash

# Array of model names
models=("google/gemma-2b-it" "google/gemma-2-2b-it")

# Run for each model
for model in "${models[@]}"; do
    # Run base version
    sbatch directopt_ind.sh "$model"
    
    # Run train_subspace version 
    sbatch directopt_ind.sh "$model" train_subspace
done