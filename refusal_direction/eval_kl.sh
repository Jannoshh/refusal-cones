#!/bin/bash

small_models=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
)

# Define lambda and cutoff values to sweep over
lambdas=(0 0.5 1 2 3 4 5 6 7 8)
n_runs=10

# Create a temporary script for each model and lambda
for MODEL in "${small_models[@]}"; do
    MODEL_ID=$(basename $MODEL)
    
    # Submit a separate job for each lambda value
    for lambda in "${lambdas[@]}"; do
        # Create temporary script
        temp_script="temp_${MODEL_ID}_lambda_${lambda}.sh"
        cat << EOF > "$temp_script"
#!/bin/bash

for i in \$(seq 1 $n_runs); do
    config="retain_lambda_${lambda}_run_\${i}"
    
    echo "Running evaluation for ${MODEL_ID} with lambda=${lambda}, run \${i}"
    python -m pipeline.run_eval \\
        --model_path ${MODEL} \\
        --wandb_group kl_ablation_${MODEL_ID} \\
        --wandb_run \${config} \\
        --lowest_loss_vector
done
EOF

        # Make the temporary script executable
        chmod +x "$temp_script"
        
        # Submit the job
        sbatch --job-name=eval_kl_ablation_${MODEL_ID} \
               --output=./logs/eval_kl_ablation/${MODEL_ID}/lambda_${lambda}_%j.out \
               --time=0-3 \
               --partition=gpu_a100 \
               --gres=gpu:a100:1 \
               --qos=deadline \
               "./$temp_script"
               
        # Clean up
        rm "$temp_script"
    done
done