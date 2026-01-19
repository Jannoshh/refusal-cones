small_models=(
    # "google/gemma-2-2b-it"
    "Qwen/Qwen2.5-1.5B-Instruct"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
)
big_models=(
    # "google/gemma-2-9b-it"
    # "Qwen/Qwen2.5-14B-Instruct"
)

conda activate grid

# Function to submit jobs
submit_eval_jobs() {
    local MODEL=$1
    local MODEL_ID=$(basename $MODEL)
    
    sbatch --job-name=dim_eval \
            --output=./logs/dim_eval/${MODEL_ID}/%j.out \
            --time=0-1 \
            --partition=${2} \
            --gres=gpu:${3}:1 \
            --qos=deadline \
            --wrap="python -m pipeline.run_pseudo_pipeline --model_path $MODEL"
}

# Process small models on A100
for MODEL in "${small_models[@]}"; do
    submit_eval_jobs "$MODEL" "gpu_a100" "a100"
done

# Process big models on H100 (adjust partition and GPU counts as needed)
for MODEL in "${big_models[@]}"; do
    submit_eval_jobs "$MODEL" "gpu_h100" "h100"
done

