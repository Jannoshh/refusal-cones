# Split models into small and large groups
small_models=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "google/gemma-2-9b-it"
    # "google/gemma-2-2b-it"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "google/gemma-2-2b-it"
)

conda activate ref
# Function to submit jobs
submit_train_job() {
    local MODEL=$1
    local MODEL_ID=$(basename $MODEL)
    
    sbatch --job-name=directopt_train \
           --output=./logs/old_repind/${MODEL_ID}/%j.out \
           --time=0-6 \
           --partition=${2} \
           --gres=gpu:${3}:1 \
           --qos=deadline \
           --wrap="python old_directopt.py --model $MODEL --n_inits 3" 
}

# Process small models on A100
for MODEL in "${small_models[@]}"; do
    submit_train_job "$MODEL" "gpu_h100" "h100"
done