# %%
import nanogcg
import torch
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# %%
import argparse
parser = argparse.ArgumentParser(description="Run GCG")
parser.add_argument('--chunk_id', type=int, required=True, help="Specify the chunk ID for this run")
parser.add_argument('--total_chunks', type=int, required=True, help="Specify the total number of chunks")
args = parser.parse_args()

# %%
# Set the CUBLAS_WORKSPACE_CONFIG environment variable for deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True, warn_only=True)

# Configuration
CACHE_DIR = '/ceph/hdd/students/elsj/huggingface'
MODEL_ID = "google/gemma-2-2b-it"
SAVE_PATH = f'results/repind_gcg/aux_50_newtargets/{args.chunk_id}.jsonl'
os.makedirs(f'results/repind_gcg/aux_50_newtargets', exist_ok=True)

# %%
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True, warn_only=True)

# %%
import json
# with open("data/splits/jailbreak_test.json", "r") as f:
#     jailbreak_dataset = json.load(f)
# print(len(jailbreak_dataset))
with open("results/refusal_dir/gemma-2-2b-it/completions/jailbreakbench_ablation_evaluations.json", "r") as f:
    dataset = json.load(f)
completions = dataset["completions"] 
jailbreak_dataset = []
for completion in completions:
    jailbreak_dataset.append({
        "instruction": completion["prompt"],
        "target": completion["response"][:20]
    })

# %%


# Track finished goals
import jsonlines
if os.path.exists(SAVE_PATH):
    with jsonlines.open(SAVE_PATH, 'r') as jsonl_f:
        finished_goals = {obj["instruction"] for obj in jsonl_f}
else:
    finished_goals = set()

# Split the dataset into chunks based on the total number of chunks
chunk_size = len(jailbreak_dataset) // args.total_chunks
start_idx = args.chunk_id * chunk_size
end_idx = start_idx + chunk_size if args.chunk_id < args.total_chunks - 1 else len(jailbreak_dataset)
dataset_chunk = jailbreak_dataset[start_idx:end_idx]

# Filter out already finished goals
print(f"Finished goals: {len(finished_goals)}/{len(dataset_chunk)}")
dataset_chunk = [attack for attack in dataset_chunk if attack["instruction"] not in finished_goals]
if len(dataset_chunk) == 0:
    print(f"No goals to run for chunk {args.chunk_id}")
    exit()

# %%
# %%
CACHE_DIR = '/ceph/hdd/students/elsj/huggingface'
model_id = "google/gemma-2-2b-it"
# model_id = "google/gemma-2b-it"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)

#%%
# %%
repind_vector = torch.load("lowest_loss_vector (2).pt").to(torch.bfloat16)
repind_vector = repind_vector / repind_vector.norm()

# %%
def test_suffix(prompt, max_new_tokens=150):
    chat = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).replace("<bos>", "")
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=False)


# Update config with new parameters
config = GCGConfig(
    seed=42,
    num_steps=250,
    batch_size=128,
    optim_str_init=" ".join(["!"] * 20),
    target_layers=[15],
    use_directional_ablation=True,
    use_activation_addition=False,
    directional_ablation_mode="current",
    use_crossentropy=True,
    aux_loss_weight=10,
    target_tokens="last_chat_token",
    use_prefix_cache=False,
)

# Run GCG for each prompt and store results
for attack in dataset_chunk:
    messages = [{"role": "user", "content": f'{attack["instruction"]} {{optim_str}}'}]
    target = attack["target"]

    result = nanogcg.run(model, tokenizer, messages, target, config, repind_vector)
    result_dict = result.__dict__
    result_dict.update(attack)

    output = test_suffix(messages[0]["content"].replace(' {optim_str}', '') + ' ' + result.best_string)
    result_dict["generated_output"] = output
    with jsonlines.open(SAVE_PATH, mode='a') as writer:
        writer.write(result_dict)
    