# %%
"""
NOTE: This file is a notebook/script for interactive analysis, not a pytest test file.
It is skipped during test collection.
"""
import sys
import os

# Skip this module during pytest collection - it's a script, not a test file
if "pytest" in sys.modules:
    import pytest
    pytest.skip("Skipping notebook/script file", allow_module_level=True)

import torch
import torch.nn as nn
import json
from torch import Tensor
from jaxtyping import Float
from tqdm import tqdm
import argparse

# wandb is optional
try:
    import wandb
except ImportError:
    wandb = None

def parse_args():
    # Default values
    defaults = {
        # 'model': 'google/gemma-2-2b-it',
        'model': 'Qwen/Qwen2.5-3B-Instruct',
        'train_direction': True,
        'train_subspace': False,
        'train_independent': False,
        'n_inits': 1,
    }
    
    # If running in interactive mode
    if not sys.argv[0].endswith('directopt.py'):
        return argparse.Namespace(**defaults)
    
    # If running from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=defaults['model'],
                    help='Model path')
    parser.add_argument('--train_direction', action='store_true', 
                    help='Train direction')
    parser.add_argument('--train_subspace', action='store_true', 
                    help='Train subspace')
    parser.add_argument('--train_independent', action='store_true', 
                    help='Train independent')
    parser.add_argument('--n_inits', type=int, default=defaults['n_inits'], 
                    help='Number of initializations')
    return parser.parse_args()

args = parse_args()
MODEL_PATH = args.model
CACHE_DIR = '/ceph/hdd/students/elsj/huggingface'

assert "gemma" in MODEL_PATH.lower() or "qwen2.5" in MODEL_PATH.lower() or "llama-3" in MODEL_PATH.lower(), "Model not supported"

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR, device_map='auto', torch_dtype=dtype)
model.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
model.requires_grad_(False)

# %%
# Test that model works (no longer need trace for this)
_ = model.tokenizer("Hello", return_tensors='pt')

# %%
model_id = MODEL_PATH.split("/")[-1]
refusal_directions = torch.load(f"results/refusal_dir/{model_id}/generate_directions/mean_diffs.pt")
refusal_results = json.load(open(f"results/refusal_dir/{model_id}/direction_metadata.json"))
best_layer = refusal_results["layer"]
best_token = refusal_results["pos"]
best_refusal_direction = torch.load(f"results/refusal_dir/{model_id}/direction.pt").to(model.dtype)
# %%
# for refusal_dir in refusal_directions.view(-1, 4096):
#     print(refusal_dir.norm())

# %%
SAVE_DIR = f"results/directopt/{MODEL_PATH.split('/')[-1]}/"
os.makedirs(SAVE_DIR, exist_ok=True)

add_layer = best_layer
alpha = best_refusal_direction.norm().detach().clone()
print(f"add_layer: {add_layer}, alpha: {alpha}")

# %%
harmful_train = json.load(open('data/splits/harmful_train.json'))
harmless_train = json.load(open('data/splits/harmless_train.json'))

harmless_train = harmless_train[:len(harmful_train)]
print(len(harmful_train), len(harmless_train))

# %%
harmful_train[:10]
# %%
# print 10 instructions with source HH-harmless
# for d in harmful_train[:50]:
#     if d["source"] == "HH-harmless":
#         print(d["instruction"])
#         print(d["source"])
#         print()
# # sort the instructions by source and print the first 10 for each source
# for source in set([d["source"] for d in harmful_train]):
#     print(f"\n=== {source} ===")
#     source_instructions = [d for d in harmful_train if d["source"] == source]
#     for d in source_instructions[:10]:
#         print(d["instruction"])
#         print(d["source"])
#         print()
# %%
GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""
QWEN25_CHAT_TEMPLATE = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def apply_chat_template(tokenizer, instructions: list[str]):
    if 'llama-3' in MODEL_PATH.lower():
        prompts = [LLAMA3_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    elif "gemma" in MODEL_PATH.lower():
        prompts = [GEMMA_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    elif "qwen2.5" in MODEL_PATH.lower():
        prompts = [QWEN25_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
        # prompts = [tokenizer.apply_chat_template([{"role": "user", "content": inst}], add_generation_prompt=True, tokenize=False) for inst in instructions]
    else:
        raise ValueError(f"Model {MODEL_PATH} not supported")
    return prompts

# %%
print(apply_chat_template(model.tokenizer, ["Hello"])[0])
# def generate_raw_completions(model, dataset, max_new_tokens=1):
#     instructions = apply_chat_template(model.tokenizer, [d['instruction'] for d in dataset])
#     decoded = []
#     tokens = []
#     for instruction in instructions:
#         input_tokens = model.tokenizer(instruction, add_special_tokens=True, padding=True, truncation=False)["input_ids"]
#         with model.generate(instruction, max_new_tokens=max_new_tokens, do_sample=False) as generator:
#             out = model.generator.output.save()
#         decoded.append(model.tokenizer.decode(out[0], skip_special_tokens=False))
#         tokens.append(out[0])
#     return decoded, tokens

# decoded, tokens = generate_raw_completions(model, harmful_train[:20])
# print(decoded)
# %%
# get first token completion
def generate_completions(model, dataset, max_new_tokens=1):
    instructions = apply_chat_template(model.tokenizer, [d['instruction'] for d in dataset])
    print(instructions[0])
    decoded = []
    tokens = []
    for instruction in instructions:
        input_tokens = model.tokenizer(instruction, add_special_tokens=True, padding=True, truncation=False, return_tensors='pt')
        start_token = input_tokens['input_ids'].shape[1]
        input_tokens = {k: v.to(model.device) for k, v in input_tokens.items()}

        with torch.no_grad():
            out = model.generate(**input_tokens, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=model.tokenizer.eos_token_id)

        decoded.append(model.tokenizer.decode(out[0][start_token:], skip_special_tokens=True))
        tokens.append(out[0][start_token:].cpu())
    return decoded, tokens

example_completions, example_tokens = generate_completions(model, harmful_train[:10])
print("example_completions:")
print(example_completions)
print("example_tokens:")
print(example_tokens)

# %%
if "gemma" in MODEL_PATH.lower():
    refusal_tokens = [235285]
elif "qwen2.5" in MODEL_PATH.lower():
    refusal_tokens = [40, 2121]
elif "llama-3" in MODEL_PATH.lower():
    refusal_tokens = [40]
else:
    raise ValueError(f"Model {MODEL_PATH} not supported")

# %%
from scoring import get_refusal_scores

def sample_hypersphere_gaussian(batch_size, dim):
    # Sample from standard normal distribution
    samples = torch.randn(batch_size, dim, dtype=model.dtype, device=model.device).abs()
    # Normalize to unit length
    samples = samples / torch.norm(samples, dim=1, keepdim=True)
    return samples
# %%
subspace = torch.load("subspace (1).pt")[-20:][6].cuda()
print(subspace.shape)
# %%
val_data = json.load(open("data/splits/harmful_val.json"))
harmful_val_instructions = apply_chat_template(model.tokenizer, [d['instruction'] for d in val_data])
# %%
import numpy as np
def evaluate_basis(basis, n_samples=16, n_instructions=32):
    samples = sample_hypersphere_gaussian(n_samples, len(basis))
    transformed_samples = [torch.matmul(sample, basis) for sample in samples]
    transformed_samples = [transformed_sample / torch.norm(transformed_sample) for transformed_sample in transformed_samples]
    scores = []
    for sample in transformed_samples:
        scores.append(get_refusal_scores(model, harmful_val_instructions[:n_instructions], refusal_tokens, sample, batch_size=n_instructions))
    return {"mean": round(np.mean(scores).item(), 2), "std": round(np.std(scores).item(), 2)}
# %%
basis = subspace
print(evaluate_basis(basis))
# %%
# Create triangular matrix of results
results = []
end_dim = 6 # len(subspace)
for i in range(0, end_dim):
    row = []
    for j in range(i+1, end_dim+1):
        basis = subspace[i:j]
        score = evaluate_basis(basis)
        row.append(score)
    results.append(row)

# %%
# Print triangular matrix using tabulate
from tabulate import tabulate

table = []
headers = ["Start Dim"] + [f"End Dim {j}" for j in range(2, end_dim+1)]

for i, row in enumerate(results):
    start_dim = i
    table_row = [start_dim]
    for j, score in enumerate(row):
        table_row.append(f"{score['mean']:.2f}±{score['std']:.2f}")
    table.append(table_row)

print("\nResults matrix:")
print(tabulate(table, headers=headers, tablefmt="grid"))
# %%
import einops

def projection_einops(activation, direction):
    proj = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction
    )
    return proj
def evaluate_basis_with_replacement(basis, n_samples=16, n_instructions=32, replace_idx=None):
    """
    Evaluate a basis set where one vector is replaced with a random orthogonal vector.
    
    Args:
        basis: Original basis vectors
        n_samples: Number of samples to evaluate
        n_instructions: Number of instructions to test
        replace_idx: Index of vector to replace (if None, evaluates original basis)
    """
    if replace_idx is not None:
        # Create copy of basis
        basis = basis.clone()
        
        # Generate random vector
        random_vec = torch.randn(basis.shape[1], device=basis.device, dtype=basis.dtype)
        
        # Make it orthogonal to all other basis vectors
        for i in range(len(basis)):
            if i != replace_idx:
                random_vec -= projection_einops(random_vec, basis[i])
        
        # Normalize
        random_vec = random_vec / random_vec.norm()
        
        # Replace the specified vector
        basis[replace_idx] = random_vec

    # Evaluate the modified basis
    samples = sample_hypersphere_gaussian(n_samples, len(basis))
    transformed_samples = [torch.matmul(sample, basis) for sample in samples]
    transformed_samples = [transformed_sample / torch.norm(transformed_sample) for transformed_sample in transformed_samples]
    scores = []
    for sample in transformed_samples:
        scores.append(get_refusal_scores(model, harmful_val_instructions[:n_instructions], refusal_tokens, sample, batch_size=n_instructions))
    return {"mean": round(np.mean(scores).item(), 2), "std": round(np.std(scores).item(), 2)}

# Test original basis and versions with each vector replaced
results = []
basis_size = 6  # Or whatever size you want to test

# Get original basis performance
original_score = evaluate_basis_with_replacement(subspace[:basis_size])
results.append(("Original", original_score))

# Test replacing each vector
for i in range(1, basis_size):
    score = evaluate_basis_with_replacement(subspace[:basis_size], replace_idx=i)
    results.append((f"Replace {i}", score))

# Print results in a table
from tabulate import tabulate

table = [[name, f"{score['mean']:.2f}±{score['std']:.2f}"] for name, score in results]
headers = ["Basis", "Score"]
print("\nReplacement Results:")
print(tabulate(table, headers=headers, tablefmt="grid"))

# %%
basis_size = len(subspace)
results = []
n_samples = 32
n_instructions = 64
for i in range(basis_size):
    # Without replacement
    score_without = evaluate_basis(subspace[:i+1], n_samples=n_samples, n_instructions=n_instructions)
    results.append([
        f"Dimensions 1 to {i+1}", 
        "Original",
        f"{score_without['mean']:.3f} ± {score_without['std']:.3f}"
    ])
    
    # With replacement
    score_with = evaluate_basis_with_replacement(subspace[:i+1], replace_idx=i)
    results.append([
        f"Dimensions 1 to {i+1}",
        "Last dimension randomized", 
        f"{score_with['mean']:.3f} ± {score_with['std']:.3f}"
    ])

headers = ["Dimension", "Version", "Score (mean ± std)"]
print("\nSubspace Dimension Analysis:")
print(tabulate(results, headers=headers, tablefmt="fancy_grid", numalign="center", stralign="center"))
# %%
