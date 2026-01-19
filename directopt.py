# %%
import torch
import torch.nn as nn
import json
from torch import Tensor
from jaxtyping import Float
from tqdm import tqdm
import os
import wandb
import argparse
import sys
import numpy as np

def parse_args():
    # Default values
    defaults = {
        'model': 'google/gemma-2-2b-it',
        # 'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
        # 'model': 'Qwen/Qwen2.5-7B-Instruct',
        'train_direction': False,
        'train_orthogonal': False,
        'train_subspace': False,
        'train_repind_subspace': False,
        'train_independent': False,
        'train_multiple_independent': False,
        'train_multiple_independent_via_cone': False,
        'make_kl_ablation': False,
        'make_add_loss_ablation': False,
        'make_loss_ablation': False,
        'job_number': 1,
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
    parser.add_argument('--train_orthogonal', action='store_true', 
                    help='Train orthogonal')
    parser.add_argument('--train_subspace', action='store_true', 
                    help='Train subspace')
    parser.add_argument('--train_repind_subspace', action='store_true', 
                    help='Train repind subspace')
    parser.add_argument('--train_independent', action='store_true', 
                    help='Train independent')
    parser.add_argument('--train_multiple_independent', action='store_true', 
                    help='Train multiple independent')
    parser.add_argument('--train_multiple_independent_via_cone', action='store_true', 
                    help='Train multiple independent via cone')
    parser.add_argument('--make_kl_ablation', action='store_true', 
                    help='Make KL ablation')
    parser.add_argument('--make_add_loss_ablation', action='store_true',
                    help='Make addition loss ablation')
    parser.add_argument('--make_loss_ablation', action='store_true',
                    help='Make loss ablation')
    parser.add_argument('--n_inits', type=int, default=defaults['n_inits'], 
                    help='Number of initializations')
    parser.add_argument('--job_number', type=int, default=defaults['job_number'], 
                    help='Job number')
    return parser.parse_args()

args = parse_args()
MODEL_PATH = args.model
CACHE_DIR = '/ceph/hdd/students/elsj/huggingface'

assert "gemma" in MODEL_PATH.lower() or "qwen2.5" in MODEL_PATH.lower() or "llama-3" in MODEL_PATH.lower(), "Model not supported"

# %%
from nnsight import LanguageModel
dtype = torch.bfloat16
model = LanguageModel(MODEL_PATH, cache_dir=CACHE_DIR, device_map='auto', torch_dtype=dtype)
model.requires_grad_(False)

# %%
with model.trace("Hello") as tracer:
    pass

# %%
model_id = MODEL_PATH.split("/")[-1]
refusal_directions = torch.load(f"results/refusal_dir/{model_id}/generate_directions/mean_diffs.pt")
refusal_results = json.load(open(f"results/refusal_dir/{model_id}/direction_metadata.json"))
best_layer = refusal_results["layer"]
best_token = refusal_results["pos"]
best_refusal_direction = torch.load(f"results/refusal_dir/{model_id}/direction.pt").to(model.dtype)

# %%
use_better_direction = False
if use_better_direction:
    direction_evaluations = json.load(open(f"results/refusal_dir/{model_id}/select_direction/direction_evaluations.json"))
    best_config = min(direction_evaluations, key=lambda x: x["refusal_score"] - x["steering_score"])
    best_layer = best_config["layer"]
    best_token = best_config["position"]
    best_refusal_direction = refusal_directions[best_token, best_layer].to(model.dtype)

# %%
SAVE_DIR = f"results/directopt/{MODEL_PATH.split('/')[-1]}/"
os.makedirs(SAVE_DIR, exist_ok=True)

add_layer = best_layer
alpha = best_refusal_direction.norm().detach().clone()
print(f"add_layer: {add_layer}, alpha: {alpha}")

# %%
# splits = "cb"
splits = "saladbench"
harmful_train = json.load(open(f'data/{splits}_splits/harmful_train.json'))
harmless_train = json.load(open(f'data/{splits}_splits/harmless_train.json'))

harmless_train = harmless_train[:len(harmful_train)]
print(len(harmful_train), len(harmless_train))

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
# %%
# get first token completion
def generate_first_token(model, dataset, max_new_tokens=1):
    instructions = apply_chat_template(model.tokenizer, [d['instruction'] for d in dataset])
    print(instructions[0])
    decoded = []
    tokens = []
    for instruction in instructions:
        input_tokens = model.tokenizer(instruction, add_special_tokens=True, padding=True, truncation=False)["input_ids"]
        start_token = len(input_tokens)
        with model.generate(instruction, max_new_tokens=max_new_tokens, do_sample=False) as generator:
            out = model.generator.output.save()
        decoded.append(model.tokenizer.decode(out[0][start_token:], skip_special_tokens=True))
        tokens.append(out[0][start_token:])
    return decoded, tokens

example_completions, example_tokens = generate_first_token(model, harmful_train[:10])
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
from torch.utils.data import DataLoader
import nnsight

harmful_train_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmful_train])
harmless_train_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmless_train])
# %%
mem_used_ratio = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
batch_size = 2 ** int(torch.log2(torch.tensor(4 / mem_used_ratio)))
print(f"Generation batch size {batch_size}")

# %%

# %%
from generate_utils import intervene_with_fn_vector_ablation, intervene_with_fn_vector_addition, generate_completions
intervene_with_fn_vector_ablation(model, harmful_train_instructions[:1], best_refusal_direction.to(model.dtype), max_new_tokens=40, batch_size=1)
# %%
module = model.model

# %%
def generate_harmful_targets(model, harmful_instructions, best_refusal_direction, targets_path, max_new_tokens):
    if os.path.exists(targets_path):
        return json.load(open(targets_path))
        
    ablation_completions = intervene_with_fn_vector_ablation(model, harmful_instructions, best_refusal_direction.to(model.dtype), max_new_tokens=max_new_tokens, batch_size=batch_size)
    
    # Create batch of dicts with new targets
    targets = []
    for i, instruction in enumerate(harmful_instructions):
        target_dict = {
            'prompt': instruction,
            'ablation': ablation_completions[i] if ablation_completions else "",
        }
        targets.append(target_dict)

    # Save all targets
    os.makedirs(os.path.dirname(targets_path), exist_ok=True)
    with open(targets_path, 'w') as f:
        json.dump(targets, f)
    
    return targets

def generate_harmless_targets(model, harmless_instructions, targets_path, max_new_tokens):
    if os.path.exists(targets_path):
        return json.load(open(targets_path))

    addition_completions = intervene_with_fn_vector_addition(model, harmless_instructions, best_layer, best_refusal_direction.norm(), best_refusal_direction, max_new_tokens=max_new_tokens, batch_size=batch_size)

    retain_completions = generate_completions(model, harmless_instructions, max_new_tokens=max_new_tokens-1, batch_size=batch_size)
    
    targets = []
    for i, instruction in enumerate(harmless_instructions):
        target_dict = {
            'prompt': instruction,
            'addition': addition_completions[i].split(".")[0] if addition_completions else "",
            'retain': retain_completions[i] if retain_completions else ""
        }
        targets.append(target_dict)
    
    os.makedirs(os.path.dirname(targets_path), exist_ok=True)
    with open(targets_path, 'w') as f:
        json.dump(targets, f)
        
    return targets

num_target_tokens = 30
# Set up paths
harmful_targets_path = f"results/directopt/{model_id}/{splits}/targets/harmful_targets.json"
harmless_targets_path = f"results/directopt/{model_id}/{splits}/targets/harmless_targets.json"

# Generate all targets
harmful_targets = generate_harmful_targets(model, harmful_train_instructions, best_refusal_direction, harmful_targets_path, num_target_tokens)
harmless_targets = generate_harmless_targets(model, harmless_train_instructions, harmless_targets_path, num_target_tokens)

# %%
from scoring import get_refusal_scores, refusal_score_fn

filter_data = True
if filter_data:
    print("Filtering data")
    harmful_train_scores = get_refusal_scores(model, harmful_train_instructions, refusal_tokens, batch_size=batch_size)
    harmless_train_scores = get_refusal_scores(model, harmless_train_instructions, refusal_tokens, batch_size=batch_size)
    
    # Filter instructions based on scores
    filtered_harmful_indices = [i for i, score in enumerate(harmful_train_scores) if score > 0]
    filtered_harmless_indices = [i for i, score in enumerate(harmless_train_scores) if score < 0]
    
    # Filter instructions
    filtered_harmful_train_instructions = [harmful_train_instructions[i] for i in filtered_harmful_indices]
    filtered_harmless_train_instructions = [harmless_train_instructions[i] for i in filtered_harmless_indices]
    
    # Filter targets
    filtered_harmful_targets = [harmful_targets[i] for i in filtered_harmful_indices]
    filtered_harmless_targets = [harmless_targets[i] for i in filtered_harmless_indices]
    
    print(f"Remaining harmful train instances: {len(filtered_harmful_train_instructions)}")
    
    # Balance datasets
    max_instances = min(len(filtered_harmful_train_instructions), len(filtered_harmless_train_instructions))
    filtered_harmful_train_instructions = filtered_harmful_train_instructions[:max_instances]
    filtered_harmless_train_instructions = filtered_harmless_train_instructions[:max_instances]
    filtered_harmful_targets = filtered_harmful_targets[:max_instances]
    filtered_harmless_targets = filtered_harmless_targets[:max_instances]
    
    print(f"Remaining harmless train instances: {len(filtered_harmless_train_instructions)}")
    
    # Update variables with filtered data
    harmful_train_instructions = filtered_harmful_train_instructions
    harmless_train_instructions = filtered_harmless_train_instructions
    harmful_targets = filtered_harmful_targets
    harmless_targets = filtered_harmless_targets

# Extract targets from filtered data
ablation_train_targets = [t["ablation"] for t in harmful_targets]
addition_train_targets = [t["addition"] for t in harmless_targets]
retain_train_targets = [t["retain"] for t in harmless_targets]

# %%
def build_prompts_and_labels(model, harmful_instructions, harmless_instructions, ablation_targets, addition_targets, retain_targets):
    ablation_prompts = []
    addition_prompts = []
    ablation_labels = []
    addition_labels = []
    retain_prompts = []
    for harmful_instruction, harmless_instruction, ablation_target, addition_target, retain_target in zip(harmful_instructions, harmless_instructions, ablation_targets, addition_targets, retain_targets):
        ablation_text = harmful_instruction + ablation_target
        addition_text = harmless_instruction + addition_target
        retain_text = harmless_instruction + retain_target
        ablation_prompts.append(ablation_text)
        addition_prompts.append(addition_text)
        retain_prompts.append(retain_text)

        # Tokenize without padding
        ablation_tokens = model.tokenizer.encode(ablation_text, add_special_tokens=True, return_tensors='pt')[0]
        addition_tokens = model.tokenizer.encode(addition_text, add_special_tokens=True, return_tensors='pt')[0]
        
        ablation_label = ablation_tokens[1:].clone()
        addition_label = addition_tokens[1:].clone()
        
        # Get the length of the instruction
        harmful_instruction_length = len(model.tokenizer.encode(harmful_instruction, add_special_tokens=True)) - 1
        harmless_instruction_length = len(model.tokenizer.encode(harmless_instruction, add_special_tokens=True)) - 1
        
        # Set labels corresponding to the instruction tokens to -100
        ablation_label[:harmful_instruction_length] = -100
        addition_label[:harmless_instruction_length] = -100

        ablation_labels.append(ablation_label)
        addition_labels.append(addition_label)
    return ablation_prompts, addition_prompts, retain_prompts, ablation_labels, addition_labels

ablation_train_prompts, addition_train_prompts, retain_train_prompts, ablation_train_labels, addition_train_labels = build_prompts_and_labels(model, harmful_train_instructions, harmless_train_instructions, ablation_train_targets, addition_train_targets, retain_train_targets)

# %%
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, harmful_prompts, harmless_prompts, ablation_prompts, ablation_targets, ablation_labels, addition_prompts, addition_targets, addition_labels, retain_prompts, retain_targets):
        self.harmful_prompts = harmful_prompts
        self.harmless_prompts = harmless_prompts
        self.ablation_prompts = ablation_prompts
        self.ablation_targets = ablation_targets
        self.ablation_labels = ablation_labels
        self.addition_prompts = addition_prompts
        self.addition_targets = addition_targets
        self.addition_labels = addition_labels
        self.retain_prompts = retain_prompts
        self.retain_targets = retain_targets

    def __len__(self):
        return len(self.harmful_prompts)
    
    def __getitem__(self, idx):
        return {
            'harmful_prompt': self.harmful_prompts[idx],
            'harmless_prompt': self.harmless_prompts[idx],
            'ablation_prompt': self.ablation_prompts[idx],
            'ablation_target': self.ablation_targets[idx],
            'ablation_labels': self.ablation_labels[idx],
            'addition_prompt': self.addition_prompts[idx],
            'addition_target': self.addition_targets[idx],
            'addition_labels': self.addition_labels[idx],
            'retain_prompt': self.retain_prompts[idx],
            'retain_target': self.retain_targets[idx],
        }

train_dataset = CustomDataset(harmful_train_instructions, harmless_train_instructions, ablation_train_prompts, ablation_train_targets, ablation_train_labels, addition_train_prompts, addition_train_targets, addition_train_labels, retain_train_prompts, retain_train_targets)
print(len(train_dataset))
d = train_dataset[0]
for item in d.items():
    print(item)

# %%
print(f"Length of harmful_prompts: {len(train_dataset.harmful_prompts)}")
print(f"Length of harmless_prompts: {len(train_dataset.harmless_prompts)}")
print(f"Length of ablation_prompts: {len(train_dataset.ablation_prompts)}")
print(f"Length of ablation_targets: {len(train_dataset.ablation_targets)}")
print(f"Length of ablation_labels: {len(train_dataset.ablation_labels)}")
print(f"Length of addition_prompts: {len(train_dataset.addition_prompts)}")
print(f"Length of addition_targets: {len(train_dataset.addition_targets)}")
print(f"Length of addition_labels: {len(train_dataset.addition_labels)}")
print(f"Length of retain_prompts: {len(train_dataset.retain_prompts)}")
print(f"Length of retain_targets: {len(train_dataset.retain_targets)}")

# %%
def custom_collate(batch):
    # Separate the different items in the batch
    return {
        'harmful_prompt': [item['harmful_prompt'] for item in batch],
        'harmless_prompt': [item['harmless_prompt'] for item in batch],
        'ablation_prompt': [item['ablation_prompt'] for item in batch],
        'ablation_target': [item['ablation_target'] for item in batch],
        'ablation_labels': torch.stack([item['ablation_labels'] for item in batch]),
        'addition_prompt': [item['addition_prompt'] for item in batch],
        'addition_target': [item['addition_target'] for item in batch], 
        'addition_labels': torch.stack([item['addition_labels'] for item in batch]),
        'retain_prompt': [item['retain_prompt'] for item in batch],
        'retain_target': [item['retain_target'] for item in batch],
    }
# %%
from nnsight.envoy import Envoy #
import einops

def projection_einops(activation, direction):
    proj = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction
    )
    return proj

def dotproduct_einops(activation, direction):
    dot_product = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
    )
    return dot_product

def sample_hypersphere_gaussian(batch_size, dim):
    # Sample from standard normal distribution
    samples = torch.randn(batch_size, dim, dtype=torch.float32, device=model.device).abs()
    # Normalize to unit length
    samples = samples / torch.norm(samples, dim=1, keepdim=True)
    return samples

def sample_prob_vectors(batch_size, dim):
    samples = torch.exp(torch.randn(batch_size, dim, dtype=torch.float32, device=model.device))
    samples = samples / samples.sum(dim=1, keepdim=True)
    return samples

def compute_ce_loss(logits, labels):
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    # Always pad labels with ignore tokens (-100) to match logits shape
    padding = torch.full((logits.size(0),), -100, device=labels.device)
    padding[-labels.size(0):] = labels
    return torch.nn.functional.cross_entropy(logits, padding, ignore_index=-100)

# def compute_ce_loss(logits, labels):
#     logits = logits.view(-1, logits.size(-1))
#     labels = labels.view(-1)
#     return torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)

def kl_div_fn(logits_a, logits_b, reduction='batchmean'):
    # Compute log-probabilities for the first distribution
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)
    
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits_a, dim=-1), 
        torch.nn.functional.softmax(logits_b, dim=-1),
        reduction=reduction
    )

def get_cosine_sims_for_vector(model, dot_vector, last_token=True):
    """Calculate cosine similarities between activations and provided vector across layers.
    
    Args:
        model: The language model
        prompt: Input prompt to get activations for
        dot_vector: Vector to compute cosine similarity against
        
    Returns:
        torch.Tensor: Tensor of cosine similarities across layers
    """
    cosine_sims = []
    for layer in model.model.layers:
        if last_token:
            cosine_sim = torch.nn.functional.cosine_similarity(layer.input[0, -1], dot_vector, dim=-1).save()
        else:
            cosine_sim = torch.nn.functional.cosine_similarity(layer.input[0, :], dot_vector, dim=-1).save()
        cosine_sims.append(cosine_sim)
    return torch.stack(cosine_sims)

def get_projections_for_vector(model, dot_vector):
    """Calculate projections of activations onto provided vector across layers.
    
    Args:
        model: The language model
        prompt: Input prompt to get activations for
        dot_vector: Vector to compute projections against
        
    Returns:
        torch.Tensor: Tensor of projections across layers
    """
    projections = []
    for layer in model.model.layers:
        norm_dot_vector = (dot_vector / dot_vector.norm()).to(model.dtype)
        projection = projection_einops(layer.input[0, :], norm_dot_vector)
        projections.append(projection)
    return torch.stack(projections)


def clip_grad_norm(grad, max_norm):
    total_norm = grad.norm()
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    return grad * clip_coef_clamped

class RobustAblation(nn.Module):
    def __init__(self, module: Envoy, dim: int, orthogonal_vectors: list[Float[Tensor, "d_model"]], add_layer_idx: int, alpha: float, train_alpha: bool, train_layer_weights: bool, init_vector: Float[Tensor, "d_model"] | None = None) -> None:
        super(RobustAblation, self).__init__()
        self.module = module
        self.fn_vector = torch.nn.Parameter(
            init_vector.to(torch.float32) if init_vector is not None else torch.randn(dim, dtype=torch.float32),
            requires_grad=True
        ).save()
        if train_layer_weights:
            alphas = [d.norm() for d in refusal_directions[best_token]]
            self.alpha = torch.nn.Parameter(torch.tensor(alphas).to(torch.float32), requires_grad=train_alpha).save()
        else:
            self.alpha = torch.nn.Parameter(torch.tensor(float(alpha.item())).to(torch.float32), requires_grad=train_alpha).save()
        nnsight.log("alpha", self.alpha)
        self.orthogonal_vectors = [(o / o.norm()).to(torch.float32).cpu() for o in orthogonal_vectors]
        self.add_layer_idx = add_layer_idx
        self.train_alpha = train_alpha
        self.train_layer_weights = train_layer_weights

        n_layers = len(module.layers)
        layer_weights = torch.ones(n_layers, dtype=torch.float32) / n_layers
        self.layer_weights = torch.nn.Parameter(layer_weights, requires_grad=train_layer_weights).save()

        self.orthogonalize()

    def __call__(self, direction):
        direction = direction / direction.norm()
        direction = direction.to(model.dtype)
        for layer in self.module.layers:
            self.ablate_input(layer, direction)
            self.ablate_output(layer.self_attn, direction, 3)
            self.ablate_output(layer.mlp, direction, 1)
    
    def ablate_output(self, layer, direction, tuple_length=1):
        if tuple_length > 1:
            activation = layer.output[0][:]
        else:
            activation = layer.output
        projection = projection_einops(activation, direction)
        new_activation = activation - projection
        if tuple_length == 2:
            layer.output = (new_activation, layer.output[1])
        elif tuple_length == 3:
            layer.output = (new_activation, layer.output[1], layer.output[2])
        elif tuple_length == 1:
            layer.output = new_activation
    
    def ablate_input(self, layer, direction):
        projection = projection_einops(layer.input, direction)
        new_activation = layer.input - projection
        layer.input = new_activation

    def add(self, direction, temperature):
        direction = direction / direction.norm()
        direction = direction.to(model.dtype)
        if self.train_layer_weights:
            layer_probs = torch.nn.functional.gumbel_softmax(self.layer_weights, tau=temperature, hard=False, dim=0)
            for idx, layer in enumerate(self.module.layers):
                layer.input += self.alpha[idx] * direction * layer_probs[idx]
            return layer_probs
        else:
            self.module.layers[self.add_layer_idx].input += self.alpha * direction

    def normalize(self):
        with torch.no_grad():
            self.fn_vector.data = self.fn_vector.data / self.fn_vector.data.norm()

    def orthogonalize(self):
        if self.orthogonal_vectors:
            self.fn_vector.data = nnsight.apply(self._orthogonalize, self.fn_vector)

    def _orthogonalize(self, vector):
        with torch.no_grad():
            v = vector.data.clone().cpu()
            
            # Stack your vectors as rows in a matrix A
            A = torch.stack([vec.flatten().to(torch.float32) for vec in self.orthogonal_vectors])
            # Compute projection matrix P = A^T(AA^T)^-1A
            # The nullspace projector is then I - P
            AAT = A @ A.t()
            AAT_inv = torch.inverse(AAT)
            P = A.t() @ AAT_inv @ A
            I = torch.eye(P.shape[0], device=P.device)
            
            # Project onto nullspace (orthogonal complement)
            v_flat = v.flatten()
            v_ortho = (I - P) @ v_flat
            
            # Reshape back to original shape and normalize
            v_ortho = v_ortho.reshape(v.shape)
            v_ortho = v_ortho / torch.norm(v_ortho)
                
            return v_ortho.to(vector.dtype).to(vector.device)

def train_robust_ablation_vector(model, train_dataset, current_batch_size, target_batch_size, epochs=1, orthogonal_vectors: list[Float[Tensor, "d_model"]]=[], lr=1e-3, use_ablation_loss=True, ablation_lambda=1, use_addition_loss=True, alpha=alpha, train_alpha=True, train_layer_weights=False, entropy_lambda=0.1, addition_lambda=1., use_repind_loss=False, repind_layers=[10], repind_lambda=1, independent_vectors=None, symmetric_repind=False, use_retain_loss=False, retain_lambda=1, verbose=False, patience=9999, n_lr_reduce=2, init_vector=None):

    train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate)

    if use_repind_loss:
        independent_vectors = [ind.detach().clone().cuda().to(model.dtype) for ind in independent_vectors]
    if init_vector is not None:
        init_vector = init_vector.detach().clone().cuda().to(model.dtype)

    operation = RobustAblation(model.model, model.config.hidden_size, orthogonal_vectors, best_layer, alpha, train_alpha, train_layer_weights, init_vector=init_vector)

    parameters = [{"params": operation.fn_vector, "lr": lr}]
    if train_alpha:
        parameters.append({"params": operation.alpha, "lr": 1})
    if train_layer_weights:
        parameters.append({"params": operation.layer_weights, "lr": lr})
    optimizer = torch.optim.AdamW(parameters, betas=(.9,.98), weight_decay=0.0, amsgrad=True)

    # log optimizer parameters
    accumulation_steps = target_batch_size // current_batch_size
    if verbose:
        print("Accumulation steps", accumulation_steps)

    temperature = 1.0
    vectors = []
    train_losses = []
    lowest_training_loss = float('inf')
    patience_counter = 0
    lr_reduce_counter = 0
    stopped = False
    
    if verbose:
        print("Starting training")

    step_counter = 0
    batch_ablation_loss = 0
    batch_addition_loss = 0
    batch_entropy_loss = 0
    batch_repind_loss = 0
    batch_retain_loss = 0
    batch_refusal_score = 0
    batch_induce_score = 0
    batch_gradients = []
    batch_alpha_gradients = []
    batch_layer_weights_gradients = []

    for epoch in range(epochs):
        if verbose:
            print(f'Epoch {epoch}')
        for batch in train_dataloader:
            ablation_prompt = batch['ablation_prompt']
            ablation_labels = batch['ablation_labels']
            addition_prompt = batch['addition_prompt']
            addition_labels = batch['addition_labels']
            retain_prompt = batch['retain_prompt']
            harmful_prompt = batch['harmful_prompt']
            harmless_prompt = batch['harmless_prompt']

            if use_ablation_loss:
                with model.trace() as tracer:
                    with tracer.invoke(ablation_prompt):
                        operation(operation.fn_vector)
                        logits = model.lm_head.output[:, :-1]
                        ablation_loss = compute_ce_loss(logits, ablation_labels)
                        log = ablation_loss.detach().item()
                    batch_ablation_loss += log
                    (ablation_lambda * ablation_loss).backward()
            
            if use_repind_loss:
                with model.trace() as tracer:
                    repind_losses = []
                    for independent_vector in independent_vectors:
                        with tracer.invoke(harmful_prompt):
                            target_independent_cosine_sims = get_cosine_sims_for_vector(model, independent_vector)
                            target_fn_vector_cosine_sims = get_cosine_sims_for_vector(model, operation.fn_vector)
                    
                        with tracer.invoke(harmful_prompt):
                            operation(operation.fn_vector)
                            current_independent_cosine_sims = get_cosine_sims_for_vector(model, independent_vector)
                            repind_losses.append((current_independent_cosine_sims - target_independent_cosine_sims)[repind_layers].square().mean())

                        if symmetric_repind:
                            with tracer.invoke(harmful_prompt):
                                operation(independent_vector)
                                current_fn_vector_cosine_sims = get_cosine_sims_for_vector(model, operation.fn_vector)
                                repind_losses.append((current_fn_vector_cosine_sims - target_fn_vector_cosine_sims)[repind_layers].square().mean())
                    
                    repind_loss = torch.stack(repind_losses).sum()
                    repind_loss = repind_loss / accumulation_steps
                    batch_repind_loss.update(batch_repind_loss + repind_loss.detach().item())
                    (repind_lambda * repind_loss).backward()
            
            if use_addition_loss:
                with model.trace() as tracer:
                    with tracer.invoke(addition_prompt) as _:
                        operation.add(operation.fn_vector)
                        logits = model.lm_head.output[:, :-1]
                        addition_loss = compute_ce_loss(logits, addition_labels)
                        log = addition_loss.detach().item().save()
                    batch_addition_loss += log
                    (addition_lambda * addition_loss).backward()

            if use_retain_loss:
                with model.trace() as tracer:
                    with tracer.invoke(retain_prompt):   
                        baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                    with tracer.invoke(retain_prompt):
                        operation(operation.fn_vector)
                        retain_logits = model.lm_head.output[:, -num_target_tokens:]
                        retain_loss = kl_div_fn(baseline_retain_logits, retain_logits).mean()
                        log = retain_loss.detach().item().save()
                    batch_retain_loss += log
                    (retain_lambda * retain_loss).backward()
            
            with torch.no_grad():
                with model.trace() as tracer:
                    with tracer.invoke(harmful_prompt) as _:
                        operation(operation.fn_vector)
                        last_token_logits = model.lm_head.output[:, -1]
                        refusal_score = refusal_score_fn(last_token_logits, refusal_tokens).detach().item().save()
                    batch_refusal_score.append(refusal_score)
                with model.trace() as tracer:
                    with tracer.invoke(harmless_prompt) as _:
                        operation.add(operation.fn_vector)
                        last_token_logits = model.lm_head.output[:, -1]
                        induce_score = refusal_score_fn(last_token_logits, refusal_tokens)
                        batch_induce_score.update(batch_induce_score + induce_score.detach().item())

                step_counter += 1
                if step_counter % accumulation_steps == 0:
                    grad_sum = nnsight.apply(sum, batch_gradients)
                    if train_alpha:
                        alpha_grad_sum = nnsight.apply(sum, batch_alpha_gradients)
                    if train_layer_weights:
                        layer_weights_grad_sum = nnsight.apply(sum, batch_layer_weights_gradients)
                    grad_sum = nnsight.apply(lambda x, y: x - projection_einops(x, y / y.norm()), grad_sum, operation.fn_vector) # project gradient to tangent of sphere
                    grad_norm = grad_sum.norm().item()
                    operation.fn_vector.grad = grad_sum
                    if train_alpha:
                        operation.alpha.grad = alpha_grad_sum
                    if train_layer_weights:
                        operation.layer_weights.grad = layer_weights_grad_sum
                    optimizer.step()
                    optimizer.zero_grad()

                    num_batches_per_epoch = len(train_dataset) / target_batch_size / 2
                    target_final_temp = 0.01
                    temperature_decay = np.exp(np.log(target_final_temp) / num_batches_per_epoch)

                    temperature.update(temperature * temperature_decay)

                    if orthogonal_vectors:
                        operation.orthogonalize()
                    operation.normalize()
                    train_loss = batch_ablation_loss + batch_addition_loss + batch_repind_loss + batch_retain_loss
                    train_losses.append(train_loss)
                    
                    vectors.append(operation.fn_vector.detach().data.clone())
                    
                    nnsight.apply(wandb.log, {
                        "train/total_loss": train_loss,
                        "train/ablation_loss": batch_ablation_loss,
                        "train/addition_loss": batch_addition_loss,
                        "train/entropy_loss": batch_entropy_loss,
                        "train/repind_loss": batch_repind_loss,
                        "train/retain_loss": batch_retain_loss,
                        "train/refusal_score": batch_refusal_score,
                        "train/induce_score": batch_induce_score,
                        "train/grad_norm": grad_norm,
                        "train/temperature": temperature
                    }, step=step_counter)
                    nnsight.log("Step", step_counter, "train/refusal_score", batch_refusal_score, "train/induce_score", batch_induce_score)
    
                    with train_iterator.cond(train_loss >= lowest_training_loss):
                        patience_counter.update(patience_counter + 1)
                    with train_iterator.cond(train_loss < lowest_training_loss):
                        lowest_training_loss.update(train_loss)
                        patience_counter.update(0)
                    with train_iterator.cond(patience_counter >= patience):
                        with train_iterator.cond(lr_reduce_counter >= n_lr_reduce):
                            if verbose:
                                nnsight.log(f'Stopping')
                            stopped.update(True)
                            train_iterator.exit()
                        with train_iterator.cond(lr_reduce_counter < n_lr_reduce):
                            lr_reduce_counter.update(lr_reduce_counter + 1)
                            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                            nnsight.log("Reducing lr to", optimizer.param_groups[0]['lr'])
                            patience_counter.update(0)
                    
                    if train_alpha:
                        nnsight.log("Alpha", operation.alpha.data.item())
                    if train_layer_weights:
                        nnsight.log("Layer weights", operation.layer_weights.data)
                        nnsight.log("layer weights max idx", operation.layer_weights.data.argmax())
                    batch_ablation_loss.update(0)
                    batch_addition_loss.update(0)
                    batch_entropy_loss.update(0)
                    batch_repind_loss.update(0)
                    batch_retain_loss.update(0)
                    batch_refusal_score.update(0)
                    batch_induce_score.update(0)
                    batch_gradients.update([])
                    batch_alpha_gradients.update([])
                    batch_layer_weights_gradients.update([])

        with epoch_iterator.cond(stopped == True):
            epoch_iterator.exit()

    save_vectors = vectors.value
    run_id = wandb.run.id
    lowest_loss_index = torch.argmin(torch.tensor(train_losses.value)).item()
    lowest_loss_vector = vectors.value[lowest_loss_index]
    artifact = wandb.Artifact(f'trained_vectors_run_{run_id}', type='vector')
    with artifact.new_file('vector.pt', mode='wb') as f:
        torch.save(save_vectors, f)
    with artifact.new_file('lowest_loss_vector.pt', mode='wb') as f:
        torch.save(lowest_loss_vector, f)
    wandb.log_artifact(artifact)
    
    return {"vectors": save_vectors, "lowest_loss_vector": lowest_loss_vector}

class SubspaceAblationGreedy(nn.Module):
    def __init__(self, module: Envoy, dim: int, n_vectors: int, basis_vectors: torch.Tensor | None = None) -> None:
        super(SubspaceAblationGreedy, self).__init__()
        self.module = module
        self.n_vectors = n_vectors
        self.fixed_vectors = []
        if basis_vectors:
            for basis_vector in basis_vectors:
                norm_basis_vector = basis_vector / basis_vector.norm()
                self.fixed_vectors.append(torch.nn.Parameter(norm_basis_vector.to(torch.float32).cuda(), requires_grad=False))
        self.trainable_vector = torch.nn.Parameter(torch.randn(dim, dtype=torch.float32).cuda(), requires_grad=True)
        self.orthogonalize()
    
    def __call__(self, direction):
        normalized_direction = direction / direction.norm()
        normalized_direction = normalized_direction.to(model.dtype)
        for layer in self.module.layers:
            self.ablate_input(layer, normalized_direction)
            self.ablate_output(layer.self_attn, normalized_direction, 3)
            self.ablate_output(layer.mlp, normalized_direction, 1)
    
    def ablate_output(self, layer, direction, tuple_length=1):
        if tuple_length > 1:
            activation = layer.output[0][:]
        else:
            activation = layer.output
        projection = projection_einops(activation, direction)
        new_activation = activation - projection
        if tuple_length == 2:
            layer.output = (new_activation, layer.output[1])
        elif tuple_length == 3:
            layer.output = (new_activation, layer.output[1], layer.output[2])
        elif tuple_length == 1:
            layer.output = new_activation
    
    def ablate_input(self, layer, direction):
        projection = projection_einops(layer.input, direction)
        new_activation = layer.input - projection
        layer.input = new_activation

    def add(self, direction, alpha, layer_idx):
        direction = direction / direction.norm()
        direction = direction.to(model.dtype)
        self.module.layers[layer_idx].input += alpha * direction
    
    def transform(self, sample):
        if self.n_vectors > 1:
            fn_vectors = torch.stack(self.fixed_vectors + [self.trainable_vector], dim=0)
        else:
            fn_vectors = self.trainable_vector.unsqueeze(0)
        transformed_sample = torch.matmul(sample, fn_vectors).to(model.dtype)
        transformed_sample = transformed_sample / torch.norm(transformed_sample)
        return transformed_sample

    def parameters(self):
        return [self.trainable_vector]

    def orthogonalize(self):
        with torch.no_grad():
            for fixed_vector in self.fixed_vectors:
                # In-place subtraction of projection
                self.trainable_vector.data.sub_(projection_einops(self.trainable_vector.data, fixed_vector.data))
            # In-place normalization
            self.trainable_vector.data.div_(self.trainable_vector.data.norm())

def train_ablation_subspace_greedy(model, train_dataset, current_batch_size, target_batch_size, epochs=10, lr=1e-3, subspace_dim=2, n_sample=1, fixed_samples=8, sampling_method="hypersphere", optimize_basis=False, use_ablation_loss=True, ablation_lambda=1, use_addition_loss=True, alpha=alpha, addition_lambda=1, use_retain_loss=False, retain_lambda=1, patience=5, verbose=False, basis_vectors=None, n_lr_reduce=0):

    train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate)

    operation = SubspaceAblationGreedy(model.model, model.config.hidden_size, subspace_dim, basis_vectors=basis_vectors)

    optimizer = torch.optim.AdamW(operation.parameters(), lr=lr, betas=(.9,.98), weight_decay=0.0, amsgrad=True)
    print("Subspace dim", subspace_dim)
    if subspace_dim == 1:
        n_sample = 0

    accumulation_steps = target_batch_size // current_batch_size
    if verbose:
        print("Accumulation steps", accumulation_steps)
    vectors = []
    train_losses = []
    stopped = False
    lowest_training_loss = float('inf')
    refusal_scores = []
    patience_counter = 0
    lr_reduce_counter = 0

    if verbose:
        print("Starting training")

    step_counter = 0
    batch_sample_ablation_loss = 0.0
    batch_sample_addition_loss = 0.0
    batch_sample_retain_loss = 0.0
    batch_basis_ablation_loss = 0.0
    batch_basis_addition_loss = 0.0
    batch_basis_retain_loss = 0.0

    batch_sample_refusal_scores = []
    batch_sample_induce_scores = []
    batch_refusal_score = 0.
    batch_induce_score = 0.

    if sampling_method == "hypersphere":
        fixed_sample_vectors = sample_hypersphere_gaussian(fixed_samples, subspace_dim)
    elif sampling_method == "interpolation":
        fixed_sample_vectors = sample_prob_vectors(fixed_samples, subspace_dim)
    fixed_sample_vectors = [fixed_sample_vectors[i] for i in range(fixed_samples)]

    for epoch in range(epochs):
        if verbose:
            print('Epoch', epoch)
        for _, batch in enumerate(train_dataloader):
            ablation_prompt = batch['ablation_prompt']
            ablation_labels = batch['ablation_labels']
            addition_prompt = batch['addition_prompt']
            addition_labels = batch['addition_labels']
            retain_prompt = batch['retain_prompt']
            harmful_prompt = batch['harmful_prompt']
            harmless_prompt = batch['harmless_prompt']

            if n_sample > 0:
                if sampling_method == "hypersphere":
                    sample_vectors = sample_hypersphere_gaussian(n_sample, subspace_dim)
                elif sampling_method == "interpolation":
                    sample_vectors = sample_prob_vectors(n_sample, subspace_dim)
                sample_vectors = [sample_vectors[i] for i in range(n_sample)]
                for sample_vector in sample_vectors:
                    if use_ablation_loss:
                        with model.trace() as tracer:
                            with tracer.invoke(ablation_prompt):
                                direction = operation.transform(sample_vector)
                                operation(direction)
                                logits = model.lm_head.output[:, :-1]
                                sample_ablation_loss = compute_ce_loss(logits, ablation_labels).save()
                            (ablation_lambda * sample_ablation_loss / n_sample).backward()
                    batch_sample_ablation_loss += sample_ablation_loss.value
                    if use_addition_loss:
                        with model.trace() as tracer:
                            with tracer.invoke(addition_prompt):
                                direction = operation.transform(sample_vector)
                                operation.add(direction, alpha, add_layer)
                                logits = model.lm_head.output[:, :-1]
                                sample_addition_loss = compute_ce_loss(logits, addition_labels).save()
                            (addition_lambda * sample_addition_loss / n_sample).backward()
                        batch_sample_addition_loss += sample_addition_loss.value
                    if use_retain_loss:
                        with model.trace() as tracer:
                            with tracer.invoke(retain_prompt):
                                baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                            with tracer.invoke(retain_prompt):
                                direction = operation.transform(sample_vector)
                                operation(direction)
                                sample_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                                sample_retain_loss = kl_div_fn(baseline_retain_logits, sample_retain_logits).mean().save()
                            (retain_lambda * sample_retain_loss / n_sample).backward()
                        batch_sample_retain_loss += sample_retain_loss.value
                
            if optimize_basis:
                if use_ablation_loss:
                    with model.trace() as tracer:
                        with tracer.invoke(ablation_prompt):
                            operation(operation.trainable_vector)
                            logits = model.lm_head.output[:, :-1]
                            basis_ablation_loss = compute_ce_loss(logits, ablation_labels).save()
                        (ablation_lambda * basis_ablation_loss).backward()
                    batch_basis_ablation_loss += basis_ablation_loss.value

                if use_addition_loss:
                    with model.trace() as tracer:
                        with tracer.invoke(addition_prompt):
                            operation.add(operation.trainable_vector, alpha, add_layer)
                            logits = model.lm_head.output[:, :-1]
                            basis_addition_loss = compute_ce_loss(logits, addition_labels).save()
                        (addition_lambda * basis_addition_loss).backward()
                    batch_basis_addition_loss += basis_addition_loss.value

                if use_retain_loss:
                    with model.trace() as tracer:
                        with tracer.invoke(retain_prompt):
                            baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                        with tracer.invoke(retain_prompt):
                            operation(operation.trainable_vector)
                            retain_logits = model.lm_head.output[:, -num_target_tokens:]
                            basis_retain_loss = kl_div_fn(baseline_retain_logits, retain_logits).mean().save()
                        (retain_lambda * basis_retain_loss).backward()
                    batch_basis_retain_loss += basis_retain_loss.value
            
            with torch.no_grad():
                with model.trace() as tracer:
                    with tracer.invoke(harmful_prompt):
                        operation(operation.trainable_vector)
                        last_token_logits = model.lm_head.output[:, -1]
                        refusal_score = refusal_score_fn(last_token_logits, refusal_tokens).to(model.dtype).save()
                batch_refusal_score += refusal_score.value
                with model.trace() as tracer:
                    with tracer.invoke(harmless_prompt):
                        operation.add(operation.trainable_vector, alpha, add_layer)
                        last_token_logits = model.lm_head.output[:, -1]
                        induce_score = refusal_score_fn(last_token_logits, refusal_tokens).to(model.dtype).save()
                batch_induce_score += induce_score.value
                if n_sample > 0:
                    with model.trace() as tracer:
                        for fixed_sample_vector in fixed_sample_vectors:
                            with tracer.invoke(harmful_prompt):
                                direction = operation.transform(fixed_sample_vector)
                                operation(direction)
                                sample_last_token_logits = model.lm_head.output[:, -1]
                                sample_refusal_score = refusal_score_fn(sample_last_token_logits, refusal_tokens).to(model.dtype).save()
                                batch_sample_refusal_scores.append(sample_refusal_score)
                    with model.trace() as tracer:
                        for fixed_sample_vector in fixed_sample_vectors:
                            with tracer.invoke(harmless_prompt):
                                direction = operation.transform(fixed_sample_vector)
                                operation.add(direction, alpha, add_layer)
                                sample_last_token_logits = model.lm_head.output[:, -1]
                                sample_induce_score = refusal_score_fn(sample_last_token_logits, refusal_tokens).to(model.dtype).save()
                                batch_sample_induce_scores.append(sample_induce_score)

                step_counter += 1
                if step_counter % accumulation_steps == 0:
                    operation.trainable_vector.grad.sub_(projection_einops(operation.trainable_vector.grad, operation.trainable_vector.data))
                    operation.trainable_vector.grad.div_(accumulation_steps)
                    torch.nn.utils.clip_grad_norm_(operation.parameters(), 10.0)
                    grad_norm = operation.trainable_vector.grad.norm().item()
                    optimizer.step()
                    optimizer.zero_grad()
                    operation.orthogonalize()

                    batch_sample_ablation_loss /= accumulation_steps
                    batch_sample_addition_loss /= accumulation_steps
                    batch_sample_retain_loss /= accumulation_steps
                    batch_basis_ablation_loss /= accumulation_steps
                    batch_basis_addition_loss /= accumulation_steps
                    batch_basis_retain_loss /= accumulation_steps

                    train_loss = batch_sample_ablation_loss + batch_sample_addition_loss + batch_sample_retain_loss + batch_basis_ablation_loss + batch_basis_addition_loss + batch_basis_retain_loss
                    train_losses.append(train_loss)

                    batch_refusal_score /= accumulation_steps
                    batch_induce_score /= accumulation_steps

                    vectors.append(operation.trainable_vector.detach().data.clone())
                    refusal_scores.append(batch_refusal_score)

                    batch_sample_refusal_scores = [s.value for s in batch_sample_refusal_scores]
                    batch_sample_induce_scores = [s.value for s in batch_sample_induce_scores]

                    sample_vector_refusal_scores = [torch.mean(torch.tensor(batch_sample_refusal_scores[i::fixed_samples])).item() for i in range(fixed_samples)]
                    min_sample_refusal_score = min(sample_vector_refusal_scores)
                    max_sample_refusal_score = max(sample_vector_refusal_scores)
                    mean_sample_refusal_score = torch.mean(torch.tensor(sample_vector_refusal_scores)).item()
                    std_sample_refusal_score = torch.std(torch.tensor(sample_vector_refusal_scores)).item()

                    sample_vector_induce_scores = [torch.mean(torch.tensor(batch_sample_induce_scores[i::fixed_samples])).item() for i in range(fixed_samples)]
                    min_sample_induce_score = min(sample_vector_induce_scores)
                    max_sample_induce_score = max(sample_vector_induce_scores)
                    mean_sample_induce_score = torch.mean(torch.tensor(sample_vector_induce_scores)).item()
                    std_sample_induce_score = torch.std(torch.tensor(sample_vector_induce_scores)).item()

                    wandb.log({
                        "train/total_loss": train_loss.item(),
                        "train/sample_ablation_loss": batch_sample_ablation_loss.item() if n_sample > 0 else 0,
                        "train/sample_addition_loss": batch_sample_addition_loss.item() if n_sample > 0 else 0,
                        "train/sample_retain_loss": batch_sample_retain_loss.item() if n_sample > 0 else 0,
                        "train/basis_ablation_loss": batch_basis_ablation_loss.item(),
                        "train/basis_addition_loss": batch_basis_addition_loss.item(),
                        "train/basis_retain_loss": batch_basis_retain_loss.item(),
                        "train/basis_refusal_score": batch_refusal_score.item(),
                        "train/basis_induce_score": batch_induce_score.item(),
                        "train/min_sample_refusal_score": min_sample_refusal_score if n_sample > 0 else 0,
                        "train/max_sample_refusal_score": max_sample_refusal_score if n_sample > 0 else 0,
                        "train/mean_sample_refusal_score": mean_sample_refusal_score if n_sample > 0 else 0,
                        "train/std_sample_refusal_score": std_sample_refusal_score if n_sample > 0 else 0,
                        "train/min_sample_induce_score": min_sample_induce_score if n_sample > 0 else 0,
                        "train/max_sample_induce_score": max_sample_induce_score if n_sample > 0 else 0,
                        "train/mean_sample_induce_score": mean_sample_induce_score if n_sample > 0 else 0,
                        "train/std_sample_induce_score": std_sample_induce_score if n_sample > 0 else 0,
                        "train/grad_norm": grad_norm
                    }, step=step_counter)
                    print("Step", step_counter, "train/basis_vector_refusal_score", batch_refusal_score.item())
                    if n_sample > 0:
                        print("train/mean_sample_refusal_scores", mean_sample_refusal_score)
                    if train_loss >= lowest_training_loss:
                        patience_counter += 1
                    else:
                        lowest_training_loss = train_loss
                        patience_counter = 0
                    if patience_counter >= patience:
                        if lr_reduce_counter >= n_lr_reduce:
                            if verbose:
                                print(f'Stopping')
                            stopped = True
                            break
                        lr_reduce_counter += 1
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                        print("Reducing lr to", optimizer.param_groups[0]['lr'])
                        patience_counter = 0

                    batch_sample_ablation_loss = 0.
                    batch_sample_addition_loss = 0.
                    batch_sample_retain_loss = 0.
                    batch_basis_ablation_loss = 0.
                    batch_basis_addition_loss = 0.
                    batch_basis_retain_loss = 0.
                    batch_sample_refusal_scores = []
                    batch_sample_induce_scores = []
                    batch_refusal_score = 0.
                    batch_induce_score = 0.
                    torch.cuda.empty_cache()

        if stopped:
            break

    save_vectors = vectors
    lowest_loss_index = torch.argmin(torch.tensor(train_losses)).item()
    lowest_loss_vector = save_vectors[lowest_loss_index]
    low_loss_refusal_score = refusal_scores[lowest_loss_index]
    run_id = wandb.run.id
    artifact = wandb.Artifact(f'trained_subspaces_run_{run_id}', type='vector')
    with artifact.new_file(f'subspaces.pt', mode='wb') as f:
        torch.save(save_vectors, f)
    with artifact.new_file(f'lowest_loss_vector.pt', mode='wb') as f:
        torch.save(torch.stack(basis_vectors + [lowest_loss_vector]), f)
    wandb.log_artifact(artifact)
    
    return {"vectors": vectors, "lowest_loss": lowest_training_loss, "refusal_scores": refusal_scores, "train_losses": train_losses, "lowest_loss_vector": lowest_loss_vector, "lowest_loss_refusal_score": low_loss_refusal_score}

class SubspaceAblationJoint(nn.Module):
    def __init__(self, module: Envoy, dim: int, n_vectors: int, init_vectors: torch.Tensor | None = None, orthogonal_vectors: torch.Tensor | None = None) -> None:
        super(SubspaceAblationJoint, self).__init__()
        self.module = module
        self.n_vectors = n_vectors
        self.fn_vectors = [torch.nn.Parameter(torch.randn(dim, dtype=torch.float32).cuda(), requires_grad=True) for _ in range(n_vectors)]
        if init_vectors is not None:
            for i, init_vector in enumerate(init_vectors):
                init_vector = init_vector / init_vector.norm()
                self.fn_vectors[i].data = init_vector.detach().clone().cuda().to(torch.float32)
        self.orthogonal_vectors = [(o / o.norm()).to(torch.float32).cpu() for o in orthogonal_vectors]
        self.orthogonalize()
    
    def __call__(self, direction):
        normalized_direction = direction / direction.norm()
        normalized_direction = normalized_direction.to(model.dtype)
        for layer in self.module.layers:
            self.ablate_input(layer, normalized_direction)
            self.ablate_output(layer.self_attn, normalized_direction, 3)
            self.ablate_output(layer.mlp, normalized_direction, 1)
    
    def ablate_output(self, layer, direction, tuple_length=1):
        if tuple_length > 1:
            activation = layer.output[0][:]
        else:
            activation = layer.output
        projection = projection_einops(activation, direction)
        new_activation = activation - projection
        if tuple_length == 2:
            layer.output = (new_activation, layer.output[1])
        elif tuple_length == 3:
            layer.output = (new_activation, layer.output[1], layer.output[2])
        elif tuple_length == 1:
            layer.output = new_activation
    
    def ablate_input(self, layer, direction):
        projection = projection_einops(layer.input, direction)
        new_activation = layer.input - projection
        layer.input = new_activation

    def add(self, direction, alpha, layer_idx):
        direction = direction / direction.norm()
        direction = direction.to(model.dtype)
        self.module.layers[layer_idx].input += alpha * direction
    
    def transform(self, sample):
        fn_vectors = torch.stack(self.fn_vectors, dim=0)
        transformed_sample = torch.matmul(sample, fn_vectors).to(model.dtype)
        transformed_sample = transformed_sample / torch.norm(transformed_sample)
        return transformed_sample

    def parameters(self):
        return self.fn_vectors
    
    def orthogonalize(self):
        with torch.no_grad():
            for i in range(len(self.fn_vectors)):
                for j in range(i):
                    self.fn_vectors[i].data.sub_(projection_einops(self.fn_vectors[i].data, self.fn_vectors[j].data))
                self.fn_vectors[i].data.div_(self.fn_vectors[i].data.norm())
            
            if self.orthogonal_vectors:
                v = self.fn_vectors[0].data.clone().cpu()
                
                # Stack your vectors as rows in a matrix A
                A = torch.stack([vec.flatten().to(torch.float32) for vec in self.orthogonal_vectors])
                # Compute projection matrix P = A^T(AA^T)^-1A
                # The nullspace projector is then I - P
                AAT = A @ A.t()
                AAT_inv = torch.inverse(AAT)
                P = A.t() @ AAT_inv @ A
                I = torch.eye(P.shape[0], device=P.device)
                
                # Project onto nullspace (orthogonal complement)
                v_flat = v.flatten()
                v_ortho = (I - P) @ v_flat
                
                # Reshape back to original shape and normalize
                v_ortho = v_ortho.reshape(v.shape)
                v_ortho = v_ortho / torch.norm(v_ortho)

                self.fn_vectors[0].data = v_ortho.to(self.fn_vectors[0].dtype).to(self.fn_vectors[0].device)
    
    def normalize(self):
        for i in range(len(self.fn_vectors)):
            self.fn_vectors[i].data.div_(self.fn_vectors[i].data.norm())
                    

def train_ablation_subspace_joint(model, train_dataset, current_batch_size, target_batch_size, epochs=10, lr=1e-2, subspace_dim=2, n_sample=1, fixed_samples=8, sampling_method="hypersphere", optimize_basis=True, fixed_basis_vectors=[], use_ablation_loss=True, ablation_lambda=1, use_addition_loss=True, alpha=alpha, addition_lambda=1, identify_add_layer=False, identify_steps=10, use_retain_loss=True, retain_lambda=1, use_repind_loss=False, repind_lambda=1, repind_layers=[], patience=5, verbose=False, init_vectors=[], n_lr_reduce=2, orthogonal_vectors=[]):

    train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate)

    operation = SubspaceAblationJoint(model.model, model.config.hidden_size, subspace_dim, init_vectors=init_vectors, orthogonal_vectors=orthogonal_vectors)

    optimizer = torch.optim.AdamW(operation.parameters(), lr=lr, betas=(.9,.98), weight_decay=0.0, amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    print("Subspace dim", subspace_dim)
    if subspace_dim == 1:
        n_sample = 0

    accumulation_steps = target_batch_size // current_batch_size
    if verbose:
        print("Accumulation steps", accumulation_steps)
    vectors = []
    train_losses = []
    stopped = False
    lowest_training_loss = float('inf')
    refusal_scores = []
    patience_counter = 0
    lr_reduce_counter = 0

    if verbose:
        print("Starting training")

    step_counter = 0
    batch_sample_ablation_loss = 0.0
    batch_sample_addition_loss = 0.0
    batch_sample_retain_loss = 0.0
    batch_basis_ablation_loss = 0.0
    batch_basis_addition_loss = 0.0
    batch_basis_retain_loss = 0.0
    batch_basis_repind_loss = 0.0

    batch_sample_refusal_scores = []
    batch_sample_induce_scores = []
    batch_basis_refusal_scores = []
    batch_basis_induce_scores = []

    if identify_add_layer:
        basis_addition_lambda = addition_lambda
        addition_lambda = 0
        most_active_layers = []
        layer_norms = []
    else:
        add_layer = best_layer

    if n_sample > 0:
        if sampling_method == "hypersphere":
            fixed_sample_vectors = sample_hypersphere_gaussian(fixed_samples, subspace_dim)
        elif sampling_method == "interpolation":
            fixed_sample_vectors = sample_prob_vectors(fixed_samples, subspace_dim)
        fixed_sample_vectors = [fixed_sample_vectors[i] for i in range(fixed_samples)]

    for epoch in range(epochs):
        if verbose:
            print('Epoch', epoch)
        for _, batch in enumerate(train_dataloader):
            ablation_prompt = batch['ablation_prompt']
            ablation_labels = batch['ablation_labels']
            addition_prompt = batch['addition_prompt']
            addition_labels = batch['addition_labels']
            retain_prompt = batch['retain_prompt']
            harmful_prompt = batch['harmful_prompt']
            harmless_prompt = batch['harmless_prompt']

            if n_sample > 0:
                if sampling_method == "hypersphere":
                    sample_vectors = sample_hypersphere_gaussian(n_sample, subspace_dim)
                elif sampling_method == "interpolation":
                    sample_vectors = sample_prob_vectors(n_sample, subspace_dim)

                for sample_vector in sample_vectors:
                    if use_ablation_loss:
                        with model.trace() as tracer:
                            with tracer.invoke(ablation_prompt):
                                direction = operation.transform(sample_vector)
                                operation(direction)
                                logits = model.lm_head.output[:, :-1]
                                sample_ablation_loss = compute_ce_loss(logits, ablation_labels) / n_sample
                                log = sample_ablation_loss.detach().item().save()
                            (ablation_lambda * sample_ablation_loss).backward()
                    batch_sample_ablation_loss += log
                    if use_addition_loss:
                        with model.trace() as tracer:
                            with tracer.invoke(addition_prompt):
                                direction = operation.transform(sample_vector)
                                operation.add(direction, alpha, add_layer)
                                logits = model.lm_head.output[:, :-1]
                                sample_addition_loss = compute_ce_loss(logits, addition_labels) / n_sample
                                log = sample_addition_loss.detach().item().save()
                            (addition_lambda * sample_addition_loss).backward()
                        batch_sample_addition_loss += log
                    if use_retain_loss:
                        with model.trace() as tracer:
                            with tracer.invoke(retain_prompt):
                                baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                            with tracer.invoke(retain_prompt):
                                direction = operation.transform(sample_vector)
                                operation(direction)
                                sample_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                                sample_retain_loss = kl_div_fn(baseline_retain_logits, sample_retain_logits).mean() / n_sample
                                log = sample_retain_loss.detach().item().save()
                            (retain_lambda * sample_retain_loss).backward()
                        batch_sample_retain_loss += log
                
            if optimize_basis:
                for fn_vector in operation.fn_vectors:
                    if use_ablation_loss:
                        with model.trace() as tracer:
                            with tracer.invoke(ablation_prompt):
                                operation(fn_vector)
                                logits = model.lm_head.output[:, :-1]
                                basis_ablation_loss = compute_ce_loss(logits, ablation_labels) / subspace_dim
                                log = basis_ablation_loss.detach().item().save()
                            (ablation_lambda * basis_ablation_loss).backward()
                        batch_basis_ablation_loss += log

                    if use_addition_loss:
                        if identify_add_layer and step_counter // accumulation_steps < identify_steps:
                            with model.trace() as tracer:
                                with tracer.invoke(harmful_prompt):
                                    cosine_sims = get_cosine_sims_for_vector(model, fn_vector)[:, -1]
                                    max_cosine_sim_layer = torch.argmax(cosine_sims).detach().save()
                                    harmful_dot_products = []
                                    for layer in model.model.layers:
                                        dot_product = dotproduct_einops(layer.input[:, -1], fn_vector.to(model.dtype))
                                        harmful_dot_products.append(dot_product)
                                    harmful_dot_products = torch.stack(harmful_dot_products).detach().save()
                            with model.trace() as tracer:
                                with tracer.invoke(harmless_prompt):
                                    harmless_dot_products = []
                                    for layer in model.model.layers:
                                        dot_product = dotproduct_einops(layer.input[:, -1], fn_vector.to(model.dtype))
                                        harmless_dot_products.append(dot_product)
                                    harmless_dot_products = torch.stack(harmless_dot_products).detach().save()
                            most_active_layers.append(max_cosine_sim_layer.value)
                            diff_dot_products = harmful_dot_products.value - harmless_dot_products.value
                            layer_norms.append(diff_dot_products.squeeze())
                        else:
                            with model.trace() as tracer:
                                with tracer.invoke(addition_prompt):
                                    operation.add(fn_vector, alpha, add_layer)
                                    logits = model.lm_head.output[:, :-1]
                                    basis_addition_loss = compute_ce_loss(logits, addition_labels) / subspace_dim
                                    log = basis_addition_loss.detach().item().save()
                                (addition_lambda * basis_addition_loss).backward()
                            batch_basis_addition_loss += log

                    if use_retain_loss:
                        with model.trace() as tracer:
                            with tracer.invoke(retain_prompt):
                                baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                            with tracer.invoke(retain_prompt):
                                operation(fn_vector)
                                retain_logits = model.lm_head.output[:, -num_target_tokens:]
                                basis_retain_loss = kl_div_fn(baseline_retain_logits, retain_logits).mean() / subspace_dim
                                log = basis_retain_loss.detach().item().save()
                            (retain_lambda * basis_retain_loss).backward()
                        batch_basis_retain_loss += log
                if use_repind_loss:
                    with model.trace() as tracer:
                        baseline_cosine_sims = []
                        for fn_vector in operation.fn_vectors:
                            with tracer.invoke(harmful_prompt):
                                baseline_cosine_sims.append(get_cosine_sims_for_vector(model, fn_vector))
                        baseline_cosine_sims = torch.stack(baseline_cosine_sims)
                        intervention_cosine_sims = []
                        for fn_vector in operation.fn_vectors:
                            with tracer.invoke(harmful_prompt):
                                operation(fn_vector)
                                intervention_cosine_sims.append(get_cosine_sims_for_vector(model, fn_vector))
                        intervention_cosine_sims = torch.stack(intervention_cosine_sims)
                        # Extract the selected dimensions for repind_layers for each basis vector.
                        baseline_sel = baseline_cosine_sims[:, repind_layers]   # shape: (subspace_dim, L)
                        intervention_sel = intervention_cosine_sims[:, repind_layers]   # shape: (subspace_dim, L)
                        
                        # Compute pairwise squared differences between all pairs (i, j) of basis vectors.
                        # This is equivalent to a double loop over the basis vectors.
                        diffs = baseline_sel.unsqueeze(1) - intervention_sel.unsqueeze(0)  # shape: (subspace_dim, subspace_dim, L)
                        pairwise_means = diffs.square().mean(dim=-1)  # shape: (subspace_dim, subspace_dim)
                        
                        # Create a mask to exclude self-comparisons (i == j).
                        mask = 1 - torch.eye(subspace_dim, device=baseline_cosine_sims.device, dtype=pairwise_means.dtype)
                        
                        # Compute the repind loss as the sum loss over all distinct vector pairs.
                        repind_loss = (pairwise_means * mask).sum()
                        (repind_lambda * repind_loss).backward()
                        log = repind_loss.detach().item().save()
                    batch_basis_repind_loss += log
                
            with torch.no_grad():
                for fn_vector in operation.fn_vectors:
                    with model.trace() as tracer:
                        with tracer.invoke(harmful_prompt):
                            operation(fn_vector)
                            last_token_logits = model.lm_head.output[:, -1]
                            refusal_score = refusal_score_fn(last_token_logits, refusal_tokens).detach().item().save()
                        batch_basis_refusal_scores.append(refusal_score)
                    if not identify_add_layer or step_counter // accumulation_steps >= identify_steps:
                        with model.trace() as tracer:
                            with tracer.invoke(harmless_prompt):
                                operation.add(fn_vector, alpha, add_layer)
                                last_token_logits = model.lm_head.output[:, -1]
                                induce_score = refusal_score_fn(last_token_logits, refusal_tokens).detach().item().save()
                            batch_basis_induce_scores.append(induce_score)
                if n_sample > 0:
                    with model.trace() as tracer:
                        for fixed_sample_vector in fixed_sample_vectors:
                            with tracer.invoke(harmful_prompt):
                                direction = operation.transform(fixed_sample_vector)
                                operation(direction)
                                sample_last_token_logits = model.lm_head.output[:, -1]
                                sample_refusal_score = refusal_score_fn(sample_last_token_logits, refusal_tokens).detach().item().save()
                                batch_sample_refusal_scores.append(sample_refusal_score)
                    with model.trace() as tracer:
                        for fixed_sample_vector in fixed_sample_vectors:
                            with tracer.invoke(harmless_prompt):
                                direction = operation.transform(fixed_sample_vector)
                                operation.add(direction, alpha, add_layer)
                                sample_last_token_logits = model.lm_head.output[:, -1]
                                sample_induce_score = refusal_score_fn(sample_last_token_logits, refusal_tokens).detach().item().save()
                                batch_sample_induce_scores.append(sample_induce_score)

                step_counter += 1
                if step_counter % accumulation_steps == 0:
                    for fn_vector in operation.fn_vectors:
                        fn_vector.grad.sub_(projection_einops(fn_vector.grad, fn_vector.data))
                    for fn_vector in operation.fn_vectors:
                        fn_vector.grad.div_(accumulation_steps)
                    torch.nn.utils.clip_grad_norm_(operation.parameters(), 10.0)
                    grad_norm = operation.fn_vectors[-1].grad.norm().item()
                    optimizer.step()
                    optimizer.zero_grad()
                    if len(fixed_basis_vectors) > 0:
                        for i, fixed_basis_vector in enumerate(fixed_basis_vectors):
                            fixed_basis_vector = fixed_basis_vector / fixed_basis_vector.norm()
                            operation.fn_vectors[i].data.copy_(fixed_basis_vector.data)
                    if use_repind_loss:
                        operation.normalize()
                    else:
                        operation.orthogonalize()

                    batch_sample_ablation_loss /= accumulation_steps
                    batch_sample_addition_loss /= accumulation_steps
                    batch_sample_retain_loss /= accumulation_steps
                    batch_basis_ablation_loss /= accumulation_steps
                    batch_basis_addition_loss /= accumulation_steps
                    batch_basis_retain_loss /= accumulation_steps
                    batch_basis_repind_loss /= accumulation_steps

                    train_loss = batch_sample_ablation_loss + batch_sample_addition_loss + batch_sample_retain_loss + batch_basis_ablation_loss + batch_basis_addition_loss + batch_basis_retain_loss + batch_basis_repind_loss
                    train_losses.append(train_loss)

                    batch_basis_refusal_scores = [s.value for s in batch_basis_refusal_scores]
                    batch_basis_induce_scores = [s.value for s in batch_basis_induce_scores]
                    basis_refusal_scores = [torch.mean(torch.tensor(batch_basis_refusal_scores[i::subspace_dim])).item() for i in range(subspace_dim)]
                    basis_induce_scores = [torch.mean(torch.tensor(batch_basis_induce_scores[i::subspace_dim])).item() for i in range(subspace_dim)]

                    vectors.append(torch.stack(operation.fn_vectors, dim=0).detach().cpu().data.clone())
                    refusal_scores.append(basis_refusal_scores)

                    wandb_logs = {
                        "train/total_loss": train_loss,
                        "train/basis_ablation_loss": batch_basis_ablation_loss,
                        "train/basis_addition_loss": batch_basis_addition_loss,
                        "train/basis_retain_loss": batch_basis_retain_loss,
                        "train/basis_repind_loss": batch_basis_repind_loss,
                        "train/basis_refusal_score": basis_refusal_scores,
                        "train/basis_induce_score": basis_induce_scores,
                        "train/grad_norm": grad_norm
                    }

                    if n_sample > 0:
                        batch_sample_refusal_scores = [s.value for s in batch_sample_refusal_scores]
                        batch_sample_induce_scores = [s.value for s in batch_sample_induce_scores]

                        sample_vector_refusal_scores = [torch.mean(torch.tensor(batch_sample_refusal_scores[i::fixed_samples])).item() for i in range(fixed_samples)]
                        min_sample_refusal_score = min(sample_vector_refusal_scores)
                        max_sample_refusal_score = max(sample_vector_refusal_scores)
                        mean_sample_refusal_score = torch.mean(torch.tensor(sample_vector_refusal_scores)).item()
                        std_sample_refusal_score = torch.std(torch.tensor(sample_vector_refusal_scores)).item()

                        sample_vector_induce_scores = [torch.mean(torch.tensor(batch_sample_induce_scores[i::fixed_samples])).item() for i in range(fixed_samples)]
                        min_sample_induce_score = min(sample_vector_induce_scores)
                        max_sample_induce_score = max(sample_vector_induce_scores)
                        mean_sample_induce_score = torch.mean(torch.tensor(sample_vector_induce_scores)).item()
                        std_sample_induce_score = torch.std(torch.tensor(sample_vector_induce_scores)).item()

                        wandb_logs.update({
                            "train/sample_ablation_loss": batch_sample_ablation_loss if n_sample > 0 else 0,
                            "train/sample_addition_loss": batch_sample_addition_loss if n_sample > 0 else 0,
                            "train/sample_retain_loss": batch_sample_retain_loss if n_sample > 0 else 0,
                            "train/min_sample_refusal_score": min_sample_refusal_score if n_sample > 0 else 0,
                            "train/max_sample_refusal_score": max_sample_refusal_score if n_sample > 0 else 0,
                            "train/mean_sample_refusal_score": mean_sample_refusal_score if n_sample > 0 else 0,
                            "train/std_sample_refusal_score": std_sample_refusal_score if n_sample > 0 else 0,
                            "train/min_sample_induce_score": min_sample_induce_score if n_sample > 0 else 0,
                            "train/max_sample_induce_score": max_sample_induce_score if n_sample > 0 else 0,
                            "train/mean_sample_induce_score": mean_sample_induce_score if n_sample > 0 else 0,
                            "train/std_sample_induce_score": std_sample_induce_score if n_sample > 0 else 0,
                        })

                    wandb.log(wandb_logs, step=step_counter)

                    print("Step", step_counter, "train/basis_vector_refusal_score", [round(s, 2) for s in basis_refusal_scores], "train/basis_vector_induce_score", [round(s, 2) for s in basis_induce_scores])
                    
                    if identify_add_layer and step_counter // accumulation_steps == identify_steps:
                        addition_lambda = basis_addition_lambda
                        most_active_layers_tensor = torch.as_tensor(most_active_layers[-target_batch_size:]).flatten()
                        print("Most active layers", most_active_layers_tensor.tolist())
                        mode_val = torch.mode(most_active_layers_tensor).values
                        if (most_active_layers_tensor == mode_val).sum().item() > 1:
                            add_layer = mode_val.item()
                        else:
                            add_layer = torch.median(most_active_layers_tensor).item()
                        print("Identified add layer", add_layer)
                        layer_norms_tensor = torch.stack(layer_norms[-target_batch_size:])
                        add_layer_layer_norms = layer_norms_tensor[:, add_layer]
                        alpha = torch.mean(add_layer_layer_norms).detach().item()
                        print("Identified alpha", alpha)
                        patience_counter = 0

                    if n_sample > 0:
                        print("train/mean_sample_refusal_scores", round(mean_sample_refusal_score, 2), "train/mean_sample_induce_scores", round(mean_sample_induce_score, 2))
                    if train_loss >= lowest_training_loss:
                        patience_counter += 1
                    else:
                        lowest_training_loss = train_loss
                        patience_counter = 0
                    if patience_counter >= patience:
                        if lr_reduce_counter >= n_lr_reduce:
                            if verbose:
                                print(f'Stopping')
                            stopped = True
                            break
                        lr_reduce_counter += 1
                        print("Reducing lr to", optimizer.param_groups[0]['lr'] / 10)
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                        patience_counter = 0

                    batch_sample_ablation_loss = 0.
                    batch_sample_addition_loss = 0.
                    batch_sample_retain_loss = 0.
                    batch_basis_ablation_loss = 0.
                    batch_basis_addition_loss = 0.
                    batch_basis_retain_loss = 0.
                    batch_basis_repind_loss = 0.

                    batch_sample_refusal_scores = []
                    batch_sample_induce_scores = []
                    batch_basis_refusal_scores = []
                    batch_basis_induce_scores = []

                    torch.cuda.empty_cache()

        if stopped:
            break

    save_vectors = vectors
    lowest_loss_index = torch.argmin(torch.tensor(train_losses)).item()
    lowest_loss_vector = save_vectors[lowest_loss_index]
    low_loss_refusal_scores = refusal_scores[lowest_loss_index]
    lowest_basis_refusal_score = min(low_loss_refusal_scores)
    run_id = wandb.run.id
    artifact = wandb.Artifact(f'trained_vectors_run_{run_id}', type='vector')
    with artifact.new_file(f'vectors.pt', mode='wb') as f:
        torch.save(save_vectors, f)
    with artifact.new_file(f'lowest_loss_vector.pt', mode='wb') as f:
        torch.save(lowest_loss_vector, f)
    wandb.log_artifact(artifact)
    
    return {"vectors": vectors, "lowest_loss": lowest_training_loss, "refusal_scores": refusal_scores, "train_losses": train_losses, "lowest_loss_vector": lowest_loss_vector, "lowest_basis_refusal_score": lowest_basis_refusal_score}

def softmin(x, beta=1.0, dim=0):
    return -torch.log(torch.exp(-beta * x).sum(dim=dim)) / beta

def train_ablation_subspace_min(model, train_dataset, current_batch_size, target_batch_size, epochs=10, lr=1e-3, subspace_dim=2, n_sample=1, fixed_samples=8, sampling_method="hypersphere", optimize_basis=False, use_ablation_loss=True, ablation_lambda=1, use_addition_loss=True, alpha=alpha, addition_lambda=1, use_retain_loss=False, retain_lambda=1, patience=5, verbose=False, init_vectors=None, n_lr_reduce=0):

    train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate)

    operation = SubspaceAblationJoint(model.model, model.config.hidden_size, subspace_dim, init_vectors=init_vectors)

    optimizer = torch.optim.AdamW(operation.parameters(), lr=lr, betas=(.9,.98), weight_decay=0.0, amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    print("Subspace dim", subspace_dim)
    if subspace_dim == 1:
        n_sample = 0

    accumulation_steps = target_batch_size // current_batch_size
    if verbose:
        print("Accumulation steps", accumulation_steps)
    vectors = []
    train_losses = []
    stopped = False
    lowest_training_loss = float('inf')
    refusal_scores = []
    patience_counter = 0
    lr_reduce_counter = 0

    if verbose:
        print("Starting training")

    step_counter = 0
    batch_sample_ablation_loss = 0.0
    batch_sample_addition_loss = 0.0
    batch_sample_retain_loss = 0.0
    batch_basis_ablation_loss = 0.0
    batch_basis_addition_loss = 0.0
    batch_basis_retain_loss = 0.0

    batch_sample_refusal_scores = []
    batch_sample_induce_scores = []
    batch_basis_refusal_scores = []
    batch_basis_induce_scores = []

    if sampling_method == "hypersphere":
        fixed_sample_vectors = sample_hypersphere_gaussian(fixed_samples, subspace_dim)
    elif sampling_method == "interpolation":
        fixed_sample_vectors = sample_prob_vectors(fixed_samples, subspace_dim)
    fixed_sample_vectors = [fixed_sample_vectors[i] for i in range(fixed_samples)]

    for epoch in range(epochs):
        if verbose:
            print('Epoch', epoch)
        for _, batch in enumerate(train_dataloader):
            ablation_prompt = batch['ablation_prompt']
            ablation_labels = batch['ablation_labels']
            addition_prompt = batch['addition_prompt']
            addition_labels = batch['addition_labels']
            retain_prompt = batch['retain_prompt']
            harmful_prompt = batch['harmful_prompt']
            harmless_prompt = batch['harmless_prompt']

            if n_sample > 0:
                if sampling_method == "hypersphere":
                    sample_vectors = sample_hypersphere_gaussian(n_sample, subspace_dim)
                elif sampling_method == "interpolation":
                    sample_vectors = sample_prob_vectors(n_sample, subspace_dim)

                if use_ablation_loss:
                    with model.trace() as tracer:
                        sample_ablation_losses = []
                        basis_ablation_losses = []
                        for sample_vector in sample_vectors:
                            with tracer.invoke(ablation_prompt):
                                direction = operation.transform(sample_vector)
                                operation(direction)
                                logits = model.lm_head.output[:, :-1]
                                sample_ablation_losses.append(compute_ce_loss(logits, ablation_labels))
                        for fn_vector in operation.fn_vectors:
                            with tracer.invoke(ablation_prompt):
                                operation(fn_vector)
                                logits = model.lm_head.output[:, :-1]
                                basis_ablation_losses.append(compute_ce_loss(logits, ablation_labels))
                        sample_ablation_loss = torch.stack(sample_ablation_losses)
                        basis_ablation_loss = torch.stack(basis_ablation_losses)
                        mean_sample_ablation_loss = sample_ablation_loss.mean(dim=0)
                        mean_basis_ablation_loss = basis_ablation_loss.mean(dim=0)
                        sample_log = mean_sample_ablation_loss.detach().item().save()
                        basis_log = mean_basis_ablation_loss.detach().item().save()
                        softmin_sample_ablation_loss = softmin(sample_ablation_loss, dim=0)
                        softmin_basis_ablation_loss = softmin(basis_ablation_loss, dim=0)
                        combined_sample_ablation_loss = 0.5 * mean_sample_ablation_loss + 0.5 * softmin_sample_ablation_loss
                        combined_basis_ablation_loss = 0.5 * mean_basis_ablation_loss + 0.5 * softmin_basis_ablation_loss
                        combined_ablation_loss = 0.5 * combined_sample_ablation_loss + 0.5 * combined_basis_ablation_loss
                        (ablation_lambda * combined_ablation_loss).backward()
                    batch_sample_ablation_loss += sample_log.value
                    batch_basis_ablation_loss += basis_log.value
                if use_addition_loss:
                    with model.trace() as tracer:
                        sample_addition_losses = []
                        basis_addition_losses = []
                        for sample_vector in sample_vectors:
                            with tracer.invoke(addition_prompt):
                                direction = operation.transform(sample_vector)
                                operation.add(direction, alpha, add_layer)
                                logits = model.lm_head.output[:, :-1]
                                sample_addition_losses.append(compute_ce_loss(logits, addition_labels))
                        for fn_vector in operation.fn_vectors:
                            with tracer.invoke(addition_prompt):
                                operation.add(fn_vector, alpha, add_layer)
                                logits = model.lm_head.output[:, :-1]
                                basis_addition_losses.append(compute_ce_loss(logits, addition_labels))
                        sample_addition_loss = torch.stack(sample_addition_losses)
                        basis_addition_loss = torch.stack(basis_addition_losses)
                        mean_sample_addition_loss = sample_addition_loss.mean(dim=0)
                        mean_basis_addition_loss = basis_addition_loss.mean(dim=0)
                        sample_log = mean_sample_addition_loss.detach().item().save()
                        basis_log = mean_basis_addition_loss.detach().item().save()
                        softmin_sample_addition_loss = softmin(sample_addition_loss, dim=0)
                        softmin_basis_addition_loss = softmin(basis_addition_loss, dim=0)
                        combined_sample_addition_loss = 0.5 * mean_sample_addition_loss + 0.5 * softmin_sample_addition_loss
                        combined_basis_addition_loss = 0.5 * mean_basis_addition_loss + 0.5 * softmin_basis_addition_loss
                        combined_addition_loss = 0.5 * combined_sample_addition_loss + 0.5 * combined_basis_addition_loss
                        (addition_lambda * combined_addition_loss).backward()
                    batch_sample_addition_loss += sample_log.value
                    batch_basis_addition_loss += basis_log.value

                if use_retain_loss:
                    with model.trace() as tracer:
                        sample_retain_losses = []
                        basis_retain_losses = []
                        with tracer.invoke(retain_prompt):
                            baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                        for sample_vector in sample_vectors:
                            with tracer.invoke(retain_prompt):
                                direction = operation.transform(sample_vector)
                                operation(direction)
                                sample_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                                sample_retain_losses.append(kl_div_fn(baseline_retain_logits, sample_retain_logits).mean())
                        for fn_vector in operation.fn_vectors:
                            with tracer.invoke(retain_prompt):
                                operation(fn_vector)
                                retain_logits = model.lm_head.output[:, -num_target_tokens:]
                                basis_retain_losses.append(kl_div_fn(baseline_retain_logits, retain_logits).mean())
                        
                        sample_retain_loss = torch.stack(sample_retain_losses)
                        basis_retain_loss = torch.stack(basis_retain_losses)
                        mean_sample_retain_loss = sample_retain_loss.mean(dim=0)
                        mean_basis_retain_loss = basis_retain_loss.mean(dim=0)
                        sample_log = mean_sample_retain_loss.detach().item().save()
                        basis_log = mean_basis_retain_loss.detach().item().save()
                        softmin_sample_retain_loss = softmin(sample_retain_loss, dim=0)
                        softmin_basis_retain_loss = softmin(basis_retain_loss, dim=0)
                        combined_sample_retain_loss = 0.5 * mean_sample_retain_loss + 0.5 * softmin_sample_retain_loss
                        combined_basis_retain_loss = 0.5 * mean_basis_retain_loss + 0.5 * softmin_basis_retain_loss
                        combined_retain_loss = 0.5 * combined_sample_retain_loss + 0.5 * combined_basis_retain_loss
                        (retain_lambda * combined_retain_loss).backward()
                    batch_sample_retain_loss += sample_log.value
                    batch_basis_retain_loss += basis_log.value
            
            with torch.no_grad():
                for fn_vector in operation.fn_vectors:
                    with model.trace() as tracer:
                        with tracer.invoke(harmful_prompt):
                            operation(fn_vector)
                            last_token_logits = model.lm_head.output[:, -1]
                            refusal_score = refusal_score_fn(last_token_logits, refusal_tokens).detach().item().save()
                        batch_basis_refusal_scores.append(refusal_score)
                    with model.trace() as tracer:
                        with tracer.invoke(harmless_prompt):
                            operation.add(fn_vector, alpha, add_layer)
                            last_token_logits = model.lm_head.output[:, -1]
                            induce_score = refusal_score_fn(last_token_logits, refusal_tokens).detach().item().save()
                        batch_basis_induce_scores.append(induce_score)
                if n_sample > 0:
                    with model.trace() as tracer:
                        for fixed_sample_vector in fixed_sample_vectors:
                            with tracer.invoke(harmful_prompt):
                                direction = operation.transform(fixed_sample_vector)
                                operation(direction)
                                sample_last_token_logits = model.lm_head.output[:, -1]
                                sample_refusal_score = refusal_score_fn(sample_last_token_logits, refusal_tokens).detach().item().save()
                                batch_sample_refusal_scores.append(sample_refusal_score)
                    with model.trace() as tracer:
                        for fixed_sample_vector in fixed_sample_vectors:
                            with tracer.invoke(harmless_prompt):
                                direction = operation.transform(fixed_sample_vector)
                                operation.add(direction, alpha, add_layer)
                                sample_last_token_logits = model.lm_head.output[:, -1]
                                sample_induce_score = refusal_score_fn(sample_last_token_logits, refusal_tokens).detach().item().save()
                                batch_sample_induce_scores.append(sample_induce_score)

            step_counter += 1
            if step_counter % accumulation_steps == 0:
                for fn_vector in operation.fn_vectors:
                    fn_vector.grad.sub_(projection_einops(fn_vector.grad, fn_vector.data))
                for fn_vector in operation.fn_vectors:
                    fn_vector.grad.div_(accumulation_steps)
                torch.nn.utils.clip_grad_norm_(operation.parameters(), 10.0)
                grad_norm = operation.fn_vectors[-1].grad.norm().item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                operation.orthogonalize()

                batch_sample_ablation_loss /= accumulation_steps
                batch_sample_addition_loss /= accumulation_steps
                batch_sample_retain_loss /= accumulation_steps
                batch_basis_ablation_loss /= accumulation_steps
                batch_basis_addition_loss /= accumulation_steps
                batch_basis_retain_loss /= accumulation_steps

                train_loss = batch_sample_ablation_loss + batch_sample_addition_loss + batch_sample_retain_loss + batch_basis_ablation_loss + batch_basis_addition_loss + batch_basis_retain_loss
                train_losses.append(train_loss)
                # scheduler.step(train_loss)

                batch_basis_refusal_scores = [s.value for s in batch_basis_refusal_scores]
                batch_basis_induce_scores = [s.value for s in batch_basis_induce_scores]
                basis_refusal_scores = [torch.mean(torch.tensor(batch_basis_refusal_scores[i::subspace_dim])).item() for i in range(subspace_dim)]
                basis_induce_scores = [torch.mean(torch.tensor(batch_basis_induce_scores[i::subspace_dim])).item() for i in range(subspace_dim)]

                vectors.append(torch.stack(operation.fn_vectors, dim=0).detach().cpu().data.clone())
                refusal_scores.append(basis_refusal_scores)

                batch_sample_refusal_scores = [s.value for s in batch_sample_refusal_scores]
                batch_sample_induce_scores = [s.value for s in batch_sample_induce_scores]

                sample_vector_refusal_scores = [torch.mean(torch.tensor(batch_sample_refusal_scores[i::fixed_samples])).item() for i in range(fixed_samples)]
                min_sample_refusal_score = min(sample_vector_refusal_scores)
                max_sample_refusal_score = max(sample_vector_refusal_scores)
                mean_sample_refusal_score = torch.mean(torch.tensor(sample_vector_refusal_scores)).item()
                std_sample_refusal_score = torch.std(torch.tensor(sample_vector_refusal_scores)).item()

                sample_vector_induce_scores = [torch.mean(torch.tensor(batch_sample_induce_scores[i::fixed_samples])).item() for i in range(fixed_samples)]
                min_sample_induce_score = min(sample_vector_induce_scores)
                max_sample_induce_score = max(sample_vector_induce_scores)
                mean_sample_induce_score = torch.mean(torch.tensor(sample_vector_induce_scores)).item()
                std_sample_induce_score = torch.std(torch.tensor(sample_vector_induce_scores)).item()

                wandb.log({
                    "train/total_loss": train_loss,
                    "train/sample_ablation_loss": batch_sample_ablation_loss if n_sample > 0 else 0,
                    "train/sample_addition_loss": batch_sample_addition_loss if n_sample > 0 else 0,
                    "train/sample_retain_loss": batch_sample_retain_loss if n_sample > 0 else 0,
                    "train/basis_ablation_loss": batch_basis_ablation_loss,
                    "train/basis_addition_loss": batch_basis_addition_loss,
                    "train/basis_retain_loss": batch_basis_retain_loss,
                    "train/basis_refusal_score": basis_refusal_scores,
                    "train/basis_induce_score": basis_induce_scores,
                    "train/min_sample_refusal_score": min_sample_refusal_score if n_sample > 0 else 0,
                    "train/max_sample_refusal_score": max_sample_refusal_score if n_sample > 0 else 0,
                    "train/mean_sample_refusal_score": mean_sample_refusal_score if n_sample > 0 else 0,
                    "train/std_sample_refusal_score": std_sample_refusal_score if n_sample > 0 else 0,
                    "train/min_sample_induce_score": min_sample_induce_score if n_sample > 0 else 0,
                    "train/max_sample_induce_score": max_sample_induce_score if n_sample > 0 else 0,
                    "train/mean_sample_induce_score": mean_sample_induce_score if n_sample > 0 else 0,
                    "train/std_sample_induce_score": std_sample_induce_score if n_sample > 0 else 0,
                    "train/grad_norm": grad_norm
                }, step=step_counter)
                print("Step", step_counter, "train/basis_vector_refusal_score", basis_refusal_scores, "train/basis_vector_induce_score", basis_induce_scores)
                if n_sample > 0:
                    print("train/mean_sample_refusal_scores", mean_sample_refusal_score)
                if train_loss >= lowest_training_loss:
                    patience_counter += 1
                else:
                    lowest_training_loss = train_loss
                    patience_counter = 0
                if patience_counter >= patience:
                    if lr_reduce_counter >= n_lr_reduce:
                        if verbose:
                            print(f'Stopping')
                        stopped = True
                        break
                    lr_reduce_counter += 1
                    print("Reducing lr to", optimizer.param_groups[0]['lr'] / 10)
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                    patience_counter = 0

                batch_sample_ablation_loss = 0.
                batch_sample_addition_loss = 0.
                batch_sample_retain_loss = 0.
                batch_basis_ablation_loss = 0.
                batch_basis_addition_loss = 0.
                batch_basis_retain_loss = 0.

                del batch_sample_refusal_scores
                del batch_sample_induce_scores
                del batch_basis_refusal_scores
                del batch_basis_induce_scores
                batch_sample_refusal_scores = []
                batch_sample_induce_scores = []
                batch_basis_refusal_scores = []
                batch_basis_induce_scores = []

                torch.cuda.empty_cache()

        if stopped:
            break

    save_vectors = vectors
    lowest_loss_index = torch.argmin(torch.tensor(train_losses)).item()
    lowest_loss_vector = save_vectors[lowest_loss_index]
    low_loss_refusal_scores = refusal_scores[lowest_loss_index]
    lowest_basis_refusal_score = min(low_loss_refusal_scores)
    run_id = wandb.run.id
    artifact = wandb.Artifact(f'trained_subspaces_run_{run_id}', type='vector')
    with artifact.new_file(f'subspaces.pt', mode='wb') as f:
        torch.save(save_vectors, f)
    with artifact.new_file(f'lowest_loss_vector.pt', mode='wb') as f:
        torch.save(lowest_loss_vector, f)
    wandb.log_artifact(artifact)
    
    return {"vectors": vectors, "lowest_loss": lowest_training_loss, "refusal_scores": refusal_scores, "train_losses": train_losses, "lowest_loss_vector": lowest_loss_vector, "lowest_basis_refusal_score": lowest_basis_refusal_score}


# %%
from transformers import set_seed
set_seed(42)
print(len(train_dataset))

# %%
def train_refusal_vector(group_name=None, run_name=None, orthogonal_vectors=[], **kwargs):
    config = {
        'epochs': 2,
        'lr': 1e-2,
        'current_batch_size': 1,
        'target_batch_size': 16,
        'use_ablation_loss': True,
        'ablation_lambda': 1,
        'use_addition_loss': True,
        'addition_lambda': 0.2,
        'use_retain_loss': True,
        'retain_lambda': 1,
        'patience': 5,
        'n_lr_reduce': 2,
    }
    config.update(kwargs)

    # Initialize wandb run
    wandb_config = config.copy()
    wandb_config.update({
        "model_id": model_id,
        "add_layer": add_layer,
        "alpha": alpha,
    })
    # Train with only the training hyperparameters
    with wandb.init(project="robust_refusal_vector", 
                   config=wandb_config, 
                   group=f"{group_name}_{model_id}", 
                   name=run_name,
                   mode="online"):
        results = train_ablation_subspace_joint(
            model=model,
            train_dataset=train_dataset,
            verbose=True,
            subspace_dim=1,
            orthogonal_vectors=orthogonal_vectors,
            **config
        )
    wandb.finish()
    return results

# %%
if args.train_direction:
    group_name = f"identify_add_layer_sb_data_with_retain" 
    for i in range(args.n_inits):
        run_name = f"run_{i+1}"
        train_refusal_vector(
            group_name=group_name,
            run_name=run_name,
            identify_add_layer=True,
            identify_steps=10,
        )

if args.train_orthogonal:
    group_name = f"sb_data_with_retain_orthogonal" 
    orthogonal_vectors = [best_refusal_direction]

    for i in range(args.n_inits):
        run_name = f"run_{i+1}"
        train_refusal_vector(
            group_name=group_name,
            run_name=run_name,
            orthogonal_vectors=orthogonal_vectors,
        )

if args.make_loss_ablation:
    group_name = f"loss_ablation_fixed"
    all_addition_lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    addition_lambda = all_addition_lambdas[args.job_number-1]
    ablation_lambda = 1 - addition_lambda
    # for i in range(args.n_inits):
    #     run_name = f"addition_lambda_{addition_lambda}_run_{i+1}"
    #     train_refusal_vector(
    #         group_name=group_name,
    #         run_name=run_name,
    #         addition_lambda=addition_lambda,
    #         ablation_lambda=ablation_lambda,
    #         use_addition_loss=addition_lambda > 0,
    #         use_ablation_loss=ablation_lambda > 0,
    #         retain_lambda=1,
    #         use_retain_loss=True,
    #     )
    group_name = f"loss_ablation_fixed_no_retain"
    for i in range(args.n_inits):
        run_name = f"addition_lambda_{addition_lambda}_run_{i+1}"
        train_refusal_vector(
            group_name=group_name,
            run_name=run_name,
            addition_lambda=addition_lambda,
            ablation_lambda=ablation_lambda,
            use_addition_loss=addition_lambda > 0,
            use_ablation_loss=ablation_lambda > 0,
            retain_lambda=0,
            use_retain_loss=False,
        )
    # ablation_lambda = float(round(1 - addition_lambda, 3))
    # for retain_lambda in retain_lambdas:
    #     retain_lambda = float(round(retain_lambda, 3))
    #     run_name = f"addition_lambda_{addition_lambda}_retain_lambda_{retain_lambda}"
    #     train_refusal_vector(
    #         group_name=group_name,
    #         run_name=run_name,
    #         addition_lambda=addition_lambda,
    #         ablation_lambda=ablation_lambda,
    #         retain_lambda=retain_lambda,
    #         use_retain_loss=retain_lambda > 0,
    #         use_addition_loss=addition_lambda > 0,
    #         use_ablation_loss=ablation_lambda > 0,
    #     )

if args.make_kl_ablation:
    group_name = f"kl_ablation"
    retain_lambdas = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]
    retain_lambda = retain_lambdas[args.job_number-1]
    for i in range(args.n_inits):
        run_name = f"retain_lambda_{retain_lambda}_run_{i+1}"
        train_refusal_vector(
            group_name=group_name,
            run_name=run_name,
            retain_lambda=retain_lambda,
            use_retain_loss=retain_lambda > 0,
        )

#  %%
# group_name = f"test_ablation"
# addition_lambda = 0.2
# ablation_lambda = 1
# run_name = f"addition_lambda_{addition_lambda}"
# train_refusal_vector(
#     group_name=group_name,
#     run_name=run_name,
#     addition_lambda=addition_lambda,
#     ablation_lambda=ablation_lambda,
#     use_addition_loss=addition_lambda > 0,
#     use_ablation_loss=ablation_lambda > 0,
#     retain_lambda=1,
#     use_retain_loss=True,
# )
# %%
@torch.no_grad()
def get_activation_differences_all_layers(model, prompt, token_position, intervention_vector):
    # Get baseline activations first
    baseline_activations = []
    with model.trace([prompt]):
        for layer in model.model.layers:
            baseline_activations.append(layer.input[0, token_position].cpu().save())
    
    # Get intervened activations
    normed_vector = intervention_vector / intervention_vector.norm()
    normed_vector = normed_vector.to(model.dtype)
    intervened_activations = []
    with model.trace([prompt]):
        for layer in model.model.layers:
            act = layer.input
            ablated_act = act - projection_einops(act, normed_vector)
            layer.input = ablated_act 
            layer.self_attn.output[0][:] -= projection_einops(layer.self_attn.output[0][:], normed_vector)
            layer.mlp.output[:] -= projection_einops(layer.mlp.output[:], normed_vector)
            save_activation = ablated_act[0, token_position].cpu().save()
            intervened_activations.append(save_activation)
    
    # Return differences between intervened and baseline for all layers
    differences = []
    for baseline, intervened in zip(baseline_activations, intervened_activations):
        # print(baseline.value.norm().item(), intervened.value.norm().item())
        diff = intervened.value - baseline.value
        # print(diff.norm().item())
        differences.append(diff)
    torch.cuda.empty_cache()
    return differences

@torch.no_grad()
def get_activation_differences(model, prompts, token_position, intervention_vector):
    all_differences = []
    for prompt in prompts:
        differences = get_activation_differences_all_layers(model, prompt, token_position, intervention_vector.cuda())
        all_differences.append(differences)
    # Average across prompts for each layer
    mean_differences = []
    for layer in range(len(all_differences[0])):
        layer_diffs = torch.stack([diffs[layer] for diffs in all_differences])
        mean_differences.append(torch.mean(layer_diffs, dim=0))
    return mean_differences

def train_independent_vector(group_name=None, run_name=None, independent_vectors=None, use_activation_differences=True, **kwargs):
    config = {
        'epochs': 2,
        'lr': 1e-2,
        'current_batch_size': 1,
        'target_batch_size': 16,
        'use_ablation_loss': True,
        'ablation_lambda': 1,
        'use_addition_loss': True,
        'addition_lambda': 0.2,
        'use_retain_loss': True,
        'retain_lambda': 1,
        'use_repind_loss': True,
        'symmetric_repind': True,
        'patience': 5,
        'n_lr_reduce': 2,
        'train_alpha': False,
        'train_layer_weights': False,
    }
    config.update(kwargs)

    refusal_results = json.load(open(f"results/refusal_dir/{model_id}/direction_metadata.json"))
    independent_layer = refusal_results["layer"]

    # Initialize wandb run
    wandb_config = config.copy()
    wandb_config.update({
        "model_id": model_id,
        "add_layer": independent_layer,
        "alpha": alpha,
        "use_activation_differences": use_activation_differences,
    })

    orthogonal_vectors = []
    if use_activation_differences:
        activation_differences = get_activation_differences(model, harmful_train_instructions, -1, intervention_vector=independent_vectors[0])
        for repind_layer in repind_layers:
            orthogonal_vectors.append(activation_differences[repind_layer])

    # Train with only the training hyperparameters
    with wandb.init(project="robust_refusal_vector", 
                   config=wandb_config, 
                   group=f"{group_name}_{model_id}", 
                   name=run_name,
                   mode="online"):
        results = train_robust_ablation_vector(
            model=model,
            train_dataset=train_dataset,
            verbose=True,
            independent_vectors=independent_vectors,
            orthogonal_vectors=orthogonal_vectors,
            **config
    )
    wandb.finish()
    return results

if args.train_independent:
    group_name = f"repind_symmetric"
    repind_lambdas = [100, 500]
    layer_cutoffs = [1]
    retain_lambdas = [0, 0.1, 1]

    for layer_cutoff in layer_cutoffs:
        repind_layers = list(range(int(model.config.num_hidden_layers * layer_cutoff)))
        for repind_lambda in repind_lambdas:
            for retain_lambda in retain_lambdas:
                for use_activation_differences in [False, True]:
                    run_name = f"repind{repind_lambda}_cutoff{layer_cutoff}_retain{retain_lambda}_diff{use_activation_differences}"
                    print(f"run_name: {run_name}")
                    results = train_independent_vector(group_name=group_name, run_name=run_name, repind_lambda=repind_lambda, repind_layers=repind_layers, use_retain_loss=retain_lambda > 0, retain_lambda=retain_lambda, use_activation_differences=use_activation_differences, use_addition_loss=True)

# if args.train_multiple_independent:

#     harmful_val = json.load(open(f'data/{splits}_splits/harmful_val.json'))
#     harmless_val = json.load(open(f'data/{splits}_splits/harmless_val.json'))
#     harmful_val_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmful_val])
#     harmless_val_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmless_val])
#     harmful_val_scores = get_refusal_scores(model, harmful_val_instructions, refusal_tokens, batch_size=batch_size)
#     harmless_val_scores = get_refusal_scores(model, harmless_val_instructions, refusal_tokens, batch_size=batch_size)
#     filtered_harmful_val_instructions = [d for d, score in zip(harmful_val_instructions, harmful_val_scores) if score > 0]
#     filtered_harmless_val_instructions = [d for d, score in zip(harmless_val_instructions, harmless_val_scores) if score < 0]
#     print(len(filtered_harmful_val_instructions))
#     harmful_val_instructions = filtered_harmful_val_instructions
#     harmless_val_instructions = filtered_harmless_val_instructions[:len(filtered_harmful_val_instructions)]
#     print(torch.mean(get_refusal_scores(model, harmful_val_instructions, refusal_tokens, batch_size=batch_size)))

#     layer_cutoff = 0.9
#     repind_layers = list(range(int(model.config.num_hidden_layers * layer_cutoff)))
#     repind_lambda = 1000
#     retain_lambda = 0.1
#     use_activation_differences = False
#     use_addition_loss = True

#     group_name = f"repind_symmetric_sum_loss_diff_cutoff09_all_tokens_final"

#     best_vectors = []
#     independent_vectors = [torch.load(f"results/refusal_dir/{model_id}/direction.pt").to(model.dtype)]
#     n_idx = 3
#     for idx in range(1, n_idx + 1):
#         mean_refusal_scores = []
#         lowest_loss_vectors = []

#         for i in range(args.n_inits):
#             run_name = f"rep_ind_{idx}_run_{i+1}"
#             results = train_independent_vector(group_name=group_name, run_name=run_name, independent_vectors=independent_vectors, repind_lambda=repind_lambda, repind_layers=repind_layers, retain_lambda=retain_lambda, use_activation_differences=use_activation_differences, use_addition_loss=use_addition_loss)

#             lowest_loss_vector = results['lowest_loss_vector']
#             lowest_loss_vectors.append(lowest_loss_vector)
#             refusal_scores = get_refusal_scores(model, harmful_val_instructions, refusal_tokens, fn_vector=lowest_loss_vector, batch_size=batch_size)
#             mean_refusal_score = refusal_scores.mean().item()
#             mean_refusal_scores.append(mean_refusal_score)
#             print(f"mean_refusal_score: {mean_refusal_score}")
        
#         # Add vector with minimal difference between refusal and induce scores
#         best_idx = np.argmin(mean_refusal_scores)
#         print(f"best_idx: {best_idx}, i.e. run name 'rep_ind{idx}_run_{best_idx+1}'")
#         independent_vectors.append(lowest_loss_vectors[best_idx])
#     print(mean_refusal_scores)
#     torch.save(independent_vectors, f"independent_vectors_final_cutoff09_1000.pt")


# if run_now:
#     group_name = f"final_test_repind_symmetric"
#     repind_lambdas = [100, 500, 1000]
#     layer_cutoffs = [1]
#     retain_lambdas = [0, 0.1, 1]
#     symmetric_repind = True
#     use_activation_differences = True
#     use_addition_loss = False
#     for layer_cutoff in layer_cutoffs:
#         repind_layers = list(range(int(model.config.num_hidden_layers * layer_cutoff)))
#         for repind_lambda in repind_lambdas:
#             run_name = f"lambda{repind_lambda}_layer_cutoff{layer_cutoff}_use_retain_loss{use_retain_loss}_symmetric_repind{symmetric_repind}_use_addition_loss{use_addition_loss}"
#             print(f"run_name: {run_name}")
#             results = train_independent_vector(group_name=group_name, run_name=run_name, repind_lambda=repind_lambda, repind_layers=repind_layers, use_retain_loss=use_retain_loss, symmetric_repind=symmetric_repind, use_addition_loss=use_addition_loss)
# %%
# intervene_with_fn_vector_ablation(model, harmful_train_instructions[:10], results['vectors'][-3].to(model.dtype), max_new_tokens=20)
# %%
# best_vector = results['vectors'][-1]
# intervene_with_fn_vector_ablation(model, harmful_train_instructions[:10], best_vector, max_new_tokens=20)
# os.makedirs(os.path.join(SAVE_DIR, "independent_vectors"), exist_ok=True)
# torch.save(best_vector, os.path.join(SAVE_DIR, "independent_vectors", "test_cos_vector_lambda1.pt"))

# %%
def train_refusal_subspace(group_name, run_name, init_vectors, **kwargs):
    config = {
        'epochs': 2,
        'lr': 1e-2,
        'current_batch_size': 1,
        'target_batch_size': 16,
        'use_ablation_loss': True,
        'n_sample': 8,
        'use_addition_loss': True,
        'addition_lambda': 0.2,
        'sampling_method': "hypersphere",
        'use_retain_loss': True,
        'retain_lambda': 1,
        'patience': 5,
        'n_lr_reduce': 2,
        'optimize_basis': True,
    }
    config.update(kwargs)

    # Initialize wandb run
    wandb_config = config.copy()
    wandb_config.update({
        "model_id": model_id,
        "add_layer": add_layer,
        "alpha": alpha,
    })
    # Train with only the training hyperparameters
    with wandb.init(project="robust_refusal_subspace", config=wandb_config, group=f"{group_name}_{model_id}", name=run_name, mode="online"):
        results = train_ablation_subspace_joint(
            model=model,
            train_dataset=train_dataset,
            verbose=True,
            init_vectors=init_vectors,
            **config
    )
    wandb.finish()
    return results

if args.train_subspace:
    subspace_dimensions = range(2, 10)
    group_name = "join_subspace"
    init_vectors = []
    lowest_basis_refusal_scores = []

    for i in subspace_dimensions:
        run_name = f"dim_{i}"
        results = train_refusal_subspace(group_name=group_name, run_name=run_name, subspace_dim=i, init_vectors=init_vectors)
        lowest_loss_vector = results['lowest_loss_vector']
        print(lowest_loss_vector.shape)
        init_vectors = list(lowest_loss_vector)
        lowest_basis_refusal_score = results['lowest_basis_refusal_score']
        lowest_basis_refusal_scores.append(lowest_basis_refusal_score)
        print(lowest_basis_refusal_score)
        if lowest_basis_refusal_score > 0 or lowest_basis_refusal_score > lowest_basis_refusal_scores[0] / 5:
            print(f"Stopping because lowest basis refusal score is {lowest_basis_refusal_score} and lowest basis refusal score is {lowest_basis_refusal_scores[0]}")
            break

# %%
def train_repind_refusal_subspace(group_name, run_name, init_vectors, repind_layers, **kwargs):
    config = {
        'epochs': 2,
        'lr': 1e-2,
        'current_batch_size': 1,
        'target_batch_size': 16,
        'use_ablation_loss': True,
        'n_sample': 16,
        'use_addition_loss': True,
        'addition_lambda': 0.2,
        'sampling_method': "hypersphere",
        'use_retain_loss': True,
        'retain_lambda': 1,
        'use_repind_loss': True,
        'repind_lambda': 200,
        'repind_layers': repind_layers,
        'patience': 5,
        'n_lr_reduce': 2,
        'optimize_basis': True,
    }
    config.update(kwargs)

    # Initialize wandb run
    wandb_config = config.copy()
    wandb_config.update({
        "model_id": model_id,
        "add_layer": add_layer,
        "alpha": alpha,
    })
    # Train with only the training hyperparameters
    with wandb.init(project="robust_refusal_subspace", config=wandb_config, group=f"{group_name}_{model_id}", name=run_name, mode="online"):
        results = train_ablation_subspace_joint(
            model=model,
            train_dataset=train_dataset,
            verbose=True,
            init_vectors=init_vectors,
            # fixed_basis_vectors=[best_refusal_direction.cuda().to(torch.float32)],
            **config
    )
    wandb.finish()
    return results

if args.train_repind_subspace:
    for retain_lambda in [1]:
        for repind_lambda in [200]:
            subspace_dimensions = range(2, 6)
            group_name = f"repind_subspace_retain{retain_lambda}_repind{repind_lambda}_fixed_sum_no_dim"
            init_vectors = []
            lowest_basis_refusal_scores = []
            repind_layers = list(range(int(model.config.num_hidden_layers * 0.9)))

            for i in subspace_dimensions:
                run_name = f"dim_{i}"
                results = train_repind_refusal_subspace(group_name=group_name, run_name=run_name, subspace_dim=i, init_vectors=init_vectors, repind_layers=repind_layers, retain_lambda=retain_lambda, repind_lambda=repind_lambda)
                lowest_loss_vector = results['lowest_loss_vector']
                print(lowest_loss_vector.shape)
                init_vectors = list(lowest_loss_vector)
                lowest_basis_refusal_score = results['lowest_basis_refusal_score']
                lowest_basis_refusal_scores.append(lowest_basis_refusal_score)

# %%
if args.train_multiple_independent_via_cone:
    retain_lambda = 0.1
    repind_lambda = 200
    subspace_dimensions = range(2, 6)
    group_name = f"multiple_repind"
    init_vectors = []
    repind_layers = list(range(int(model.config.num_hidden_layers * 0.9)))
    n_sample = 0

    for i in subspace_dimensions:
        run_name = f"dim_{i}"
        results = train_repind_refusal_subspace(group_name=group_name, run_name=run_name, subspace_dim=i, init_vectors=init_vectors, repind_layers=repind_layers, retain_lambda=retain_lambda, repind_lambda=repind_lambda, n_sample=n_sample)
    

# %%
def train_iterative_repind_refusal_subspace(group_name, run_name, init_vectors, repind_layers, **kwargs):
    config = {
        'epochs': 2,
        'lr': 1e-2,
        'current_batch_size': 1,
        'target_batch_size': 16,
        'use_ablation_loss': True,
        'n_sample': 16,
        'use_addition_loss': True,
        'addition_lambda': 0.2,
        'sampling_method': "hypersphere",
        'use_retain_loss': True,
        'retain_lambda': 1,
        'use_repind_loss': True,
        'repind_lambda': 200,
        'repind_layers': repind_layers,
        'patience': 5,
        'n_lr_reduce': 2,
        'optimize_basis': True,
    }
    config.update(kwargs)

    # Initialize wandb run
    wandb_config = config.copy()
    wandb_config.update({
        "model_id": model_id,
        "add_layer": add_layer,
        "alpha": alpha,
    })
    # Train with only the training hyperparameters
    with wandb.init(project="robust_refusal_subspace", config=wandb_config, group=f"{group_name}_{model_id}", name=run_name, mode="online"):
        results = train_ablation_subspace_joint(
            model=model,
            train_dataset=train_dataset,
            verbose=True,
            init_vectors=init_vectors,
            fixed_basis_vectors=init_vectors,
            **config
    )
    wandb.finish()
    return results

# if args.train_multiple_independent:
if True:
    harmful_val = json.load(open(f'data/{splits}_splits/harmful_val.json'))
    harmless_val = json.load(open(f'data/{splits}_splits/harmless_val.json'))
    harmful_val_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmful_val])
    harmless_val_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmless_val])
    harmful_val_scores = get_refusal_scores(model, harmful_val_instructions, refusal_tokens, batch_size=batch_size)
    harmless_val_scores = get_refusal_scores(model, harmless_val_instructions, refusal_tokens, batch_size=batch_size)
    filtered_harmful_val_instructions = [d for d, score in zip(harmful_val_instructions, harmful_val_scores) if score > 0]
    filtered_harmless_val_instructions = [d for d, score in zip(harmless_val_instructions, harmless_val_scores) if score < 0]
    print(len(filtered_harmful_val_instructions))
    harmful_val_instructions = filtered_harmful_val_instructions
    harmless_val_instructions = filtered_harmless_val_instructions[:len(filtered_harmful_val_instructions)]
    print(torch.mean(get_refusal_scores(model, harmful_val_instructions, refusal_tokens, batch_size=batch_size)))

    layer_cutoff = 0.9
    repind_layers = list(range(int(model.config.num_hidden_layers * layer_cutoff)))
    repind_lambda = 600
    retain_lambda = 0.1

    # group_name = f"repind_symmetric_sum_loss_diff_cutoff09_all_tokens_final"
    group_name = f"repind_iterative"

    best_vectors = []
    independent_vectors = [torch.load(f"results/refusal_dir/{model_id}/direction.pt").to(model.dtype)]
    n_idx = 5
    n_sample = 0
    for dim in range(2, 6):
        mean_refusal_scores = []
        lowest_loss_vectors = []

        for init in range(1, 2):
            run_name = f"dim_{dim}_run_{init}"

            results = train_iterative_repind_refusal_subspace(group_name=group_name, run_name=run_name, subspace_dim=dim, init_vectors=independent_vectors, repind_layers=repind_layers, retain_lambda=retain_lambda, repind_lambda=repind_lambda, n_sample=n_sample, use_repind_loss=True)

            lowest_loss_vector = results['lowest_loss_vector'][-1].squeeze()
            lowest_loss_vectors.append(lowest_loss_vector)
            refusal_scores = get_refusal_scores(model, harmful_val_instructions, refusal_tokens, fn_vector=lowest_loss_vector, batch_size=batch_size)
            mean_refusal_score = refusal_scores.mean().item()
            mean_refusal_scores.append(mean_refusal_score)
            print(f"mean_refusal_score: {mean_refusal_score}")
        
        # Add vector with minimal difference between refusal and induce scores
        best_idx = np.argmin(mean_refusal_scores)
        print(f"best_idx: {best_idx}, i.e. run name 'dim_{dim}_run_{best_idx+1}'")
        independent_vectors.append(lowest_loss_vectors[best_idx])
    print(mean_refusal_scores)

# %%
import sys
sys.exit()

# %%
def train_min_refusal_subspace(group_name, run_name, init_vectors, **kwargs):
    config = {
        'epochs': 2,
        'lr': 1e-2,
        'current_batch_size': 1,
        'target_batch_size': 16,
        'use_ablation_loss': True,
        'n_sample': 16,
        'use_addition_loss': True,
        'addition_lambda': 0.2,
        'sampling_method': "hypersphere",
        'use_retain_loss': True,
        'retain_lambda': 1,
        'patience': 5,
        'n_lr_reduce': 2,
        'optimize_basis': False,
    }
    config.update(kwargs)

    # Initialize wandb run
    wandb_config = config.copy()
    wandb_config.update({
        "model_id": model_id,
        "add_layer": add_layer,
        "alpha": alpha,
    })
    # Train with only the training hyperparameters
    with wandb.init(project="robust_refusal_subspace", config=wandb_config, group=f"{group_name}_{model_id}", name=run_name, mode="online"):
        results = train_ablation_subspace_min(
            model=model,
            train_dataset=train_dataset,
            verbose=True,
            init_vectors=init_vectors,
            **config
    )
    wandb.finish()
    return results
    
subspace_dimensions = range(2, 10)
group_name = "test_min_subspace_with_basis"
init_vectors = []

for i in subspace_dimensions:
    run_name = f"dim_{i}"
    results = train_min_refusal_subspace(group_name=group_name, run_name=run_name, subspace_dim=i, init_vectors=init_vectors, use_retain_loss=True, use_addition_loss=True)
    lowest_loss_vector = results['lowest_loss_vector']
    print(lowest_loss_vector.shape)
    init_vectors = list(lowest_loss_vector)
