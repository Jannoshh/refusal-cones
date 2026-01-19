import nanogcg
import torch
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True, warn_only=True)

# %%
import json
harmful_test = json.load(open('data/saladbench_splits/harmful_test.json'))

# %%
# %%
CACHE_DIR = '/ceph/hdd/students/elsj/huggingface'
model_id = "google/gemma-2-2b-it"
# model_id = "google/gemma-2b-it"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)

#%%
import einops
def projection_einops(activation, direction):
    proj = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction
    )
    return proj

@torch.no_grad()
def get_activations(model, prompt, intervention_vector):
    """
    Get activations from all layers with directional ablation applied.

    Args:
        model: HuggingFace model
        prompt: String prompt
        intervention_vector: Direction vector to ablate

    Returns:
        Tensor of shape (num_layers, batch_size, seq_len, hidden_dim)
    """
    activations = []

    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Storage for layer activations
    layer_acts = []

    def create_hook(layer_idx):
        def hook(module, input, output):
            # Get the activation (handle tuple outputs)
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output

            # Apply ablation
            ablated_act = act - projection_einops(act, intervention_vector)

            # Store the ablated activation
            layer_acts.append(ablated_act.detach().cpu())

            # Return the ablated activation
            if isinstance(output, tuple):
                return (ablated_act,) + output[1:]
            else:
                return ablated_act
        return hook

    # Register hooks on each layer
    handles = []
    for idx, layer in enumerate(model.model.layers):
        handle = layer.register_forward_hook(create_hook(idx))
        handles.append(handle)

    # Forward pass
    _ = model(**inputs)

    # Clean up hooks
    for handle in handles:
        handle.remove()

    # Stack activations
    return torch.stack(layer_acts)

# Note: We already have the model loaded above using AutoModelForCausalLM
# The nnsight_model is no longer needed as we'll use hooks directly

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

test_suffix("Hello, how are you?")
messages = []
for i, d in enumerate(harmful_test):
    if i >= 10:  # Only process first 10 prompts
        break
    messages.append([
        {"role": "user", "content": d['instruction'] + ' {optim_str}'},
    ])

# Update config with new parameters
config = GCGConfig(
    seed=42,
    num_steps=500,  # Increased from 10 to 500
    batch_size=64,
    optim_str_init=" ".join(["!"] * 20),
    target_layers=[15],
    use_directional_ablation=True,
    use_activation_addition=False,
    directional_ablation_mode="initial",
    use_crossentropy=False,
    aux_loss_weight=1,
    target_tokens="last_chat_token",
    use_prefix_cache=False,
)

# Run GCG for each prompt and store results
results = []
for i, message in enumerate(messages):
    prompt = f"""<start_of_turn>user
{message[0]['content'].replace(' {optim_str}', '')}<end_of_turn>
<start_of_turn>model
"""
    activations = get_activations(model, prompt, repind_vector)
    target_hidden_states = activations.transpose(0, 1)[:, :, -1]
    
    result = nanogcg.run(model, tokenizer, message, "## The \"Lucky Winner\" Lottery Scam\n\n**Subject", config, 
                        direction=repind_vector, target_hidden_states=target_hidden_states)
    results.append(result)
    
    # Save results after each message
    result_dict = {
        "prompt": message[0]['content'].replace(' {optim_str}', ''),
        "best_string": result.best_string,
        "output": test_suffix(message[0]['content'].replace(' {optim_str}', '') + ' ' + result.best_string)
    }
    os.makedirs(f'results/repind_gcg', exist_ok=True)
    with open(f'results/repind_gcg/results_{i}.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
        
    print(f"Completed prompt {i+1}/{len(messages)}")
    print(f"Best string: {result.best_string}")
    print(f"Output: {test_suffix(message[0]['content'].replace(' {optim_str}', '') + ' ' + result.best_string)}\n")