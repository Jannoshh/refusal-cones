"""
Run geometry diagnostics on Qwen3-0.6B with proper Circuit Breaker data.
"""

import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.e6_unsupervised.geometry_diagnostics import run_full_diagnostics


def load_circuit_breaker_data(max_samples=100):
    """Load harmful prompts from circuit breaker dataset."""
    with open("data/splits/harmful_train.json") as f:
        harmful = json.load(f)
    return [h["instruction"] for h in harmful[:max_samples]]


def load_harmless_data(max_samples=100):
    """Load harmless prompts from alpaca."""
    with open("data/splits/harmless_train.json") as f:
        harmless = json.load(f)
    return [h["instruction"] for h in harmless[:max_samples]]


def format_chat_prompt(tokenizer, prompt: str) -> str:
    """Format prompt using the model's chat template."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_activations(model, tokenizer, prompts, layers, device, batch_size=8):
    """Extract activations at last non-pad token."""
    all_acts = {l: [] for l in layers}

    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting"):
        batch = prompts[i:i + batch_size]
        formatted = [format_chat_prompt(tokenizer, p) for p in batch]

        inputs = tokenizer(formatted, return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(device)

        seq_lens = inputs["attention_mask"].sum(dim=1) - 1

        activations = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                batch_acts = []
                for b in range(hidden.shape[0]):
                    batch_acts.append(hidden[b, seq_lens[b].item(), :])
                activations[layer_idx] = torch.stack(batch_acts).detach().cpu()
            return hook_fn

        for layer_idx in layers:
            hook = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)

        with torch.no_grad():
            model(**inputs)

        for hook in hooks:
            hook.remove()

        for layer in layers:
            all_acts[layer].append(activations[layer])

    return {l: torch.cat(all_acts[l], dim=0).numpy() for l in layers}


def main():
    print("=" * 60)
    print("GEOMETRY DIAGNOSTICS - CIRCUIT BREAKER DATA")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading Circuit Breaker data...")
    harmful_prompts = load_circuit_breaker_data(max_samples=100)
    harmless_prompts = load_harmless_data(max_samples=100)
    print(f"Loaded {len(harmful_prompts)} harmful, {len(harmless_prompts)} harmless")

    # Show examples
    print(f"\nExample harmful: {harmful_prompts[0][:80]}...")
    print(f"Example harmless: {harmless_prompts[0][:80]}...")

    # Load model
    model_name = "Qwen/Qwen3-0.6B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()

    n_layers = len(model.model.layers)
    print(f"Model: {n_layers} layers, {model.config.hidden_size} hidden dim")

    # Analyze middle layers
    layers = list(range(n_layers // 3, 2 * n_layers // 3))
    print(f"Analyzing layers: {layers}")

    # Extract activations
    print("\nExtracting activations...")
    harmful_acts = get_activations(model, tokenizer, harmful_prompts, layers, device)
    harmless_acts = get_activations(model, tokenizer, harmless_prompts, layers, device)

    # Run diagnostics per layer
    print("\n" + "=" * 60)
    print("PER-LAYER DIAGNOSTICS")
    print("=" * 60)

    results = {}
    for layer in layers:
        diag = run_full_diagnostics(harmful_acts[layer], harmless_acts[layer], verbose=False)
        results[layer] = diag
        print(f"Layer {layer}: Linear={diag.linear_score:.3f}, Curved={diag.curved_score:.3f}, "
              f"IntrDim={diag.intrinsic_dimension}, → {diag.recommended.split('(')[0].strip()}")

    # Overall
    print("\n" + "=" * 60)
    print("CONCATENATED LAYERS")
    print("=" * 60)

    harmful_concat = np.concatenate([harmful_acts[l] for l in layers], axis=1)
    harmless_concat = np.concatenate([harmless_acts[l] for l in layers], axis=1)

    full_diag = run_full_diagnostics(harmful_concat, harmless_concat, verbose=True)
    print(full_diag)


if __name__ == "__main__":
    main()
