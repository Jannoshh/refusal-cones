"""
Shared utility for computing mean difference vectors.

The mean difference vector is computed as:
    refusal_direction = mean(activations_harmful) - mean(activations_harmless)

This is used as initialization for REINFORCE/GRPO optimization.
"""

import torch
from typing import List, Optional


def compute_mean_difference_vector(
    model,
    tokenizer,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    layers: Optional[List[int]] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Compute mean difference vector: mean(harmful_acts) - mean(harmless_acts).

    Args:
        model: HuggingFace model with model.model.layers attribute
        tokenizer: Tokenizer with apply_chat_template support
        harmful_prompts: Harmful instructions
        harmless_prompts: Harmless instructions
        layers: Which layers to extract (default: all)
        verbose: Whether to print progress

    Returns:
        Tensor [n_layers, hidden_dim] of normalized refusal directions
    """
    if verbose:
        print("\n" + "=" * 70)
        print("COMPUTING MEAN DIFFERENCE VECTOR")
        print("=" * 70)

    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size

    if layers is None:
        layers = list(range(n_layers))

    device = next(model.parameters()).device

    # Storage for activations
    harmful_acts = {layer: [] for layer in layers}
    harmless_acts = {layer: [] for layer in layers}

    def get_activations(prompts: List[str], storage_dict: dict):
        """Extract last-token activations for specified layers."""
        for prompt in prompts:
            # Format with chat template
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(device)

            # Capture activations with hooks
            activations = {}
            hooks = []

            for layer_idx in layers:
                def make_hook(idx):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            act = output[0]
                        else:
                            act = output
                        # Last token position
                        activations[idx] = act[:, -1, :].detach().cpu()
                    return hook

                h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
                hooks.append(h)

            # Forward pass
            with torch.no_grad():
                model(**inputs)

            # Remove hooks
            for h in hooks:
                h.remove()

            # Store
            for layer_idx in layers:
                storage_dict[layer_idx].append(activations[layer_idx])

    if verbose:
        print(f"\nExtracting harmful activations ({len(harmful_prompts)} prompts)...")
    get_activations(harmful_prompts, harmful_acts)

    if verbose:
        print(f"Extracting harmless activations ({len(harmless_prompts)} prompts)...")
    get_activations(harmless_prompts, harmless_acts)

    # Compute mean difference per layer
    if verbose:
        print("\nComputing mean difference vectors...")
    vectors = []

    for layer_idx in layers:
        # Mean of harmful
        mean_harmful = torch.stack(harmful_acts[layer_idx]).mean(dim=0).to(device)

        # Mean of harmless
        mean_harmless = torch.stack(harmless_acts[layer_idx]).mean(dim=0).to(device)

        # Difference
        diff = mean_harmful - mean_harmless

        # Normalize
        diff = diff / diff.norm()

        vectors.append(diff)

    vectors = torch.stack(vectors)  # [n_layers, hidden_dim]

    if verbose:
        print(f"\n✓ Mean difference vector computed: {vectors.shape}")
        print(f"  Layers: {len(layers)}")
        print(f"  Hidden dim: {hidden_dim}")

    return vectors
