import os
import torch
import json
from generate_utils import intervene_with_fn_vector_ablation, intervene_with_fn_vector_addition, generate_completions

def generate_harmful_targets(model, harmful_instructions, best_refusal_direction, targets_path, max_new_tokens):
    """
    Generate 'harmful' targets by ablation.
    """
    if os.path.exists(targets_path):
        current_targets = json.load(open(targets_path))
        existing_prompts = {t['prompt'] for t in current_targets}
        new_instructions = [i for i in harmful_instructions if i not in existing_prompts]
        if not new_instructions:
            return current_targets
    else:
        current_targets = []
        new_instructions = harmful_instructions

    ablation_completions = intervene_with_fn_vector_ablation(
        model,
        new_instructions,
        best_refusal_direction.to(model.dtype),
        max_new_tokens=max_new_tokens,
        batch_size=1  # or use your desired batch_size
    )

    new_targets = []
    for i, instruction in enumerate(new_instructions):
        target_dict = {
            'prompt': instruction,
            'ablation': ablation_completions[i] if ablation_completions else "",
        }
        new_targets.append(target_dict)

    all_targets = current_targets + new_targets
    os.makedirs(os.path.dirname(targets_path), exist_ok=True)
    with open(targets_path, 'w') as f:
        json.dump(all_targets, f)

    return all_targets

def generate_harmless_targets(model, harmless_instructions, targets_path, max_new_tokens):
    """
    Generate 'harmless' targets by addition (and a baseline).
    """
    if os.path.exists(targets_path):
        current_targets = json.load(open(targets_path))
        existing_prompts = {t['prompt'] for t in current_targets}
        new_instructions = [i for i in harmless_instructions if i not in existing_prompts]
        if not new_instructions:
            return current_targets
    else:
        current_targets = []
        new_instructions = harmless_instructions

    # You might want to reference your best_layer and best_refusal_direction here,
    # or pass them as function parameters.
    # For example:
    # from directopt import best_layer, best_refusal_direction
    # alpha = best_refusal_direction.norm()

    addition_completions = intervene_with_fn_vector_addition(
        model,
        new_instructions,
        best_layer,                  # Example usage
        best_refusal_direction.norm(), 
        best_refusal_direction, 
        max_new_tokens=max_new_tokens,
        batch_size=1
    )

    retain_completions = generate_completions(
        model, 
        new_instructions, 
        max_new_tokens=max_new_tokens - 1,
        batch_size=1
    )

    new_targets = []
    for i, instruction in enumerate(new_instructions):
        target_dict = {
            'prompt': instruction,
            'addition': addition_completions[i].split(".")[0] if addition_completions else "",
            'retain': retain_completions[i] if retain_completions else ""
        }
        new_targets.append(target_dict)

    all_targets = current_targets + new_targets
    os.makedirs(os.path.dirname(targets_path), exist_ok=True)
    with open(targets_path, 'w') as f:
        json.dump(all_targets, f)

    return all_targets 