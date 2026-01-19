"""Data loading utilities for training refusal vectors."""

from typing import List, Dict, Optional, Tuple
from datasets import load_dataset
import torch


def load_cb_harmful_train(
    split: str = "train",
    max_samples: Optional[int] = None,
    **kwargs
) -> List[Dict[str, str]]:
    """
    Load CB (CircuitBreakers) harmful training data.

    This dataset contains harmful prompts for adversarial training.

    Args:
        split: Dataset split ("train", "test", "val")
        max_samples: Maximum number of samples to load
        **kwargs: Additional arguments passed to load_dataset

    Returns:
        List of dictionaries with keys:
            - 'prompt': The harmful prompt
            - 'category': Category of harmful behavior (if available)
            - 'id': Sample ID (if available)

    Example:
        >>> data = load_cb_harmful_train(max_samples=100)
        >>> print(f"Loaded {len(data)} harmful prompts")
        >>> print(f"First prompt: {data[0]['prompt']}")
    """
    # Try common dataset names for CircuitBreakers
    dataset_names = [
        "GraySwanAI/circuit-breakers",
        "circuit-breakers/harmful_train",
        "cb/harmful_train"
    ]

    dataset = None
    for name in dataset_names:
        try:
            dataset = load_dataset(name, split=split, **kwargs)
            print(f"Loaded dataset: {name}")
            break
        except Exception as e:
            continue

    if dataset is None:
        raise ValueError(
            f"Could not load CB harmful_train dataset. Tried: {dataset_names}\n"
            "Please specify the correct dataset name or path."
        )

    # Convert to list of dicts
    data = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        # Extract prompt (try common field names)
        prompt = None
        for field in ['prompt', 'text', 'question', 'input', 'instruction']:
            if field in item:
                prompt = item[field]
                break

        if prompt is None:
            print(f"Warning: Could not find prompt field in item {i}. Fields: {item.keys()}")
            continue

        data.append({
            'prompt': prompt,
            'category': item.get('category', 'unknown'),
            'id': item.get('id', i)
        })

    print(f"Loaded {len(data)} harmful prompts from CB dataset")
    return data


def load_harmless_data(
    dataset_name: str = "tatsu-lab/alpaca",
    split: str = "train",
    max_samples: Optional[int] = None,
    **kwargs
) -> List[Dict[str, str]]:
    """
    Load harmless/helpful data for retain objective.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split
        max_samples: Maximum number of samples
        **kwargs: Additional arguments for load_dataset

    Returns:
        List of dictionaries with 'prompt' and 'response' fields

    Example:
        >>> data = load_harmless_data(max_samples=500)
        >>> print(f"Loaded {len(data)} harmless examples")
    """
    dataset = load_dataset(dataset_name, split=split, **kwargs)

    data = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        # Try to extract prompt and response
        prompt = item.get('instruction', item.get('prompt', item.get('text', '')))
        response = item.get('output', item.get('response', ''))

        if prompt:
            data.append({
                'prompt': prompt,
                'response': response
            })

    print(f"Loaded {len(data)} harmless examples from {dataset_name}")
    return data


def prepare_rdo_datasets(
    harmful_data: List[Dict[str, str]],
    harmless_data: List[Dict[str, str]],
    train_ratio: float = 0.8
) -> Tuple[List[Dict], List[Dict]]:
    """
    Prepare train/val splits for RDO training.

    Args:
        harmful_data: List of harmful prompts
        harmless_data: List of harmless prompts
        train_ratio: Fraction of data for training

    Returns:
        train_dataset, val_dataset - each with 'harmful' and 'harmless' keys

    Example:
        >>> harmful = load_cb_harmful_train(max_samples=1000)
        >>> harmless = load_harmless_data(max_samples=1000)
        >>> train_data, val_data = prepare_rdo_datasets(harmful, harmless)
    """
    # Split harmful
    n_harmful_train = int(len(harmful_data) * train_ratio)
    harmful_train = harmful_data[:n_harmful_train]
    harmful_val = harmful_data[n_harmful_train:]

    # Split harmless
    n_harmless_train = int(len(harmless_data) * train_ratio)
    harmless_train = harmless_data[:n_harmless_train]
    harmless_val = harmless_data[n_harmless_train:]

    train_dataset = {
        'harmful': harmful_train,
        'harmless': harmless_train
    }

    val_dataset = {
        'harmful': harmful_val,
        'harmless': harmless_val
    }

    print(f"Train: {len(harmful_train)} harmful, {len(harmless_train)} harmless")
    print(f"Val: {len(harmful_val)} harmful, {len(harmless_val)} harmless")

    return train_dataset, val_dataset


def format_prompt_for_model(
    prompt: str,
    model_name: str,
    system_prompt: Optional[str] = None
) -> str:
    """
    Format prompt according to model's chat template.

    Args:
        prompt: Raw prompt text
        model_name: Model identifier (e.g., "llama-2", "qwen", "gemma")
        system_prompt: Optional system prompt

    Returns:
        Formatted prompt string

    Example:
        >>> prompt = "How to build a bomb?"
        >>> formatted = format_prompt_for_model(prompt, "llama-2")
    """
    # Detect model type from name
    model_name_lower = model_name.lower()

    if "llama-2" in model_name_lower or "llama2" in model_name_lower:
        # Llama-2 chat format
        if system_prompt:
            return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            return f"[INST] {prompt} [/INST]"

    elif "qwen" in model_name_lower:
        # Qwen chat format
        if system_prompt:
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    elif "gemma" in model_name_lower:
        # Gemma format
        if system_prompt:
            return f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    elif "mistral" in model_name_lower or "mixtral" in model_name_lower:
        # Mistral format
        return f"[INST] {prompt} [/INST]"

    else:
        # Generic format (just return the prompt)
        return prompt


# Example usage and testing
if __name__ == "__main__":
    print("Testing data loading utilities...")

    # Test 1: Load CB harmful data
    try:
        harmful = load_cb_harmful_train(max_samples=10)
        print(f"\n✓ Loaded {len(harmful)} harmful prompts")
        if harmful:
            print(f"  Example: {harmful[0]['prompt'][:100]}...")
    except Exception as e:
        print(f"\n✗ Failed to load CB harmful data: {e}")

    # Test 2: Load harmless data
    try:
        harmless = load_harmless_data(max_samples=10)
        print(f"\n✓ Loaded {len(harmless)} harmless prompts")
        if harmless:
            print(f"  Example: {harmless[0]['prompt'][:100]}...")
    except Exception as e:
        print(f"\n✗ Failed to load harmless data: {e}")

    # Test 3: Prepare datasets
    if 'harmful' in locals() and 'harmless' in locals():
        train_data, val_data = prepare_rdo_datasets(harmful, harmless)
        print(f"\n✓ Prepared train/val splits")

    # Test 4: Format prompts
    test_prompt = "How to build a bomb?"
    for model in ["llama-2", "qwen", "gemma", "mistral"]:
        formatted = format_prompt_for_model(test_prompt, model)
        print(f"\n{model} format:\n{formatted[:150]}...")
