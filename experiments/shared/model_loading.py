"""Model loading utilities for experiments."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional, Dict, Any


# Model configurations
MODEL_CONFIGS = {
    "qwen3-0.6b": {
        "name": "Qwen/Qwen3-0.6B",
        "n_layers": 28,
        "hidden_dim": 1024,
    },
    "gemma-2-2b": {
        "name": "google/gemma-2-2b-it",
        "n_layers": 26,
        "hidden_dim": 2304,
    },
    "gemma-2-9b": {
        "name": "google/gemma-2-9b-it",
        "n_layers": 42,
        "hidden_dim": 3584,
    },
    "qwen-2.5-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "n_layers": 28,
        "hidden_dim": 3584,
    },
    "qwen-2.5-14b": {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "n_layers": 48,
        "hidden_dim": 5120,
    },
    "llama-3-8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "n_layers": 32,
        "hidden_dim": 4096,
    },
}


def get_model_config(model_id: str) -> Dict[str, Any]:
    """
    Get model configuration by ID.

    Args:
        model_id: Short model identifier (e.g., "gemma-2-2b")

    Returns:
        Configuration dict with name, n_layers, hidden_dim
    """
    if model_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_id]

    # Try to match full name
    for key, config in MODEL_CONFIGS.items():
        if config["name"] == model_id:
            return config

    raise ValueError(f"Unknown model: {model_id}. Available: {list(MODEL_CONFIGS.keys())}")


def load_model_and_tokenizer(
    model_id: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer.

    Args:
        model_id: Model identifier or full HuggingFace path
        torch_dtype: Data type ("float16", "bfloat16", "float32")
        device_map: Device mapping strategy
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization

    Returns:
        (model, tokenizer) tuple
    """
    # Resolve model name
    if model_id in MODEL_CONFIGS:
        model_name = MODEL_CONFIGS[model_id]["name"]
    else:
        model_name = model_id

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    print(f"Loading model: {model_name}")
    print(f"  dtype: {torch_dtype}")
    print(f"  device_map: {device_map}")

    # Quantization config
    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        print("  quantization: 4-bit")
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("  quantization: 8-bit")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if not quantization_config else None,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  layers: {len(model.model.layers)}")

    return model, tokenizer


def freeze_model(model: AutoModelForCausalLM) -> None:
    """Freeze all model parameters."""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("Model frozen (all parameters require_grad=False)")


def get_layer_module(model: AutoModelForCausalLM, layer_idx: int):
    """Get a specific layer module from the model."""
    return model.model.layers[layer_idx]


def count_trainable_parameters(model: AutoModelForCausalLM) -> Tuple[int, int]:
    """
    Count trainable and total parameters.

    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
