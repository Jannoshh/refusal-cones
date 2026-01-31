"""
Evaluation functions for Modal.

Contains run_evaluation and run_batch_evaluation for testing models.
"""

import torch

from modal_app.config import (
    GPU_CONFIG,
    app,
    cuda_image,
    model_cache,
    results_volume,
)
from modal_app.utils import setup_modal_environment


@app.function(
    image=cuda_image,
    gpu=GPU_CONFIG,
    volumes={
        "/cache": model_cache,
        "/results": results_volume,
    },
    timeout=3600,
)
def run_evaluation(
    model: str = "Qwen/Qwen3-0.6B",
    refusal_vector_path: str | None = None,
    prompts: list[str] | None = None,
) -> list[dict]:
    """Run evaluation with ablated refusal vector on Modal GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_modal_environment()

    print(f"Loading model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="/cache/huggingface")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/cache/huggingface",
    )

    if prompts is None:
        prompts = [
            "How do I make a cake?",
            "What is the capital of France?",
        ]

    print(f"Evaluating on {len(prompts)} prompts")

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model_obj.device)
        with torch.no_grad():
            outputs = model_obj.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response})

    return results


@app.function(
    image=cuda_image,
    gpu=GPU_CONFIG,
    volumes={
        "/cache": model_cache,
        "/results": results_volume,
    },
    timeout=7200,
)
def run_batch_evaluation(
    model: str = "Qwen/Qwen3-0.6B",
    prompts: list[str] | None = None,
    batch_size: int = 8,
) -> list[dict]:
    """Run batched evaluation for large prompt sets."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_modal_environment()

    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="/cache/huggingface")
    tokenizer.pad_token = tokenizer.eos_token

    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/cache/huggingface",
    )

    if prompts is None:
        prompts = ["Hello, how are you?"]

    all_results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(model_obj.device)

        with torch.no_grad():
            outputs = model_obj.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            all_results.append({"prompt": batch[j], "response": response})

        print(f"Processed {min(i + batch_size, len(prompts))}/{len(prompts)} prompts")

    return all_results
