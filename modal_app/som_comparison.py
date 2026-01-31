"""
SOM ACE vs Linear comparison on Modal.

Usage:
    # Basic run with keyword matching evaluation
    modal run run_modal.py::run_som_comparison --model qwen2-1.5b

    # Run with HarmBench test set (159 standard prompts)
    modal run run_modal.py::run_som_comparison --model qwen2-3b --use-harmbench-testset

    # Full HarmBench evaluation (requires A100-80GB for classifier)
    modal run run_modal.py::run_som_comparison --model qwen2-3b --use-harmbench-testset --use-harmbench-classifier
"""

import modal

from .config import app, model_cache, results_volume

# HuggingFace secret for gated datasets
# Create with: modal secret create huggingface HF_TOKEN=<your-token>
hf_secret = modal.Secret.from_name("huggingface")

# Image with SOM dependencies
som_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.30",
        "datasets>=2.0",
        "scikit-learn>=1.0",
        "numpy>=1.24",
        "tqdm>=4.65",
        "accelerate>=0.20",
        "huggingface_hub>=0.20",
        "minisom>=2.3",
        "einops>=0.6",
        "jaxtyping>=0.2",
        "matplotlib>=3.7",
        "plotly>=5.0",
        "seaborn>=0.12",
        "fastchat",
        "vllm>=0.4.0",  # For HarmBench classifier
    )
    .add_local_dir("som-refusal-directions", "/root/som")
)

GPU_A100 = "a100-40gb"
GPU_A100_80GB = "a100-80gb"  # Needed for HarmBench classifier


@app.function(
    image=som_image,
    gpu=GPU_A100_80GB,  # Use 80GB for classifier support
    volumes={
        "/cache": model_cache,
        "/results": results_volume,
    },
    secrets=[hf_secret],
    timeout=7200,  # 2 hours (classifier takes time)
)
def run_som_comparison(
    model: str = "qwen2-3b",
    n_test: int = 50,
    n_control: int = 50,
    som_x: int = 4,
    som_y: int = 4,
    layer: int | None = None,
    n_directions: int = 5,
    use_harmbench_testset: bool = False,
    use_harmbench_classifier: bool = False,
):
    """
    Run full SOM pipeline and compare ACE vs Linear ablation.

    Steps:
    1. Generate representations (HLx, HFx)
    2. Train SOM and generate directions
    3. Compare ACE (exact via hooks) vs Linear ablation

    Args:
        model: Model name (e.g., 'qwen2-3b', 'qwen2-1.5b')
        n_test: Number of test prompts for evaluation
        n_control: Number of control prompts for evaluation
        som_x: SOM grid X dimension
        som_y: SOM grid Y dimension
        layer: Layer for representations (auto-selected if None)
        n_directions: Number of top directions to use
        use_harmbench_testset: Use HarmBench standard 159 prompts
        use_harmbench_classifier: Use HarmBench Llama-2-13B classifier
    """
    import json
    import os
    import sys
    from datetime import datetime

    # Add SOM repo to path
    sys.path.insert(0, "/root/som")
    os.chdir("/root/som")

    # Set cache dirs
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"

    import torch

    # Auto-select layer based on model
    if layer is None:
        layer_map = {
            'qwen2-1.5b': 14,
            'qwen2-3b': 18,
            'qwen2-7b': 14,
            'llama2-7b': 16,
            'llama3-8b': 16,
        }
        layer = layer_map.get(model, 14)

    eval_method = "harmbench_classifier" if use_harmbench_classifier else "keyword_matching"
    test_set = "harmbench_standard" if use_harmbench_testset else "random_sample"

    print(f"\n{'#'*60}")
    print(f"# SOM ACE vs Linear Comparison")
    print(f"# Model: {model}")
    print(f"# SOM: {som_x}x{som_y}, Layer: {layer}")
    print(f"# Directions: {n_directions}")
    print(f"# Test set: {test_set}")
    print(f"# Evaluation: {eval_method}")
    print(f"# GPU: A100-80GB")
    print(f"{'#'*60}\n")

    results = {
        "model": model,
        "som_size": f"{som_x}x{som_y}",
        "layer": layer,
        "n_directions": n_directions,
        "n_test": n_test,
        "n_control": n_control,
        "test_set": test_set,
        "eval_method": eval_method,
    }

    # ============ STEP 1: Generate Representations ============
    print("\n" + "="*60)
    print("STEP 1: Generate Representations")
    print("="*60)

    # Cache representations on the persistent volume
    cache_repr_dir = f"/cache/representations/{model}"
    local_repr_dir = f"./dataset/representations/{model}"
    hlx_cache = f"{cache_repr_dir}/HLx_train.pt"
    hfx_cache = f"{cache_repr_dir}/HFx_train.pt"
    hlx_local = f"{local_repr_dir}/HLx_train.pt"
    hfx_local = f"{local_repr_dir}/HFx_train.pt"

    os.makedirs(cache_repr_dir, exist_ok=True)
    os.makedirs(local_repr_dir, exist_ok=True)

    if os.path.exists(hlx_cache) and os.path.exists(hfx_cache):
        print(f"Loading cached representations from {cache_repr_dir}")
        import shutil
        shutil.copy(hlx_cache, hlx_local)
        shutil.copy(hfx_cache, hfx_local)
        print("Copied to local directory")
    else:
        print("Generating representations (will cache for future runs)...")
        from create_representation_dataset import convert_samples
        convert_samples(
            model_name=model,
            generate=False,
            token_pos=-1,
            device="cuda",
        )
        # Cache for future runs
        import shutil
        shutil.copy(hlx_local, hlx_cache)
        shutil.copy(hfx_local, hfx_cache)
        model_cache.commit()
        print(f"Cached representations to {cache_repr_dir}")

    # ============ STEP 2: Train SOM ============
    print("\n" + "="*60)
    print("STEP 2: Train SOM and Generate Directions")
    print("="*60)

    from config import Config
    from som_generate_directions import load_data, train_som, som_generate_save_dirs
    from som import train_som
    from dataset.utils import compute_centroid
    import numpy as np

    model_alias = os.path.basename(model)
    cfg = Config(
        model_alias=model_alias,
        model_path=model,
        n_train=max(100, n_test * 2),  # Need enough for baseline
        n_val=max(n_test, n_control) * 2,  # Need enough for evaluation
    )

    HL_x, HF_x, Yhl, Yhf = load_data(model)
    X = np.concatenate([HF_x])
    Y = np.concatenate([Yhf])
    X_in = X[:, layer, :]

    print(f"Training {som_x}x{som_y} SOM on layer {layer}...")
    som = train_som(X_in, som_x=som_x, som_y=som_y, iterations=10000, sigma=0.1, learning_rate=0.01)

    c0 = compute_centroid(HL_x, layer=layer)
    aux_name = f'centroid_to_som{som_x}_sigma0.1_layer{layer}'
    directions, dir_path = som_generate_save_dirs(
        cfg, som, X_in, Y,
        aux_name=aux_name,
        save_weights=False,
        centroid=c0
    )

    print(f"Generated {len(directions)} directions")

    # Select top directions
    multi_dirs = directions[:n_directions]
    print(f"Using top {n_directions} directions")

    # ============ STEP 3: Compare Methods ============
    print("\n" + "="*60)
    print("STEP 3: Compare ACE vs Linear Ablation")
    print("="*60)

    from models.load_models import load_model
    from utils.ace_ablation_utils import ACEAblator
    from utils.ablation_utils import ablate_weights

    device = torch.device("cuda")

    # Get model type
    model_lower = model.lower()
    if "qwen2" in model_lower:
        model_type = "qwen2"
    elif "qwen" in model_lower:
        model_type = "qwen"
    elif "llama" in model_lower or "mistral" in model_lower:
        model_type = "llama"
    else:
        model_type = "llama"

    # Load datasets
    from dataset.load_dataset import load_dataset_split, load_dataset
    import random
    random.seed(42)

    if use_harmbench_testset:
        # Use HarmBench standard test set (159 prompts)
        print("Using HarmBench standard test set (159 prompts)")
        test_data = load_dataset('harmbench_test')
        test_prompts = [d['instruction'] for d in test_data]
        n_test = len(test_prompts)  # Use all 159
    else:
        # Use random sample from training data
        test_all = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
        random.shuffle(test_all)
        test_prompts = test_all[:n_test]

    # Control set (always from harmless training data)
    control_all = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)
    random.shuffle(control_all)

    control_prompts = control_all[:n_control]
    baseline_prompts = control_all[n_control:n_control + 100]  # Different prompts for ACE baseline

    test_dataset = [{'instruction': p, 'category': 'test'} for p in test_prompts]
    control_dataset = [{'instruction': p, 'category': 'control'} for p in control_prompts]

    print(f"Test prompts: {len(test_prompts)}, Control prompts: {len(control_prompts)}")

    # Import evaluation utilities
    from utils.completion_eval import evaluate_completions

    # Optional: HarmBench classifier
    harmbench_judge = None
    if use_harmbench_classifier:
        print("Loading HarmBench classifier (cais/HarmBench-Llama-2-13b-cls)...")
        from eval_jailbreaks import harmbench_judge_fn
        harmbench_judge = harmbench_judge_fn

    def evaluate_with_classifier(completions, prompts):
        """Evaluate with either keyword matching or HarmBench classifier."""
        keyword_eval = evaluate_completions(completions)

        if harmbench_judge is not None:
            responses = [c.get('response', '') for c in completions]
            classifications = harmbench_judge(prompts, responses)
            classifier_compliance = sum(classifications) / len(classifications)
            keyword_eval['classifier_compliance'] = classifier_compliance
            keyword_eval['classifier_refusal'] = 1.0 - classifier_compliance

        return keyword_eval

    # --- Baseline ---
    print("\n--- Baseline (no ablation) ---")
    model_base = load_model(model, device=device)

    baseline_test = model_base.generate_completions(test_dataset, max_new_tokens=64)
    baseline_test_eval = evaluate_with_classifier(baseline_test, test_prompts)
    print(f"  Test set: {baseline_test_eval['refusal_rate']*100:.1f}% refusal")
    if 'classifier_compliance' in baseline_test_eval:
        print(f"  (Classifier: {baseline_test_eval['classifier_compliance']*100:.1f}% compliance)")

    baseline_ctrl = model_base.generate_completions(control_dataset, max_new_tokens=64)
    baseline_ctrl_eval = evaluate_completions(baseline_ctrl)
    print(f"  Control set: {baseline_ctrl_eval['refusal_rate']*100:.1f}% refusal")

    results['baseline'] = {
        'test': baseline_test_eval,
        'control': baseline_ctrl_eval,
    }

    del model_base
    torch.cuda.empty_cache()

    # --- Linear Ablation ---
    print("\n--- Linear Ablation (sequential) ---")
    model_linear = load_model(model, device=device)

    for d in multi_dirs:
        ablate_weights(model_linear, d)

    linear_test = model_linear.generate_completions(test_dataset, max_new_tokens=64)
    linear_test_eval = evaluate_with_classifier(linear_test, test_prompts)
    print(f"  Test set: {linear_test_eval['refusal_rate']*100:.1f}% refusal (compliance: {linear_test_eval['compliance_rate']*100:.1f}%)")
    if 'classifier_compliance' in linear_test_eval:
        print(f"  (Classifier: {linear_test_eval['classifier_compliance']*100:.1f}% compliance)")

    linear_ctrl = model_linear.generate_completions(control_dataset, max_new_tokens=64)
    linear_ctrl_eval = evaluate_completions(linear_ctrl)
    print(f"  Control set: {linear_ctrl_eval['refusal_rate']*100:.1f}% refusal")

    results['linear'] = {
        'test': linear_test_eval,
        'control': linear_ctrl_eval,
    }

    del model_linear
    torch.cuda.empty_cache()

    # --- ACE (Exact via Hooks) ---
    print("\n--- ACE Ablation (exact via hooks) ---")
    model_ace = load_model(model, device=device)

    ablator = ACEAblator(
        model=model_ace.model,
        tokenizer=model_ace.tokenizer,
        harmless_prompts=baseline_prompts,
        model_type=model_type,
        device="cuda",
    )

    overlaps = ablator.get_subspace_baseline_overlap(multi_dirs)
    mean_overlap = sum(overlaps.values()) / len(overlaps)
    print(f"  Baseline-subspace overlap: {mean_overlap:.3f}")

    with ablator.hooks(multi_dirs):
        ace_test = model_ace.generate_completions(test_dataset, max_new_tokens=64)
    ace_test_eval = evaluate_with_classifier(ace_test, test_prompts)
    print(f"  Test set: {ace_test_eval['refusal_rate']*100:.1f}% refusal (compliance: {ace_test_eval['compliance_rate']*100:.1f}%)")
    if 'classifier_compliance' in ace_test_eval:
        print(f"  (Classifier: {ace_test_eval['classifier_compliance']*100:.1f}% compliance)")

    with ablator.hooks(multi_dirs):
        ace_ctrl = model_ace.generate_completions(control_dataset, max_new_tokens=64)
    ace_ctrl_eval = evaluate_completions(ace_ctrl)
    print(f"  Control set: {ace_ctrl_eval['refusal_rate']*100:.1f}% refusal")

    results['ace_hooks'] = {
        'test': ace_test_eval,
        'control': ace_ctrl_eval,
        'baseline_overlap': mean_overlap,
    }

    del model_ace
    torch.cuda.empty_cache()

    # ============ Summary ============
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<15} {'Test Compliance':<18} {'Control Refusal':<18}")
    print("-"*50)
    for method, data in [('baseline', results['baseline']),
                          ('linear', results['linear']),
                          ('ace_hooks', results['ace_hooks'])]:
        test_compl = data['test']['compliance_rate'] * 100
        ctrl_ref = data['control']['refusal_rate'] * 100
        print(f"{method:<15} {test_compl:<18.1f}% {ctrl_ref:<18.1f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results['timestamp'] = timestamp

    output_path = f"/results/som_comparison_{model}_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also save directions
    directions_output = f"/results/som_directions_{model}_{timestamp}.pt"
    torch.save(directions, directions_output)
    print(f"Directions saved to {directions_output}")

    results_volume.commit()

    return results
