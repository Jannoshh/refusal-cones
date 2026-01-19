"""Example: Training refusal vectors with CB harmful_train dataset.

This example shows how to:
1. Load CB harmful_train data
2. Load harmless data for retain objective
3. Run gradient-based discovery
4. Train refusal vectors with RDO
5. Evaluate the results

Usage:
    python examples/example_train_with_cb_data.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from reorganized structure
from src.utils import (
    load_cb_harmful_train,
    load_harmless_data,
    prepare_rdo_datasets,
    format_prompt_for_model
)
from src.discovery import GradientGeometryDiscovery, GradientDiscoveryConfig
from src.training import train_rdo_with_peft, RDOConfig, get_rdo_model
from src.measurement import HybridMeasurement, HybridMeasurementConfig


def main():
    """Main training pipeline with CB data."""

    # ========================================================================
    # Step 1: Configuration
    # ========================================================================

    MODEL_NAME = "Qwen/Qwen3-0.6B"  # Or your model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Data config
    N_HARMFUL_TRAIN = 1000
    N_HARMLESS_TRAIN = 1000
    N_HARMFUL_DISCOVERY = 50  # Subset for discovery

    # Discovery config
    DISCOVERY_CONFIG = GradientDiscoveryConfig(
        n_gradient_steps=20,
        n_local_iterations=30,
        enable_global_search=True,
        n_global_iterations=20,
        gradient_lr=0.1
    )

    # Training config
    TRAINING_CONFIG = RDOConfig(
        target_modules=["self_attn.o_proj"],
        operation='both',  # Ablation + addition
        projection_alpha=1.0,
        addition_alpha=1.0
    )

    print("="*80)
    print("Training Refusal Vectors with CB harmful_train Dataset")
    print("="*80)

    # ========================================================================
    # Step 2: Load Data
    # ========================================================================

    print("\n[Step 1/5] Loading datasets...")

    try:
        # Load CB harmful_train data
        harmful_data = load_cb_harmful_train(
            max_samples=N_HARMFUL_TRAIN,
            split="train"
        )
        print(f"✓ Loaded {len(harmful_data)} harmful prompts from CB dataset")

        # Load harmless data (for retain objective)
        harmless_data = load_harmless_data(
            dataset_name="tatsu-lab/alpaca",
            max_samples=N_HARMLESS_TRAIN,
            split="train"
        )
        print(f"✓ Loaded {len(harmless_data)} harmless prompts")

        # Prepare train/val splits
        train_data, val_data = prepare_rdo_datasets(
            harmful_data,
            harmless_data,
            train_ratio=0.8
        )

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nPlease ensure:")
        print("  1. CB dataset is available on HuggingFace")
        print("  2. You have access to the dataset")
        print("  3. Dataset name is correct in data_utils.py")
        return

    # ========================================================================
    # Step 3: Load Model
    # ========================================================================

    print("\n[Step 2/5] Loading model...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"✓ Loaded {MODEL_NAME}")

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # ========================================================================
    # Step 4: Discovery (Optional but Recommended)
    # ========================================================================

    print("\n[Step 3/5] Running geometry discovery...")
    print("This will take ~25-30 minutes on A100 GPU")

    # Prepare discovery prompts (subset of training data)
    discovery_prompts = [
        format_prompt_for_model(item['prompt'], MODEL_NAME)
        for item in harmful_data[:N_HARMFUL_DISCOVERY]
    ]

    # Create measurement function
    hybrid_measurement = HybridMeasurement(
        model=model,
        tokenizer=tokenizer,
        use_vllm=False,  # Set to True if vLLM available
        config=HybridMeasurementConfig(
            batch_size=8,
            max_new_tokens=50
        )
    )

    def measure_refusal_with_grad(v: torch.Tensor) -> tuple:
        """Measure refusal when ablating with vector v."""
        return hybrid_measurement.measure_with_grad(
            v=v,
            prompts=discovery_prompts
        )

    try:
        # Initialize with random vector
        n_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        v_init = torch.randn(n_layers, hidden_dim)
        v_init = v_init / v_init.norm(dim=1, keepdim=True)  # Normalize

        # Run discovery
        discovery = GradientGeometryDiscovery(
            measure_refusal_with_grad=measure_refusal_with_grad,
            v_init=v_init,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            config=DISCOVERY_CONFIG
        )

        results = discovery.discover()

        print(f"\n✓ Discovery complete!")
        print(f"  Modes found: {len(results['modes'])}")
        print(f"  Intrinsic dimension: {results['geometry']['intrinsic_dimension']}")
        print(f"  Total measurements: {results['n_measurements']}")

        # Use discovered geometry for initialization
        v_init = results['modes'][0]  # Use best mode

    except Exception as e:
        print(f"✗ Discovery failed: {e}")
        print("Continuing with random initialization...")
        v_init = torch.randn(n_layers, hidden_dim)
        v_init = v_init / v_init.norm(dim=1, keepdim=True)

    # ========================================================================
    # Step 5: Training
    # ========================================================================

    print("\n[Step 4/5] Training refusal vectors...")
    print("This will take ~6 hours on A100 GPU")

    try:
        # Get RDO model
        rdo_model = get_rdo_model(model, TRAINING_CONFIG)

        # Initialize from discovery
        # (In practice, you'd need to properly initialize the PEFT adapters)
        # For now, we'll use the default PEFT initialization

        # Train
        trained_model, trainer = train_rdo_with_peft(
            model_name=MODEL_NAME,
            harmful_data=train_data['harmful'],
            harmless_data=train_data['harmless'],
            output_dir="rdo_adapters_cb",
            num_epochs=10,
            batch_size=4,
            learning_rate=1e-3,
            lambda_ablate=1.0,   # Maximize harmfulness when ablated
            lambda_add=1.0,      # Maximize refusal when added
            lambda_retain=0.5,   # Retain helpfulness on harmless
            fp16=True
        )

        print("\n✓ Training complete!")
        print(f"  Adapters saved to: rdo_adapters_cb/")

    except Exception as e:
        print(f"✗ Training failed: {e}")
        return

    # ========================================================================
    # Step 6: Evaluation
    # ========================================================================

    print("\n[Step 5/5] Evaluating on validation set...")

    try:
        # Prepare validation prompts
        val_prompts = [
            format_prompt_for_model(item['prompt'], MODEL_NAME)
            for item in val_data['harmful'][:50]  # Test on 50 examples
        ]

        # Generate responses with ablation
        from src.training import set_operation_mode
        set_operation_mode(trained_model, operation='ablate')

        responses = []
        for prompt in val_prompts[:10]:  # Quick test on 10
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            outputs = trained_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)

        print("\n✓ Evaluation complete!")
        print("\nExample responses:")
        for i, (prompt, response) in enumerate(zip(val_prompts[:3], responses[:3])):
            print(f"\n--- Example {i+1} ---")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Response: {response[:200]}...")

    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return

    # ========================================================================
    # Done!
    # ========================================================================

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Evaluate on full test set with HarmBench classifier")
    print("  2. Analyze attack success rate (ASR)")
    print("  3. Test on different categories of harmful prompts")
    print("  4. Compare with baseline (no ablation)")
    print("\nSee docs/training/RDO_WITH_PEFT.md for more details.")


if __name__ == "__main__":
    main()
