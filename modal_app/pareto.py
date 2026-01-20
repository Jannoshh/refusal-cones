"""
Pareto-based discovery functions for Modal.

Contains run_pareto_discovery, run_pareto_ablations, and run_high_budget_discovery.
"""

from dataclasses import asdict, dataclass

import torch

from modal_app.config import (
    DEFAULT_N_HARMFUL,
    DEFAULT_N_HARMLESS,
    GPU_CONFIG,
    app,
    cuda_image,
    model_cache,
    results_volume,
)
from modal_app.utils import (
    get_refusal_tokens,
    load_prompts,
    make_serializable,
    setup_modal_environment,
    to_list,
)


def create_pareto_visualization(
    results,
    timestamp: str,
    title_prefix: str = "",
) -> str | None:
    """
    Create visualization for Pareto discovery results.

    Args:
        results: Discovery results with V_observed, scores_observed, pareto_scores
        timestamp: Timestamp for filename
        title_prefix: Optional prefix for plot title

    Returns:
        Path to saved plot, or None if visualization failed
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Get all scores
        all_refusal = np.array(to_list(results.scores_observed["refusal_score"]))
        all_kl = np.array(to_list(results.scores_observed["kl_score"]))
        pareto_refusal = np.array(to_list(results.pareto_scores["refusal_score"]))
        pareto_kl = np.array(to_list(results.pareto_scores["kl_score"]))

        # Get Pareto mask
        n_total = len(all_refusal)
        n_pareto = len(pareto_refusal)
        pareto_mask = np.zeros(n_total, dtype=bool)
        # Find which indices are Pareto (approximate by matching scores)
        for pr, pk in zip(pareto_refusal, pareto_kl):
            for i, (ar, ak) in enumerate(zip(all_refusal, all_kl)):
                if abs(ar - pr) < 0.01 and abs(ak - pk) < 0.01:
                    pareto_mask[i] = True
                    break

        # === Plot 1: Objective Space (Pareto frontier) ===
        ax1 = axes[0]
        ax1.scatter(
            all_refusal[~pareto_mask],
            all_kl[~pareto_mask],
            c="lightgray",
            alpha=0.5,
            s=30,
            label="Dominated",
        )
        ax1.scatter(
            all_refusal[pareto_mask],
            all_kl[pareto_mask],
            c="red",
            s=100,
            marker="*",
            label="Pareto",
            zorder=5,
        )

        # Connect Pareto points
        pareto_pairs = sorted(zip(pareto_refusal, pareto_kl))
        if pareto_pairs:
            pareto_r_sorted, pareto_k_sorted = zip(*pareto_pairs)
            ax1.plot(pareto_r_sorted, pareto_k_sorted, "r--", alpha=0.7, linewidth=2)

        ax1.set_xlabel("Refusal Score (lower = better)")
        ax1.set_ylabel("KL Score (lower = better)")
        ax1.set_title(
            f"{title_prefix}Objective Space\n{n_pareto} Pareto vectors, HV={results.hypervolume:.1f}"
        )
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # === Plot 2: t-SNE of vector space ===
        ax2 = axes[1]
        V_all = results.V_observed.reshape(n_total, -1).numpy()

        # Use PCA first to reduce to 50 dims, then t-SNE to 2D
        if V_all.shape[1] > 50:
            pca = PCA(n_components=50)
            V_pca = pca.fit_transform(V_all)
        else:
            V_pca = V_all

        tsne = TSNE(n_components=2, perplexity=min(30, n_total - 1), random_state=42)
        V_2d = tsne.fit_transform(V_pca)

        # Color by refusal score
        scatter = ax2.scatter(
            V_2d[~pareto_mask, 0],
            V_2d[~pareto_mask, 1],
            c=all_refusal[~pareto_mask],
            cmap="viridis",
            alpha=0.6,
            s=30,
        )
        ax2.scatter(
            V_2d[pareto_mask, 0],
            V_2d[pareto_mask, 1],
            c="red",
            s=100,
            marker="*",
            edgecolors="black",
            linewidths=1,
            label="Pareto",
            zorder=5,
        )
        plt.colorbar(scatter, ax=ax2, label="Refusal Score")
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        ax2.set_title("Vector Space (t-SNE)\nColored by Refusal Score")
        ax2.legend(loc="upper right")

        # === Plot 3: PCA of vector space ===
        ax3 = axes[2]
        pca_2d = PCA(n_components=2)
        V_pca2d = pca_2d.fit_transform(V_all)
        var_explained = pca_2d.explained_variance_ratio_

        scatter = ax3.scatter(
            V_pca2d[~pareto_mask, 0],
            V_pca2d[~pareto_mask, 1],
            c=all_kl[~pareto_mask],
            cmap="plasma",
            alpha=0.6,
            s=30,
        )
        ax3.scatter(
            V_pca2d[pareto_mask, 0],
            V_pca2d[pareto_mask, 1],
            c="red",
            s=100,
            marker="*",
            edgecolors="black",
            linewidths=1,
            label="Pareto",
            zorder=5,
        )
        plt.colorbar(scatter, ax=ax3, label="KL Score")
        ax3.set_xlabel(f"PC1 ({var_explained[0] * 100:.1f}% var)")
        ax3.set_ylabel(f"PC2 ({var_explained[1] * 100:.1f}% var)")
        ax3.set_title("Vector Space (PCA)\nColored by KL Score")
        ax3.legend(loc="upper right")

        plt.tight_layout()
        plot_path = f"/results/pareto_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Visualization saved to {plot_path}")
        return plot_path
    except Exception as e:
        import traceback

        print(f"Could not create visualization: {e}")
        traceback.print_exc()
        return None


@app.function(
    image=cuda_image,
    gpu=GPU_CONFIG,
    volumes={
        "/cache": model_cache,
        "/results": results_volume,
    },
    timeout=3600,
)
def run_pareto_discovery(
    model: str = "Qwen/Qwen3-0.6B",
    n_init_samples: int = 30,
    n_pareto_iterations: int = 70,
    max_measurements: int = 150,
    n_harmful: int = DEFAULT_N_HARMFUL,
    n_harmless: int = DEFAULT_N_HARMLESS,
    generate_completions: bool = False,
    n_kl_tokens: int = 1,
    generation_batch_size: int = 32,
) -> dict:
    """
    Run Pareto boundary discovery on Modal GPU.

    Finds vectors on the Pareto frontier of (refusal_score, kl_score),
    representing optimal trade-offs between refusal ablation and
    behavior preservation.

    Args:
        model: HuggingFace model name
        n_init_samples: Number of initial random samples
        n_pareto_iterations: Number of Pareto optimization iterations
        max_measurements: Maximum total measurements
        n_harmful: Number of harmful prompts for refusal scoring (default 32)
        n_harmless: Number of harmless prompts for KL scoring (default 32)
        generate_completions: If True, generate completions for multi-token KL.
            This follows the RDO paper's approach where KL is computed on the
            model's own completions. Adds ~1-2 min startup time.
        n_kl_tokens: Number of completion tokens for KL (default 1 = single token).
            Set to 30 for RDO-style multi-token KL. Only used if generate_completions=True.
        generation_batch_size: Batch size for completion generation (default 32).
    """
    import json
    from datetime import datetime

    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_modal_environment()

    from src.discovery.pareto_boundary_discovery import (
        MultiObjectiveScorer,
        ParetoDiscoveryConfig,
        ParetoGeometryDiscovery,
    )

    print(f"Loading model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="/cache/huggingface")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/cache/huggingface",
    )

    # Get model dimensions
    hidden_dim = model_obj.config.hidden_size
    n_layers = model_obj.config.num_hidden_layers
    print(f"Model: hidden_dim={hidden_dim}, n_layers={n_layers}")

    # Load prompts and targets from dataset files
    harmful_prompts, harmful_targets, harmless_prompts = load_prompts(
        n_harmful=n_harmful, n_harmless=n_harmless
    )
    print(f"Using {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts")
    print(f"Harmful targets available: {len(harmful_targets)} (for multi-token ablation loss)")

    # Refusal tokens (common refusal starters)
    refusal_toks = get_refusal_tokens(tokenizer)
    print(f"Refusal tokens: {refusal_toks.tolist()}")

    # Create scorer
    # Use harmful_targets for multi-token ablation loss (like RDO paper)
    # Use generate_completions for multi-token retain loss
    print("Creating multi-objective scorer...")
    scorer = MultiObjectiveScorer(
        model=model_obj,
        tokenizer=tokenizer,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
        refusal_toks=refusal_toks,
        device="cuda",
        harmful_completions=harmful_targets,  # Precomputed ablation targets
        n_score_tokens=n_kl_tokens,  # Use same token count for ablation loss
        generate_completions=generate_completions,
        n_kl_tokens=n_kl_tokens,
        generation_batch_size=generation_batch_size,
    )

    # Configure discovery
    config = ParetoDiscoveryConfig(
        n_init_samples=n_init_samples,
        n_pareto_iterations=n_pareto_iterations,
        max_measurements=max_measurements,
        gp_type="structured",  # Use StructuredLayerGP for layer smoothness
        layer_lengthscale=3.0,  # Smooth across ~3 adjacent layers
        learn_layer_weights=True,  # Learn which layers matter (ARD)
        kernel_lengthscale=0.5,
        # Gradient-based candidate generation (efficient in high dims)
        use_gradient_candidates=True,  # Default: use gradient descent for candidates
        gradient_steps=5,  # Steps per candidate
        gradient_lr=0.1,  # Riemannian gradient step size
        n_gradient_candidates=20,  # Gradient-based candidates per iteration
    )

    # Run discovery
    # use_mean_diff_init=True (default) computes v_init from mean-diff of activations
    print("Starting Pareto discovery...")
    discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        use_mean_diff_init=True,  # Compute v_init from scorer (ACE-style initialization)
    )

    results = discovery.discover()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"/results/pareto_discovery_{timestamp}.json"

    serializable_results = {
        "n_measurements": results.n_measurements,
        "hypervolume": float(results.hypervolume),
        "n_pareto_vectors": len(results.pareto_vectors),
        "pareto_scores": {k: to_list(v) for k, v in results.pareto_scores.items()},
        "all_scores": {k: to_list(v) for k, v in results.scores_observed.items()},
        "model": model,
        "config": {
            "n_init_samples": n_init_samples,
            "n_pareto_iterations": n_pareto_iterations,
            "max_measurements": max_measurements,
        },
    }

    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    # Create visualization
    create_pareto_visualization(results, timestamp)

    # Save pareto vectors as tensor
    vectors_path = f"/results/pareto_vectors_{timestamp}.pt"
    torch.save(results.pareto_vectors, vectors_path)

    print(f"\nResults saved to {results_path}")
    print(f"Pareto vectors saved to {vectors_path}")
    results_volume.commit()

    return serializable_results


@app.function(
    image=cuda_image,
    gpu=GPU_CONFIG,
    volumes={
        "/cache": model_cache,
        "/results": results_volume,
    },
    timeout=14400,  # 4 hours for full ablation suite
)
def run_pareto_ablations(
    model: str = "Qwen/Qwen3-0.6B",
    configs: str = "",  # Comma-separated config names, or empty for all
    n_harmful: int = DEFAULT_N_HARMFUL,
    n_harmless: int = DEFAULT_N_HARMLESS,
) -> list[dict]:
    """
    Run hyperparameter ablations for Pareto discovery.

    Tests different configurations and reports quality metrics:
    - Hypervolume (HV): volume dominated by Pareto frontier
    - Pareto size: number of non-dominated vectors
    - Best refusal: minimum refusal score achieved
    - Best KL: minimum KL divergence achieved
    - Efficiency: HV per measurement

    Args:
        model: Model to use
        configs: List of config names to run (None = all)
        n_harmful: Number of harmful prompts for refusal scoring (default 32)
        n_harmless: Number of harmless prompts for KL scoring (default 32)
    """
    import json
    import time
    from datetime import datetime

    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_modal_environment()

    from src.discovery.pareto_boundary_discovery import (
        MultiObjectiveScorer,
        ParetoDiscoveryConfig,
        ParetoGeometryDiscovery,
    )

    @dataclass
    class AblationConfig:
        name: str
        n_init_samples: int = 30
        n_pareto_iterations: int = 70
        kernel_lengthscale: float = 0.3
        beta: float = 1.96
        n_candidates: int = 500
        use_sparse_gp: bool = False
        num_inducing: int = 64

    # Load model once
    print(f"Loading model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="/cache/huggingface")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/cache/huggingface",
    )

    hidden_dim = model_obj.config.hidden_size
    n_layers = model_obj.config.num_hidden_layers
    print(f"Model: hidden_dim={hidden_dim}, n_layers={n_layers}")

    # Load prompts and targets from dataset files
    harmful_prompts, harmful_targets, harmless_prompts = load_prompts(
        n_harmful=n_harmful, n_harmless=n_harmless
    )
    print(f"Using {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts")

    refusal_toks = get_refusal_tokens(tokenizer)

    # Define ablation configs
    all_configs = [
        AblationConfig(name="baseline", beta=0.5),  # Updated default
        # Vary n_init_samples
        AblationConfig(name="n_init_10", n_init_samples=10, beta=0.5),
        AblationConfig(name="n_init_20", n_init_samples=20, beta=0.5),
        AblationConfig(name="n_init_50", n_init_samples=50, beta=0.5),
        # Vary n_pareto_iterations
        AblationConfig(name="n_iter_30", n_pareto_iterations=30, beta=0.5),
        AblationConfig(name="n_iter_50", n_pareto_iterations=50, beta=0.5),
        AblationConfig(name="n_iter_100", n_pareto_iterations=100, beta=0.5),
        # Vary kernel lengthscale
        AblationConfig(name="lengthscale_0.1", kernel_lengthscale=0.1, beta=0.5),
        AblationConfig(name="lengthscale_0.2", kernel_lengthscale=0.2, beta=0.5),
        AblationConfig(name="lengthscale_0.5", kernel_lengthscale=0.5, beta=0.5),
        AblationConfig(name="lengthscale_1.0", kernel_lengthscale=1.0, beta=0.5),
        # Vary beta (exploration)
        AblationConfig(name="beta_0.25", beta=0.25),
        AblationConfig(name="beta_1.0", beta=1.0),
        AblationConfig(name="beta_2.0", beta=2.0),
        AblationConfig(name="beta_3.0", beta=3.0),
        # Vary n_candidates
        AblationConfig(name="n_candidates_100", n_candidates=100, beta=0.5),
        AblationConfig(name="n_candidates_250", n_candidates=250, beta=0.5),
        AblationConfig(name="n_candidates_1000", n_candidates=1000, beta=0.5),
    ]

    # Filter configs if specified
    if configs:
        config_names = [c.strip() for c in configs.split(",")]
        all_configs = [c for c in all_configs if c.name in config_names]

    print(f"\nRunning {len(all_configs)} ablation configurations")

    results = []
    for i, config in enumerate(all_configs):
        print(f"\n{'=' * 70}")
        print(f"Ablation {i + 1}/{len(all_configs)}: {config.name}")
        print(f"{'=' * 70}")

        try:
            torch.manual_seed(42)

            # Build config
            pareto_config = ParetoDiscoveryConfig(
                n_init_samples=config.n_init_samples,
                n_pareto_iterations=config.n_pareto_iterations,
                kernel_lengthscale=config.kernel_lengthscale,
                beta=config.beta,
                n_candidates=config.n_candidates,
                use_sparse_gp=config.use_sparse_gp,
                num_inducing=config.num_inducing,
            )

            # Create scorer with harmful targets for multi-token ablation loss
            scorer = MultiObjectiveScorer(
                model=model_obj,
                tokenizer=tokenizer,
                harmful_prompts=harmful_prompts,
                harmless_prompts=harmless_prompts,
                refusal_toks=refusal_toks,
                device="cuda",
                harmful_completions=harmful_targets,
                n_score_tokens=30,  # Multi-token ablation loss
            )

            # Initial vector
            v_init = torch.randn(n_layers, hidden_dim)
            v_init = v_init / v_init.norm(dim=-1, keepdim=True)

            # Run discovery
            start_time = time.time()
            discovery = ParetoGeometryDiscovery(
                scorer=scorer,
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                config=pareto_config,
                v_init=v_init,
            )
            disc_results = discovery.discover()
            wall_time = time.time() - start_time

            # Compute spread
            spreads = []
            for obj in ["refusal_score", "kl_score"]:
                if len(disc_results.pareto_scores[obj]) > 1:
                    spreads.append(disc_results.pareto_scores[obj].std().item())
            spread = sum(spreads) / len(spreads) if spreads else 0.0

            result = {
                "config": asdict(config),
                "hypervolume": float(disc_results.hypervolume),
                "pareto_size": len(disc_results.pareto_vectors),
                "best_refusal": float(disc_results.pareto_scores["refusal_score"].min()),
                "best_kl": float(disc_results.pareto_scores["kl_score"].min()),
                "spread": spread,
                "n_measurements": disc_results.n_measurements,
                "wall_time": wall_time,
                "hv_per_measurement": disc_results.hypervolume / disc_results.n_measurements,
            }
            results.append(result)

            print(f"\n  HV: {result['hypervolume']:.4f}")
            print(f"  Pareto size: {result['pareto_size']}")
            print(f"  Best refusal: {result['best_refusal']:.4f}")
            print(f"  Best KL: {result['best_kl']:.4f}")
            print(f"  Efficiency: {result['hv_per_measurement']:.4f} HV/meas")
            print(f"  Time: {result['wall_time']:.1f}s")

        except Exception as e:
            import traceback

            print(f"ERROR: {e}")
            traceback.print_exc()
            continue

    # Print summary table
    print("\n" + "=" * 100)
    print("ABLATION SUMMARY")
    print("=" * 100)
    print(
        f"{'Config':<25} {'HV':>10} {'Pareto':>8} {'Refusal':>10} "
        f"{'KL':>10} {'Spread':>8} {'Eff':>8} {'Time':>8}"
    )
    print("-" * 100)

    sorted_results = sorted(results, key=lambda r: r["hypervolume"], reverse=True)
    for r in sorted_results:
        print(
            f"{r['config']['name']:<25} {r['hypervolume']:>10.2f} {r['pareto_size']:>8} "
            f"{r['best_refusal']:>10.4f} {r['best_kl']:>10.4f} {r['spread']:>8.4f} "
            f"{r['hv_per_measurement']:>8.4f} {r['wall_time']:>7.1f}s"
        )
    print("=" * 100)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"/results/ablation_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)

    print(f"\nResults saved to {results_path}")
    results_volume.commit()

    return make_serializable(results)


@app.function(
    image=cuda_image,
    gpu=GPU_CONFIG,
    volumes={
        "/cache": model_cache,
        "/results": results_volume,
    },
    timeout=7200,  # 2 hours for high budget run
)
def run_high_budget_discovery(
    model: str = "Qwen/Qwen3-0.6B",
    n_init_samples: int = 50,
    n_pareto_iterations: int = 200,
    max_measurements: int = 300,
    n_harmful: int = DEFAULT_N_HARMFUL,
    n_harmless: int = DEFAULT_N_HARMLESS,
    generate_completions: bool = False,
    n_kl_tokens: int = 1,
    generation_batch_size: int = 32,
) -> dict:
    """
    Run high-budget Pareto discovery with optimized hyperparameters.

    Uses beta=0.5 (best from ablations) with more samples and iterations.

    Args:
        model: HuggingFace model name
        n_init_samples: Number of initial random samples
        n_pareto_iterations: Number of Pareto optimization iterations
        max_measurements: Maximum total measurements
        n_harmful: Number of harmful prompts for refusal scoring (default 32)
        n_harmless: Number of harmless prompts for KL scoring (default 32)
        generate_completions: If True, generate completions for multi-token KL
        n_kl_tokens: Number of completion tokens for KL (default 1)
        generation_batch_size: Batch size for completion generation (default 32)
    """
    import json
    import time
    from datetime import datetime

    import matplotlib
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_modal_environment()

    from src.discovery.pareto_boundary_discovery import (
        MultiObjectiveScorer,
        ParetoDiscoveryConfig,
        ParetoGeometryDiscovery,
    )

    print(f"Loading model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="/cache/huggingface")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/cache/huggingface",
    )

    hidden_dim = model_obj.config.hidden_size
    n_layers = model_obj.config.num_hidden_layers
    print(f"Model: hidden_dim={hidden_dim}, n_layers={n_layers}")
    print(
        f"High-budget config: n_init={n_init_samples}, "
        f"n_iter={n_pareto_iterations}, max={max_measurements}"
    )

    # Load prompts and targets from dataset files
    harmful_prompts, harmful_targets, harmless_prompts = load_prompts(
        n_harmful=n_harmful, n_harmless=n_harmless
    )
    print(f"Using {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts")
    print(f"Harmful targets available: {len(harmful_targets)} (for multi-token ablation loss)")

    refusal_tokens = ["I", "Sorry", "I'm", "As", "I cannot", "I can't"]
    refusal_toks = tokenizer(refusal_tokens, add_special_tokens=False)["input_ids"]
    refusal_toks = torch.tensor([t[0] for t in refusal_toks if len(t) > 0])

    scorer = MultiObjectiveScorer(
        model=model_obj,
        tokenizer=tokenizer,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
        refusal_toks=refusal_toks,
        device="cuda",
        harmful_completions=harmful_targets,  # Precomputed ablation targets
        n_score_tokens=n_kl_tokens,  # Use same token count for ablation loss
        generate_completions=generate_completions,
        n_kl_tokens=n_kl_tokens,
        generation_batch_size=generation_batch_size,
    )

    # Optimized config from ablations
    config = ParetoDiscoveryConfig(
        n_init_samples=n_init_samples,
        n_pareto_iterations=n_pareto_iterations,
        max_measurements=max_measurements,
        beta=0.5,  # Best from ablations
        gp_type="structured",  # Use StructuredLayerGP for layer smoothness
        layer_lengthscale=3.0,  # Smooth across ~3 adjacent layers
        learn_layer_weights=True,  # Learn which layers matter (ARD)
        kernel_lengthscale=0.3,
        # Gradient-based candidate generation (efficient in high dims)
        use_gradient_candidates=True,
        gradient_steps=5,
        gradient_lr=0.1,
        n_gradient_candidates=20,
    )

    print("\nStarting high-budget Pareto discovery...")
    discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        use_mean_diff_init=True,  # Compute v_init from scorer
    )

    start_time = time.time()
    results = discovery.discover()
    wall_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print("HIGH-BUDGET RUN COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time: {wall_time:.1f}s ({wall_time / 60:.1f} min)")
    print(f"  Measurements: {results.n_measurements}")
    print(f"  Hypervolume: {results.hypervolume:.4f}")
    print(f"  Pareto vectors: {len(results.pareto_vectors)}")
    print(f"  Best refusal: {results.pareto_scores['refusal_score'].min():.4f}")
    print(f"  Best KL: {results.pareto_scores['kl_score'].min():.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    serializable_results = {
        "n_measurements": results.n_measurements,
        "hypervolume": float(results.hypervolume),
        "n_pareto_vectors": len(results.pareto_vectors),
        "pareto_scores": {k: to_list(v) for k, v in results.pareto_scores.items()},
        "all_scores": {k: to_list(v) for k, v in results.scores_observed.items()},
        "wall_time": wall_time,
        "model": model,
        "config": {
            "n_init_samples": n_init_samples,
            "n_pareto_iterations": n_pareto_iterations,
            "max_measurements": max_measurements,
            "beta": 0.5,
        },
    }

    results_path = f"/results/high_budget_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    # Save vectors
    vectors_path = f"/results/high_budget_vectors_{timestamp}.pt"
    torch.save(results.pareto_vectors, vectors_path)

    # Create visualization
    try:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        all_refusal = np.array(to_list(results.scores_observed["refusal_score"]))
        all_kl = np.array(to_list(results.scores_observed["kl_score"]))
        pareto_refusal = np.array(to_list(results.pareto_scores["refusal_score"]))
        pareto_kl = np.array(to_list(results.pareto_scores["kl_score"]))

        n_total = len(all_refusal)
        pareto_mask = np.zeros(n_total, dtype=bool)
        for pr, pk in zip(pareto_refusal, pareto_kl):
            for i, (ar, ak) in enumerate(zip(all_refusal, all_kl)):
                if abs(ar - pr) < 0.01 and abs(ak - pk) < 0.01:
                    pareto_mask[i] = True
                    break

        # Plot 1: Objective space
        ax1 = axes[0]
        ax1.scatter(
            all_refusal[~pareto_mask],
            all_kl[~pareto_mask],
            c="lightgray",
            alpha=0.5,
            s=20,
            label="Dominated",
        )
        ax1.scatter(
            all_refusal[pareto_mask],
            all_kl[pareto_mask],
            c="red",
            s=100,
            marker="*",
            label="Pareto",
            zorder=5,
        )
        pareto_pairs = sorted(zip(pareto_refusal, pareto_kl))
        if pareto_pairs:
            pr_sorted, pk_sorted = zip(*pareto_pairs)
            ax1.plot(pr_sorted, pk_sorted, "r--", alpha=0.7, linewidth=2)
        ax1.set_xlabel("Refusal Score (lower = better)")
        ax1.set_ylabel("KL Score (lower = better)")
        ax1.set_title(
            f"HIGH BUDGET: {len(pareto_refusal)} Pareto, HV={results.hypervolume:.1f}"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: t-SNE
        ax2 = axes[1]
        V_all = results.V_observed.reshape(n_total, -1).numpy()
        if V_all.shape[1] > 50:
            V_pca = PCA(n_components=50).fit_transform(V_all)
        else:
            V_pca = V_all
        V_2d = TSNE(
            n_components=2, perplexity=min(30, n_total - 1), random_state=42
        ).fit_transform(V_pca)
        scatter = ax2.scatter(
            V_2d[~pareto_mask, 0],
            V_2d[~pareto_mask, 1],
            c=all_refusal[~pareto_mask],
            cmap="viridis",
            alpha=0.6,
            s=20,
        )
        ax2.scatter(
            V_2d[pareto_mask, 0],
            V_2d[pareto_mask, 1],
            c="red",
            s=100,
            marker="*",
            edgecolors="black",
            linewidths=1,
            zorder=5,
        )
        plt.colorbar(scatter, ax=ax2, label="Refusal Score")
        ax2.set_title("t-SNE (colored by refusal)")

        # Plot 3: PCA on Pareto vectors only (more meaningful)
        ax3 = axes[2]
        V_pareto = V_all[pareto_mask]
        pca_pareto = PCA(n_components=min(2, len(V_pareto) - 1))
        V_pareto_pca = pca_pareto.fit_transform(V_pareto)
        var_explained = pca_pareto.explained_variance_ratio_[:2].sum() * 100

        # Project dominated points using Pareto-fitted PCA
        V_dom_pca = pca_pareto.transform(V_all[~pareto_mask])

        scatter = ax3.scatter(
            V_dom_pca[:, 0],
            V_dom_pca[:, 1],
            c=all_kl[~pareto_mask],
            cmap="plasma",
            alpha=0.3,
            s=15,
            label="Dominated",
        )
        ax3.scatter(
            V_pareto_pca[:, 0],
            V_pareto_pca[:, 1],
            c=pareto_kl,
            cmap="plasma",
            s=120,
            marker="*",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
            label="Pareto",
        )
        plt.colorbar(scatter, ax=ax3, label="KL Score")
        ax3.set_title(f"PCA (Pareto-fitted, {var_explained:.1f}% var)")
        ax3.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plot_path = f"/results/high_budget_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not create plot: {e}")

    print(f"\nResults saved to {results_path}")
    print(f"Vectors saved to {vectors_path}")
    results_volume.commit()

    return serializable_results
