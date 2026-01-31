"""
Boundary-based discovery functions for Modal.

Contains run_boundary_then_pareto, run_boundary_only, and run_pareto_from_cache.
"""

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
    setup_modal_environment,
    to_list,
)


def compute_baseline_scores(
    model_obj,
    tokenizer,
    prompts: list[str],
    refusal_toks: torch.Tensor,
    batch_size: int = 8,
) -> torch.Tensor:
    """Compute refusal scores without any ablation."""
    from src.measurement.scoring import refusal_score_fn

    scores = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        # Apply chat template for instruction-tuned models
        if tokenizer.chat_template is not None:
            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch
            ]
        else:
            formatted = batch
        inputs = tokenizer(formatted, padding=True, return_tensors="pt").to(model_obj.device)
        with torch.no_grad():
            outputs = model_obj(**inputs)
        # Get last token logits for each sequence
        last_positions = inputs.attention_mask.sum(dim=1) - 1
        batch_logits = outputs.logits[
            torch.arange(len(batch), device=model_obj.device), last_positions
        ]
        batch_scores = refusal_score_fn(batch_logits, refusal_toks)
        # Debug: check for NaN
        if torch.isnan(batch_scores).any():
            print(f"    WARNING: NaN scores detected in batch {i // batch_size}")
            print(f"    Logits range: [{batch_logits.min():.2f}, {batch_logits.max():.2f}]")
            print(
                f"    refusal_toks device: {refusal_toks.device}, "
                f"logits device: {batch_logits.device}"
            )
        scores.extend(batch_scores.tolist())
    return torch.tensor(scores)


@app.function(
    image=cuda_image,
    gpu=GPU_CONFIG,
    volumes={
        "/cache": model_cache,
        "/results": results_volume,
    },
    timeout=7200,  # 2 hours for two-phase discovery
)
def run_boundary_then_pareto(
    model: str = "Qwen/Qwen3-0.6B",
    # Phase 1: Boundary discovery
    n_boundary_init: int = 30,
    n_boundary_iterations: int = 50,
    boundary_threshold: float = 0.0,  # Fixed threshold (used if use_dynamic_threshold=False)
    use_dynamic_threshold: bool = True,  # Compute threshold from baseline
    boundary_fraction: float = 0.05,  # Fraction of gap from refusal to non-refusal (5%)
    # Phase 2: Pareto search inside boundary
    n_pareto_init: int = 20,
    n_pareto_iterations: int = 50,
    # Common settings - default to 8 for faster debugging
    n_harmful: int = 8,
    n_harmless: int = 8,
    use_gradient_candidates: bool = False,  # Disable by default - much faster
) -> dict:
    """
    Run two-phase discovery: boundary first, then Pareto inside.

    Phase 1: Boundary Discovery
    - Find the boundary of the refusal space where ablation starts working
    - Uses straddle acquisition to explore where refusal_score crosses threshold
    - Identifies which vectors are "inside" (working) vs "outside" (not working)

    Phase 2: Pareto Search (constrained)
    - Only search among vectors inside the boundary
    - Find Pareto-optimal trade-offs between refusal_score and kl_score
    - All vectors are guaranteed to have refusal_score below threshold

    This two-phase approach ensures:
    1. We understand the shape of the working region
    2. We only optimize within the region that matters
    3. No wasted samples in regions where ablation doesn't work

    Args:
        model: HuggingFace model name
        n_boundary_init: Initial samples for boundary phase
        n_boundary_iterations: Iterations for boundary search
        boundary_threshold: Threshold defining "inside" boundary (refusal_score < threshold)
        use_dynamic_threshold: Whether to compute threshold dynamically from baseline
        boundary_fraction: Fraction of gap from refusal to non-refusal for dynamic threshold
        n_pareto_init: Initial samples for Pareto phase
        n_pareto_iterations: Iterations for Pareto search
        n_harmful: Number of harmful prompts
        n_harmless: Number of harmless prompts
        use_gradient_candidates: Whether to use gradient-based candidate generation
    """
    import json
    import time
    import traceback
    from datetime import datetime

    import matplotlib
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_modal_environment()

    from src.discovery.pareto_boundary_discovery import (
        BoundaryThenParetoConfig,
        BoundaryThenParetoDiscovery,
        MultiObjectiveScorer,
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

    # Load prompts (load more than needed, then filter)
    harmful_prompts, harmful_targets, harmless_prompts = load_prompts(
        n_harmful=n_harmful * 4, n_harmless=n_harmless * 4  # Load extra for filtering
    )

    # Refusal tokens - put on CUDA to match logits device
    refusal_tokens = ["I", "Sorry", "I'm", "As", "I cannot"]
    refusal_toks = tokenizer(refusal_tokens, add_special_tokens=False)["input_ids"]
    refusal_toks = torch.tensor(
        [t[0] for t in refusal_toks if len(t) > 0], device=model_obj.device
    )

    # Filter prompts: only keep ones the model actually refuses/doesn't refuse
    print("Filtering prompts based on baseline refusal scores...")

    # Debug: show refusal tokens
    print(f"  Refusal tokens: {refusal_tokens}")
    print(f"  Refusal token IDs: {refusal_toks.tolist()}")
    # Decode them to verify
    for tok_id in refusal_toks.tolist():
        print(f"    {tok_id} -> '{tokenizer.decode([tok_id])}'")

    # Debug: generate 10 tokens to see what model outputs
    print("\n  Testing generation on first harmful prompt:")
    test_prompt = harmful_prompts[0]
    print(f"    Prompt: {test_prompt[:80]}...")
    if tokenizer.chat_template is not None:
        test_formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": test_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        test_formatted = test_prompt
    test_inputs = tokenizer(test_formatted, return_tensors="pt").to(model_obj.device)
    with torch.no_grad():
        test_outputs = model_obj.generate(
            **test_inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    test_response = tokenizer.decode(
        test_outputs[0][test_inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    print(f"    Response (10 tokens): {test_response}")

    harmful_baseline_scores = compute_baseline_scores(
        model_obj, tokenizer, harmful_prompts, refusal_toks
    )
    harmless_baseline_scores = compute_baseline_scores(
        model_obj, tokenizer, harmless_prompts, refusal_toks
    )

    # Debug: show score distribution
    print(
        f"  Harmful scores: min={harmful_baseline_scores.min():.3f}, "
        f"max={harmful_baseline_scores.max():.3f}, mean={harmful_baseline_scores.mean():.3f}"
    )
    print(
        f"  Harmless scores: min={harmless_baseline_scores.min():.3f}, "
        f"max={harmless_baseline_scores.max():.3f}, mean={harmless_baseline_scores.mean():.3f}"
    )
    print(f"  First 5 harmful scores: {harmful_baseline_scores[:5].tolist()}")

    # Filter: keep harmful where model refuses (score > 0), harmless where doesn't (score < 0)
    harmful_mask = harmful_baseline_scores > 0
    harmless_mask = harmless_baseline_scores < 0

    n_harmful_kept = harmful_mask.sum().item()
    n_harmless_kept = harmless_mask.sum().item()
    print(
        f"  Harmful: {n_harmful_kept}/{len(harmful_prompts)} prompts "
        f"where model refuses (score > 0)"
    )
    print(
        f"  Harmless: {n_harmless_kept}/{len(harmless_prompts)} prompts "
        f"where model doesn't refuse (score < 0)"
    )

    # Apply filter
    harmful_prompts = [
        p for p, keep in zip(harmful_prompts, harmful_mask.tolist()) if keep
    ][:n_harmful]
    harmful_targets = [
        t for t, keep in zip(harmful_targets, harmful_mask.tolist()) if keep
    ][:n_harmful]
    harmless_prompts = [
        p for p, keep in zip(harmless_prompts, harmless_mask.tolist()) if keep
    ][:n_harmless]

    if len(harmful_prompts) < n_harmful:
        print(
            f"  WARNING: Only {len(harmful_prompts)} harmful prompts pass filter "
            f"(requested {n_harmful})"
        )
    if len(harmless_prompts) < n_harmless:
        print(
            f"  WARNING: Only {len(harmless_prompts)} harmless prompts pass filter "
            f"(requested {n_harmless})"
        )

    print(
        f"Using {len(harmful_prompts)} harmful and {len(harmless_prompts)} "
        f"harmless prompts (filtered)"
    )

    # Create scorer
    print("Creating multi-objective scorer...")
    # Only pass harmful_completions when using gradients (for ablation_loss)
    # Without gradients, single-token refusal_score is sufficient and faster
    scorer = MultiObjectiveScorer(
        model=model_obj,
        tokenizer=tokenizer,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
        refusal_toks=refusal_toks,
        device="cuda",
        harmful_completions=harmful_targets if use_gradient_candidates else None,
    )

    # Configure two-phase discovery
    config = BoundaryThenParetoConfig(
        # Phase 1: Boundary
        n_boundary_init=n_boundary_init,
        n_boundary_iterations=n_boundary_iterations,
        boundary_threshold=boundary_threshold,
        use_dynamic_threshold=use_dynamic_threshold,
        boundary_fraction=boundary_fraction,
        # Phase 2: Pareto
        n_pareto_init=n_pareto_init,
        n_pareto_iterations=n_pareto_iterations,
        # Common
        gp_type="structured",
        layer_lengthscale=3.0,
        use_gradient_candidates=use_gradient_candidates,
        gradient_steps=5,
        gradient_lr=0.1,
    )

    print(f"\n{'=' * 70}")
    print("TWO-PHASE DISCOVERY: Boundary -> Pareto")
    print(f"{'=' * 70}")
    print(f"Phase 1: Boundary discovery (threshold={boundary_threshold})")
    print(f"  - {n_boundary_init} init samples + {n_boundary_iterations} iterations")
    print("Phase 2: Pareto search inside boundary")
    print(f"  - {n_pareto_init} init samples + {n_pareto_iterations} iterations")
    print(f"{'=' * 70}\n")

    # Compute mean-diff initialization (like ACE)
    print("Computing mean-diff initialization...")
    v_init, v_minus, v_plus = scorer.compute_mean_diff_vector()
    print(f"  v_init shape: {v_init.shape}")
    print(
        f"  v_init per-layer norms: min={v_init.norm(dim=-1).min():.4f}, "
        f"max={v_init.norm(dim=-1).max():.4f}"
    )
    # Cache v_minus/v_plus in scorer for ACE
    scorer._v_minus = v_minus
    scorer._v_plus = v_plus
    scorer._v_minus_computed = True

    # Run discovery
    start_time = time.time()

    discovery = BoundaryThenParetoDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        v_init=v_init,
    )

    results = discovery.discover()
    wall_time = time.time() - start_time

    # Print summary
    print(f"\n{'=' * 70}")
    print("TWO-PHASE DISCOVERY COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time: {wall_time:.1f}s ({wall_time / 60:.1f} min)")
    print("\nPhase 1 (Boundary):")
    print(f"  Measurements: {results.n_boundary_measurements}")
    print(f"  Boundary points: {len(results.boundary_points)}")
    print(f"  Inside points: {len(results.inside_points)}")
    print(f"  Outside points: {len(results.outside_points)}")
    print("\nPhase 2 (Pareto):")
    print(f"  Measurements: {results.n_pareto_measurements}")
    print(f"  Hypervolume: {results.hypervolume:.4f}")
    print(f"  Pareto vectors: {len(results.pareto_vectors)}")
    if len(results.pareto_vectors) > 0:
        print(f"  Best refusal: {results.pareto_scores['refusal_score'].min():.4f}")
        print(f"  Best KL: {results.pareto_scores['kl_score'].min():.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    serializable_results = {
        "total_time": wall_time,
        "boundary": {
            "n_measurements": results.n_boundary_measurements,
            "n_boundary_points": len(results.boundary_points),
            "n_inside_points": len(results.inside_points),
            "n_outside_points": len(results.outside_points),
            "threshold": boundary_threshold,
        },
        "pareto": {
            "n_measurements": results.n_pareto_measurements,
            "hypervolume": float(results.hypervolume),
            "n_pareto_vectors": len(results.pareto_vectors),
            "pareto_scores": {k: to_list(v) for k, v in results.pareto_scores.items()},
        },
        "model": model,
        "config": {
            "n_boundary_init": n_boundary_init,
            "n_boundary_iterations": n_boundary_iterations,
            "boundary_threshold": boundary_threshold,
            "n_pareto_init": n_pareto_init,
            "n_pareto_iterations": n_pareto_iterations,
        },
    }

    results_path = f"/results/boundary_pareto_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    # Save vectors
    if len(results.pareto_vectors) > 0:
        vectors_path = f"/results/boundary_pareto_vectors_{timestamp}.pt"
        torch.save(results.pareto_vectors, vectors_path)
        print(f"Vectors saved to {vectors_path}")

    # Create visualization
    try:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Boundary phase results (all observations)
        ax1 = axes[0]
        all_refusal = np.array(to_list(results.scores_observed["refusal_score"]))
        all_kl = np.array(to_list(results.scores_observed["kl_score"]))

        inside_mask = all_refusal < boundary_threshold
        ax1.scatter(
            all_refusal[~inside_mask],
            all_kl[~inside_mask],
            c="gray",
            alpha=0.5,
            s=30,
            label=f"Outside (refusal >= {boundary_threshold})",
        )
        ax1.scatter(
            all_refusal[inside_mask],
            all_kl[inside_mask],
            c="green",
            alpha=0.7,
            s=50,
            label=f"Inside (refusal < {boundary_threshold})",
        )
        ax1.axvline(
            x=boundary_threshold, color="red", linestyle="--", label="Boundary threshold"
        )
        ax1.set_xlabel("Refusal Score")
        ax1.set_ylabel("KL Score")
        ax1.set_title(
            f"All Observations\n{inside_mask.sum()} inside, {(~inside_mask).sum()} outside"
        )
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Pareto results (if available)
        ax2 = axes[1]
        if len(results.pareto_vectors) > 0:
            pareto_refusal = np.array(to_list(results.pareto_scores["refusal_score"]))
            pareto_kl = np.array(to_list(results.pareto_scores["kl_score"]))

            # Find Pareto mask
            n_total = len(all_refusal)
            pareto_mask = np.zeros(n_total, dtype=bool)
            for pr, pk in zip(pareto_refusal, pareto_kl):
                for i, (ar, ak) in enumerate(zip(all_refusal, all_kl)):
                    if abs(ar - pr) < 0.01 and abs(ak - pk) < 0.01:
                        pareto_mask[i] = True
                        break

            # Plot inside-boundary points
            inside_not_pareto = inside_mask & ~pareto_mask
            ax2.scatter(
                all_refusal[inside_not_pareto],
                all_kl[inside_not_pareto],
                c="lightblue",
                alpha=0.5,
                s=30,
                label="Dominated",
            )
            ax2.scatter(
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
                pr_sorted, pk_sorted = zip(*pareto_pairs)
                ax2.plot(pr_sorted, pk_sorted, "r--", alpha=0.7, linewidth=2)

            ax2.axvline(x=boundary_threshold, color="gray", linestyle=":", alpha=0.5)
            ax2.set_xlabel("Refusal Score")
            ax2.set_ylabel("KL Score")
            ax2.set_title(
                f"Pareto Search (inside boundary)\n"
                f"{len(pareto_refusal)} Pareto, HV={results.hypervolume:.1f}"
            )
            ax2.legend(loc="upper right")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "No Pareto vectors found\n(no points inside boundary?)",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=14,
            )
            ax2.set_title("Pareto Search")

        plt.tight_layout()
        plot_path = f"/results/boundary_pareto_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not create plot: {e}")
        traceback.print_exc()

    print(f"\nResults saved to {results_path}")
    results_volume.commit()

    return serializable_results


@app.function(
    image=cuda_image,
    gpu=GPU_CONFIG,
    volumes={
        "/cache": model_cache,
        "/results": results_volume,
    },
    timeout=3600,
)
def run_boundary_only(
    model: str = "Qwen/Qwen2-1.5B-Instruct",
    n_boundary_init: int = 30,
    n_boundary_iterations: int = 50,
    boundary_threshold: float = 0.0,
    use_dynamic_threshold: bool = True,  # Compute threshold from baseline
    boundary_fraction: float = 0.05,  # Fraction of gap from refusal to non-refusal (5%)
    n_harmful: int = 8,
    n_harmless: int = 8,
    run_id: str | None = None,  # Optional ID to identify this run
) -> dict:
    """
    Run ONLY boundary discovery (Phase 1) and cache results.

    This allows debugging boundary discovery separately from Pareto search.
    Results are saved to /results/boundary_cache_{run_id}.pt

    Use run_pareto_from_cache to continue with Phase 2.
    """
    import json
    import time
    from datetime import datetime

    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_modal_environment()

    from src.discovery.pareto_boundary_discovery import (
        BoundaryThenParetoConfig,
        BoundaryThenParetoDiscovery,
        MultiObjectiveScorer,
    )

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"=== BOUNDARY DISCOVERY ONLY (run_id={run_id}) ===")
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
    if hasattr(model_obj, "config"):
        hidden_dim = model_obj.config.hidden_size
        n_layers = model_obj.config.num_hidden_layers
    else:
        hidden_dim = model_obj.model.config.hidden_size
        n_layers = model_obj.model.config.num_hidden_layers

    print(f"Model: hidden_dim={hidden_dim}, n_layers={n_layers}")

    # Load prompts
    harmful_prompts, harmful_targets, harmless_prompts = load_prompts(
        n_harmful=n_harmful * 4, n_harmless=n_harmless * 4
    )

    # Refusal tokens
    refusal_tokens = ["I", "Sorry", "I'm", "As", "I cannot"]
    refusal_toks = tokenizer(refusal_tokens, add_special_tokens=False)["input_ids"]
    refusal_toks = torch.tensor(
        [t[0] for t in refusal_toks if len(t) > 0], device=model_obj.device
    )

    print(f"Using {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts")

    # Create scorer
    scorer = MultiObjectiveScorer(
        model=model_obj,
        tokenizer=tokenizer,
        harmful_prompts=harmful_prompts[:n_harmful],
        harmless_prompts=harmless_prompts[:n_harmless],
        refusal_toks=refusal_toks,
        device="cuda",
    )

    # Compute initialization
    print("Computing mean-diff initialization...")
    v_init, v_minus, v_plus = scorer.compute_mean_diff_vector()
    scorer._v_minus = v_minus
    scorer._v_plus = v_plus
    scorer._v_minus_computed = True

    # Compute dynamic threshold if enabled
    if use_dynamic_threshold:
        baseline_refusal, baseline_harmless = scorer.compute_baseline_refusal_scores()
        gap = baseline_refusal - baseline_harmless
        actual_threshold = baseline_refusal - boundary_fraction * gap
        print(f"Dynamic threshold: refusal={baseline_refusal:.3f}, harmless={baseline_harmless:.3f}")
        print(f"  Gap={gap:.3f}, fraction={boundary_fraction:.0%}")
        print(
            f"  Boundary = {baseline_refusal:.3f} - {boundary_fraction:.0%}*{gap:.3f} "
            f"= {actual_threshold:.3f}"
        )
    else:
        actual_threshold = boundary_threshold
        print(f"Fixed boundary threshold: {actual_threshold}")

    # Configure for boundary-only (use two-phase config but skip Pareto)
    config = BoundaryThenParetoConfig(
        # Boundary phase
        n_boundary_init=n_boundary_init,
        n_boundary_iterations=n_boundary_iterations,
        boundary_threshold=actual_threshold,
        use_dynamic_threshold=False,  # Already computed above
        # Pareto phase (minimal, we'll skip it)
        n_pareto_init=0,
        n_pareto_iterations=0,
        # Common
        gp_type="structured",
        layer_lengthscale=3.0,
    )

    # Run boundary discovery only
    start_time = time.time()

    discovery = BoundaryThenParetoDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        v_init=v_init,
    )

    # Evaluate v_init first (matches discover() behavior)
    print("Evaluating initial vector (difference-in-means)...")
    v_init_normed = v_init / (v_init.norm() + 1e-8)
    scores = scorer.score(v_init_normed)
    discovery.V_observed.append(v_init_normed.detach().cpu())
    for obj, val in scores.items():
        discovery.scores_observed[obj].append(val)
    print(f"  v_init refusal={scores['refusal_score']:.4f}, kl={scores.get('kl_score', 0):.4f}")

    # Run only boundary phase (internal method)
    boundary_points, inside_points, outside_points = discovery._boundary_phase()
    wall_time = time.time() - start_time

    # Create results struct
    class BoundaryResults:
        pass

    boundary_results = BoundaryResults()
    boundary_results.boundary_points = boundary_points
    boundary_results.inside_points = inside_points
    boundary_results.outside_points = outside_points
    boundary_results.V_observed = discovery.V_observed
    boundary_results.scores_observed = discovery.scores_observed
    boundary_results.n_measurements = len(discovery.V_observed)

    # Print summary
    print(f"\n{'=' * 70}")
    print("BOUNDARY DISCOVERY COMPLETE")
    print(f"{'=' * 70}")
    print(f"Time: {wall_time:.1f}s")
    print(f"Total measurements: {boundary_results.n_measurements}")
    print(f"Boundary points: {len(boundary_results.boundary_points)}")
    print(f"Inside (ablation works): {len(boundary_results.inside_points)}")
    print(f"Outside (ablation fails): {len(boundary_results.outside_points)}")

    # Save cache with all necessary data
    cache_data = {
        "v_init": v_init.cpu(),
        "v_minus": v_minus.cpu(),
        "v_plus": v_plus.cpu(),
        "boundary_points": [v.cpu() for v in boundary_results.boundary_points],
        "inside_points": [v.cpu() for v in boundary_results.inside_points],
        "outside_points": [v.cpu() for v in boundary_results.outside_points],
        "V_observed": [v.cpu() for v in boundary_results.V_observed],
        "scores_observed": boundary_results.scores_observed,
        "n_measurements": boundary_results.n_measurements,
        "boundary_threshold": actual_threshold,
        "model": model,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "wall_time": wall_time,
    }

    cache_path = f"/results/boundary_cache_{run_id}.pt"
    torch.save(cache_data, cache_path)
    print(f"\nBoundary cache saved to: {cache_path}")

    # Also save JSON summary
    summary = {
        "run_id": run_id,
        "model": model,
        "wall_time": wall_time,
        "n_measurements": boundary_results.n_measurements,
        "n_boundary_points": len(boundary_results.boundary_points),
        "n_inside_points": len(boundary_results.inside_points),
        "n_outside_points": len(boundary_results.outside_points),
        "boundary_threshold": actual_threshold,
        "config": {
            "n_boundary_init": n_boundary_init,
            "n_boundary_iterations": n_boundary_iterations,
            "use_dynamic_threshold": use_dynamic_threshold,
            "boundary_fraction": boundary_fraction,
        },
    }

    summary_path = f"/results/boundary_summary_{run_id}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    results_volume.commit()
    print(f"Summary saved to: {summary_path}")
    print("\nTo continue with Pareto search, run:")
    print(f"  modal run modal_app.py::run_pareto_from_cache --run-id {run_id}")

    return summary


@app.function(
    image=cuda_image,
    gpu=GPU_CONFIG,
    volumes={
        "/cache": model_cache,
        "/results": results_volume,
    },
    timeout=3600,
)
def run_pareto_from_cache(
    run_id: str,
    n_pareto_init: int = 20,
    n_pareto_iterations: int = 50,
) -> dict:
    """
    Run Pareto discovery (Phase 2) from cached boundary results.

    Loads boundary cache from run_boundary_only and continues with Pareto search.
    """
    import json
    import os
    import time
    from datetime import datetime

    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_modal_environment()

    from src.discovery.pareto_boundary_discovery import (
        MultiObjectiveScorer,
        ParetoDiscoveryConfig,
        ParetoGeometryDiscovery,
    )

    print(f"=== PARETO SEARCH FROM CACHE (run_id={run_id}) ===")

    # Load cached boundary results
    cache_path = f"/results/boundary_cache_{run_id}.pt"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Boundary cache not found: {cache_path}")

    print(f"Loading boundary cache from: {cache_path}")
    cache = torch.load(cache_path)

    model_name = cache["model"]
    n_layers = cache["n_layers"]
    hidden_dim = cache["hidden_dim"]
    boundary_threshold = cache["boundary_threshold"]

    print(f"Model: {model_name}")
    print(f"Dimensions: {n_layers} layers x {hidden_dim} hidden")
    print(f"Inside points from boundary: {len(cache['inside_points'])}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/cache/huggingface")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_obj = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/cache/huggingface",
    )

    # Load prompts (same as boundary phase)
    harmful_prompts, harmful_targets, harmless_prompts = load_prompts()

    # Refusal tokens
    refusal_toks = get_refusal_tokens(tokenizer).to(model_obj.device)

    # Create scorer
    scorer = MultiObjectiveScorer(
        model=model_obj,
        tokenizer=tokenizer,
        harmful_prompts=harmful_prompts[:8],
        harmless_prompts=harmless_prompts[:8],
        refusal_toks=refusal_toks,
        device="cuda",
    )

    # Restore cached activations
    scorer._v_minus = cache["v_minus"].to("cuda")
    scorer._v_plus = cache["v_plus"].to("cuda")
    scorer._v_minus_computed = True

    # Configure Pareto discovery
    config = ParetoDiscoveryConfig(
        n_init=n_pareto_init,
        n_iterations=n_pareto_iterations,
        gp_type="structured",
        layer_lengthscale=3.0,
        constraint_threshold=boundary_threshold,
    )

    # Initialize Pareto discovery with cached inside points
    inside_points = [v.to("cuda") for v in cache["inside_points"]]

    print(f"\n{'=' * 70}")
    print("PARETO SEARCH (constrained to boundary)")
    print(f"{'=' * 70}")
    print(f"Starting from {len(inside_points)} inside points")

    start_time = time.time()

    pareto_discovery = ParetoGeometryDiscovery(
        scorer=scorer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
        v_init=cache["v_init"].to("cuda"),
    )

    # Seed with cached inside points
    for v in inside_points:
        pareto_discovery.V_observed.append(v)
        scores = scorer.score(v)
        for obj, val in scores.items():
            if obj in pareto_discovery.scores_observed:
                pareto_discovery.scores_observed[obj].append(val)

    pareto_results = pareto_discovery.discover()
    wall_time = time.time() - start_time

    # Print summary
    print(f"\n{'=' * 70}")
    print("PARETO SEARCH COMPLETE")
    print(f"{'=' * 70}")
    print(f"Time: {wall_time:.1f}s")
    print(f"Total measurements: {pareto_results.n_measurements}")
    print(f"Pareto vectors: {len(pareto_results.pareto_vectors)}")
    print(f"Hypervolume: {pareto_results.hypervolume:.4f}")

    if len(pareto_results.pareto_vectors) > 0:
        print(f"Best refusal: {pareto_results.pareto_scores['refusal_score'].min():.4f}")
        print(f"Best KL: {pareto_results.pareto_scores['kl_score'].min():.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "run_id": run_id,
        "pareto_timestamp": timestamp,
        "model": model_name,
        "wall_time": wall_time,
        "boundary_wall_time": cache["wall_time"],
        "total_time": wall_time + cache["wall_time"],
        "n_pareto_measurements": pareto_results.n_measurements,
        "n_boundary_measurements": cache["n_measurements"],
        "hypervolume": float(pareto_results.hypervolume),
        "n_pareto_vectors": len(pareto_results.pareto_vectors),
        "pareto_scores": {k: to_list(v) for k, v in pareto_results.pareto_scores.items()},
        "boundary_threshold": boundary_threshold,
    }

    results_path = f"/results/pareto_results_{run_id}_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    if len(pareto_results.pareto_vectors) > 0:
        vectors_path = f"/results/pareto_vectors_{run_id}_{timestamp}.pt"
        torch.save(pareto_results.pareto_vectors, vectors_path)
        print(f"Vectors saved to: {vectors_path}")

    results_volume.commit()
    print(f"Results saved to: {results_path}")

    return results
