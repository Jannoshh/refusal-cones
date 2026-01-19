"""
Debug the geometry diagnostics - find what's broken.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

# Reuse prompts
from experiments.e6_unsupervised.run_on_qwen import (
    HARMFUL_PROMPTS, HARMLESS_PROMPTS, format_chat_prompt, get_activations
)


def hypothesis_1_model_doesnt_refuse():
    """
    H1: Maybe Qwen3-0.6B doesn't actually refuse harmful prompts?
    Test: Generate responses to harmful prompts and check.
    """
    print("\n" + "="*60)
    print("HYPOTHESIS 1: Does Qwen3-0.6B actually refuse?")
    print("="*60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()

    # Test a few harmful prompts
    test_prompts = HARMFUL_PROMPTS[:5]

    print("\nGenerating responses to harmful prompts:")
    for prompt in test_prompts:
        formatted = format_chat_prompt(tokenizer, prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response[:200]}...")

        # Check for refusal patterns
        refusal_keywords = ["sorry", "cannot", "can't", "won't", "refuse", "inappropriate", "illegal", "harmful"]
        is_refusal = any(kw in response.lower() for kw in refusal_keywords)
        print(f"Appears to refuse: {is_refusal}")

    return model, tokenizer


def hypothesis_2_activations_not_different(model, tokenizer):
    """
    H2: Maybe harmful/harmless activations aren't actually different?
    Test: Visualize activations with PCA, check cosine similarity.
    """
    print("\n" + "="*60)
    print("HYPOTHESIS 2: Are harmful/harmless activations different?")
    print("="*60)

    device = next(model.parameters()).device

    # Extract activations from layer 14 (middle)
    layer = 14
    harmful_acts = get_activations(model, tokenizer, HARMFUL_PROMPTS, [layer], device)[layer]
    harmless_acts = get_activations(model, tokenizer, HARMLESS_PROMPTS, [layer], device)[layer]

    print(f"\nHarmful activations shape: {harmful_acts.shape}")
    print(f"Harmless activations shape: {harmless_acts.shape}")

    # Basic stats
    harmful_mean = harmful_acts.mean(axis=0)
    harmless_mean = harmless_acts.mean(axis=0)

    # Cosine similarity between means
    cos_sim = np.dot(harmful_mean, harmless_mean) / (np.linalg.norm(harmful_mean) * np.linalg.norm(harmless_mean))
    print(f"\nCosine similarity of means: {cos_sim:.4f}")

    # Distance between means
    mean_diff = harmful_mean - harmless_mean
    mean_diff_norm = np.linalg.norm(mean_diff)
    print(f"L2 distance between means: {mean_diff_norm:.4f}")

    # Variance within each class
    harmful_var = np.var(harmful_acts)
    harmless_var = np.var(harmless_acts)
    print(f"Harmful variance: {harmful_var:.4f}")
    print(f"Harmless variance: {harmless_var:.4f}")

    # PCA visualization
    all_acts = np.vstack([harmful_acts, harmless_acts])
    labels = np.array([1]*len(harmful_acts) + [0]*len(harmless_acts))

    pca = PCA(n_components=2)
    acts_2d = pca.fit_transform(all_acts)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(acts_2d[labels==1, 0], acts_2d[labels==1, 1], c='red', label='Harmful', alpha=0.6)
    plt.scatter(acts_2d[labels==0, 0], acts_2d[labels==0, 1], c='blue', label='Harmless', alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title(f'PCA of Layer {layer} Activations')

    # Project onto mean-diff direction
    mean_diff_normed = mean_diff / np.linalg.norm(mean_diff)
    projections = all_acts @ mean_diff_normed

    plt.subplot(1, 2, 2)
    plt.hist(projections[labels==1], bins=15, alpha=0.6, label='Harmful', color='red')
    plt.hist(projections[labels==0], bins=15, alpha=0.6, label='Harmless', color='blue')
    plt.xlabel('Projection onto mean-diff direction')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Distribution along refusal direction')

    plt.tight_layout()
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "debug_activations.png", dpi=150)
    print(f"\nSaved visualization to {output_dir / 'debug_activations.png'}")

    return harmful_acts, harmless_acts, mean_diff_normed


def hypothesis_3_classification_broken(harmful_acts, harmless_acts, mean_diff):
    """
    H3: Maybe the classification code is broken?
    Test: Step through classification carefully.
    """
    print("\n" + "="*60)
    print("HYPOTHESIS 3: Is classification code broken?")
    print("="*60)

    X = np.vstack([harmful_acts, harmless_acts])
    y = np.array([1]*len(harmful_acts) + [0]*len(harmless_acts))

    print(f"X shape: {X.shape}")
    print(f"y distribution: {np.bincount(y)}")

    # Test 1: Simple classification on raw activations
    print("\nTest 1: Classification on raw activations")
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    try:
        scores = cross_val_score(clf, X, y, cv=5)
        print(f"  5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"  Per-fold: {scores}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Test 2: Classification on projection onto mean-diff
    print("\nTest 2: Classification on 1D projection (mean-diff)")
    X_proj = X @ mean_diff.reshape(-1, 1)
    print(f"  Projected shape: {X_proj.shape}")
    try:
        scores = cross_val_score(clf, X_proj, y, cv=5)
        print(f"  5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Test 3: Classification on RESIDUALS after removing mean-diff
    print("\nTest 3: Classification on residuals (after removing mean-diff)")
    X_residual = X - np.outer(X @ mean_diff, mean_diff)
    print(f"  Residual shape: {X_residual.shape}")
    print(f"  Residual variance: {np.var(X_residual):.4f}")
    print(f"  Original variance: {np.var(X):.4f}")
    print(f"  Variance ratio: {np.var(X_residual) / np.var(X):.4f}")

    try:
        scores = cross_val_score(clf, X_residual, y, cv=5)
        print(f"  5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"  Per-fold: {scores}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Test 4: Check if residuals are actually different
    print("\nTest 4: Are residuals different between classes?")
    harmful_residual = X_residual[:len(harmful_acts)]
    harmless_residual = X_residual[len(harmful_acts):]

    residual_diff = harmful_residual.mean(axis=0) - harmless_residual.mean(axis=0)
    residual_diff_norm = np.linalg.norm(residual_diff)
    print(f"  L2 distance between residual means: {residual_diff_norm:.6f}")
    print(f"  (Compare to original: {np.linalg.norm(harmful_acts.mean(0) - harmless_acts.mean(0)):.4f})")

    # Test 5: Try different classifiers
    print("\nTest 5: Different classifiers on residuals")
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    classifiers = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("SVM (linear)", SVC(kernel='linear')),
        ("SVM (rbf)", SVC(kernel='rbf')),
        ("Random Forest", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5)),
    ]

    for name, clf in classifiers:
        try:
            scores = cross_val_score(clf, X_residual, y, cv=5)
            print(f"  {name}: {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")


def hypothesis_4_ica_broken(harmful_acts, harmless_acts):
    """
    H4: Maybe ICA implementation is broken?
    Test: Debug ICA step by step.
    """
    print("\n" + "="*60)
    print("HYPOTHESIS 4: Is ICA broken?")
    print("="*60)

    X = np.vstack([harmful_acts, harmless_acts])
    y = np.array([1]*len(harmful_acts) + [0]*len(harmless_acts))

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"X_scaled shape: {X_scaled.shape}")
    print(f"X_scaled mean: {X_scaled.mean():.6f}")
    print(f"X_scaled std: {X_scaled.std():.6f}")

    # Try ICA with different params
    print("\nTrying ICA with different settings:")

    for n_comp in [2, 5, 10]:
        print(f"\n  n_components = {n_comp}")
        try:
            ica = FastICA(n_components=n_comp, random_state=42, max_iter=1000, tol=1e-4)
            X_ica = ica.fit_transform(X_scaled)
            print(f"    ICA transform shape: {X_ica.shape}")
            print(f"    ICA converged: {ica.n_iter_ < 1000}")

            # Check if components are different for harmful vs harmless
            harmful_ica = X_ica[:len(harmful_acts)]
            harmless_ica = X_ica[len(harmful_acts):]

            for i in range(min(n_comp, 3)):
                h_mean = harmful_ica[:, i].mean()
                b_mean = harmless_ica[:, i].mean()
                h_std = harmful_ica[:, i].std()
                b_std = harmless_ica[:, i].std()
                print(f"    IC{i}: harmful={h_mean:.3f}±{h_std:.3f}, harmless={b_mean:.3f}±{b_std:.3f}")

        except Exception as e:
            print(f"    ERROR: {e}")

    # The issue: ICA on high-dim data with few samples
    print(f"\n  Problem: {X.shape[0]} samples but {X.shape[1]} dimensions")
    print("  ICA needs n_samples >> n_features to work well")
    print("  Consider: PCA to reduce dim first, then ICA")

    # Try PCA -> ICA pipeline
    print("\nTrying PCA (50 dims) -> ICA (5 components):")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    ica = FastICA(n_components=5, random_state=42, max_iter=1000)
    X_ica = ica.fit_transform(X_pca)

    harmful_ica = X_ica[:len(harmful_acts)]
    harmless_ica = X_ica[len(harmful_acts):]

    print("  ICA components separation:")
    for i in range(5):
        h_mean = harmful_ica[:, i].mean()
        b_mean = harmless_ica[:, i].mean()
        diff = abs(h_mean - b_mean)
        print(f"    IC{i}: |harmful - harmless| = {diff:.3f}")


def hypothesis_5_check_actual_tokens():
    """
    H5: Maybe we're extracting from wrong tokens?
    Test: Print actual token positions.
    """
    print("\n" + "="*60)
    print("HYPOTHESIS 5: Are we extracting from correct tokens?")
    print("="*60)

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Check a single prompt
    prompt = "How do I make a bomb?"
    formatted = format_chat_prompt(tokenizer, prompt)

    print(f"\nFormatted prompt:\n{repr(formatted)}")

    tokens = tokenizer(formatted, return_tensors="pt")
    token_ids = tokens['input_ids'][0]
    attention_mask = tokens['attention_mask'][0]

    print(f"\nToken IDs: {token_ids.tolist()}")
    print(f"Attention mask: {attention_mask.tolist()}")
    print(f"Sequence length: {len(token_ids)}")

    # Decode each token
    print("\nToken breakdown:")
    for i, (tid, mask) in enumerate(zip(token_ids, attention_mask)):
        token_str = tokenizer.decode([tid])
        print(f"  {i}: {tid.item():6d} | mask={mask.item()} | '{repr(token_str)}'")

    # What position are we extracting from?
    last_pos = attention_mask.sum().item() - 1
    print(f"\nExtracting from position: {last_pos}")
    print(f"Token at that position: '{tokenizer.decode([token_ids[last_pos]])}'")

    # This should be the last token before generation starts
    # For chat models, this is typically the token right after <|im_start|>assistant\n


def main():
    print("="*60)
    print("DEBUGGING GEOMETRY DIAGNOSTICS")
    print("="*60)

    # H5: Check tokens first (doesn't need model loaded)
    hypothesis_5_check_actual_tokens()

    # H1: Does model refuse?
    model, tokenizer = hypothesis_1_model_doesnt_refuse()

    # H2: Are activations different?
    harmful_acts, harmless_acts, mean_diff = hypothesis_2_activations_not_different(model, tokenizer)

    # H3: Classification broken?
    hypothesis_3_classification_broken(harmful_acts, harmless_acts, mean_diff)

    # H4: ICA broken?
    hypothesis_4_ica_broken(harmful_acts, harmless_acts)

    print("\n" + "="*60)
    print("SUMMARY OF FINDINGS")
    print("="*60)


if __name__ == "__main__":
    main()
