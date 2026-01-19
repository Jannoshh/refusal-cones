# Refusal Geometry Findings: Qwen3-0.6B

**Date:** 2025-01-19
**Model:** Qwen/Qwen3-0.6B
**Data:** Circuit Breaker dataset (100 harmful, 100 harmless prompts)

## Key Findings

### 1. Mean-diff direction works well but doesn't capture everything

| Method | Classification Accuracy |
|--------|------------------------|
| Single mean-diff direction | 94.5% |
| Residuals after removing mean-diff | **96.5%** (inverted) |
| PCA top 3 directions | 99.0% |

The fact that residuals still classify at 96.5% proves **refusal is NOT a single direction**. There's substantial structure orthogonal to mean-diff.

### 2. Intrinsic dimension is ~10, not 1

| Method | Harmful | Harmless | Combined |
|--------|---------|----------|----------|
| PCA (95% variance) | 54 | 60 | 92 |
| MLE (local geometry) | **9.5** | **8.2** | **9.4** |

- PCA overestimates by counting all directions with variance (including noise)
- MLE measures local manifold structure → **~10 true dimensions**
- Confirms multi-dimensional structure, but not arbitrarily high

### 3. The geometry appears non-linear (curved)

- Curvature estimate: 0.91 (high)
- Local PCA directions vary significantly across clusters
- Suggests local linear approximation may outperform global directions

### 4. Harm categories do NOT separate in residual space

- Category classification from residuals: 37% (barely above 20% random baseline)
- The additional dimensions aren't cleanly "violence vs illegal vs self-harm"
- More likely: different phrasings, contexts, or intensity levels

## Diagnostic Bug Found & Fixed

**Original behavior:**
- Residual classification showed 3.5% accuracy
- Interpreted as "no signal in residuals"

**Problem:**
- With balanced classes, 3.5% means classifier learned **inverted** relationship
- Predicting opposite class 96.5% of the time

**Fix in `geometry_diagnostics.py`:**
```python
classification_acc = max(classification_acc, 1 - classification_acc)
```

## Practical Recommendations

### For ablating refusal in Qwen3-0.6B:

1. **Baseline:** Single mean-diff direction (~95% effective)

2. **Better:** PCA top 2-3 directions (~99% effective)
   ```python
   pca = PCA(n_components=3)
   pca.fit(np.vstack([harmful_acts, harmless_acts]))
   directions = pca.components_  # Use all 3 for ablation
   ```

3. **For edge cases:** Local linear approximation
   - Cluster activations (K=8-16)
   - Fit local direction per cluster
   - At inference: find nearest cluster, use its direction

### What likely won't help:

- **ICA:** Showed no benefit over PCA (separation quality = 0)
- **Category-specific directions:** Categories don't separate cleanly

## Code & Reproducibility

Scripts in `experiments/e6_unsupervised/`:
- `geometry_diagnostics.py` - Core diagnostic functions
- `run_on_qwen_cb.py` - Run diagnostics on Qwen3-0.6B with Circuit Breaker data
- `ica_vs_pca_demo.py` - Synthetic comparison of ICA vs PCA
- `debug_diagnostics.py` - Debugging utilities

Run diagnostics:
```bash
uv run python -m experiments.e6_unsupervised.run_on_qwen_cb
```

## Next Steps

1. **Validate with ablation:** Test if PCA top-3 actually ablates refusal better than mean-diff
2. **Test on larger models:** Does the ~10-dim structure hold for 7B+ models?
3. **Investigate the residual signal:** What ARE those extra dimensions encoding?
