# Key Excerpts: Yu et al. (2505.21800) - Truth Cones

## Explicit Citation & Framework Extension

### Abstract
> "In this work, we extend the concept cone framework, recently introduced for modeling refusal, to the domain of truth. We identify multi-dimensional cones that causally mediate truth-related behavior across multiple LLM families."

---

### Section 1 - Introduction
> "Recent work has developed more sophisticated non-linear frameworks and found multiple latent dimensions that capture fundamental high-level concepts, **notably for refusal [Hildebrandt et al. 2025, Wollschläger et al. 2025].**"

> "Concept cones use a gradient-based search algorithm that, given candidate vectors, learns a specific behavior. Each vector is validated to causally influence the target concept through steering or ablation. This method extends the interpretability toolkit beyond linear assumptions by enabling both analysis and controlled intervention [Liu et al. 2023, **Wollschläger et al. 2025**]."

> "In this paper, we extend the concept cone framework to the domain of propositional fact, a subcategory of truthfulness, exploring how this property is internally represented by LLMs."

---

### Section 2.4 - Gradient-Based Methods (Direct Inheritance)
> "More recently, **Wollschläger et al. [2025] have used gradients to steer model behavior**: specific objectives, such as refusing unsafe inputs, can be encoded directly as loss functions. By optimizing a single vector that is added to or ablated from activations at specific layers, models can be guided toward target behaviors (e.g., safe refusals) while minimizing side effects on unrelated outputs. When applied to truthfulness, this framework enables precise, interpretable interventions and allows models to express truth-aligned responses without requiring full fine-tuning."

---

### Section 2.5 - Concept Cones (Formal Definition)
> "As described in **Wollschläger et al. [2025]**, given a set of orthonormal vectors V = [v₁, v₂, ..., vₖ] ∈ ℝ^(d_model×k), a matrix whose columns are vectors each exhibit truth properties. The cone is the set of all nonnegative linear combinations:

> C = {Σᵢ λᵢ·vᵢ | λᵢ ≥ 0} \ {0}

> All directions used in the cone correspond to the same truth concept. The constraint {λᵢ ≥ 0} ensures that all directions within the cone consistently strengthen truth behavior."

---

## Three-Term Loss Optimization (Adapted from Wollschläger)

### Section 3.3 - Loss-Guided Concept Cone Discovery

> "Following **Wollschläger et al. [2025]**, our optimisation target is a three-term loss:

> L_total = λ₁·L_add + λ₂·L_ablate + λ₃·L_retain

> but with two implementation tweaks that adapt it to binary truth-judgement:

> 1. **Binary generation.** At generation time we zero out every logit except the two tokens `Yes` and `No` and sample one token (t=1), which converts the addition/ablation terms into standard binary cross-entropy losses.

> 2. **Wide-scope retention.** To guard against collateral drift, L_retain is measured on 30-token continuations of Alpaca instructions, providing a broad behavioural footprint."

### Definition 3.1 - Activation Addition Loss
> "L_add = -(1/|D_false|) Σ_{x∈D_false} log ŷ_add(x + **v**)  [Add, target y = 1]"

### Definition 3.2 - Ablation Loss
> "L_ablate = -(1/|D_true|) Σ_{x∈D_true} log(1 - ŷ_ablate(x - **v**·**v**^⊤·x))  [Ablate, target y = 0]"

### Definition 3.3 - Retention Loss
> "L_retain = (1/|D_alpaca|) Σ_{x∈D_alpaca} D_KL[p₀(y_{1:30} | x) || p_**v**(y_{1:30} | x)]  [Retain, KL]"

---

## Evidence for Multi-Dimensionality

### Experiment 2 Results
> "The results in Table 1 suggest that increasing the dimensionality of the concept cone generally improves the model's ability to internalize and respond to truth-aligned interventions. **Larger models, such as Qwen-2.5-7B and Gemma-2-9B, maintain high ASR even as dimensionality increases, meaning that higher dimension cones exist within their activation space. This is consistent with findings from Wollschläger et al. [2025] in the domain of refusal behavior.**"

> "However, the trend is not monotonic: beyond a certain point, ASR begins to decline, indicating that additional directions may start to capture unrelated features and dilute the effectiveness of the intervention. This effect is especially evident in **smaller models, where cone dimensions above 2 or 3 yield diminishing or negative returns.** Nonetheless, the models still show multiple dimensions that independently support truth-aligned behavior."

---

## Critical Distinction: Orthogonality to DIM

### Section 4.4 & Discussion
> "We measure how closely the classic DIM truth vector aligns with the orthonormal directions discovered by our concept cone. Cosine similarity is reported in Table 3; values near 1 indicate strong overlap."

> "**Only the first cone axis has any alignment with DIM, confirming that DIM captures just one facet of the multi-dimensional truth subspace; the remaining axes encode additional, orthogonal structure.**"

### Discussion - Key Insight
> "Our findings reveal that while a single direction derived from the DIM method already captures a strong causal representation of truth in LLMs, **it does not fully exhaust the structure underlying truth-related behavior.** Through our concept cone approach, we identified additional orthogonal directions with low cosine similarity to the DIM vector that also reliably steer model outputs on propositional truth tasks."

> "This suggests that truthful behavior may not be confined to a single axis—**multiple directions can be independently influenced.** These directions likely correspond to distinct or semantically adjacent components of factual reasoning, such as modality, certainty, or domain-specific features."

---

## Surgical Precision (Preserving Unrelated Capabilities)

### Experiment 3 Results
> "We find that the truth-direction ablation leads to **only minimal divergence from the original output distribution, suggesting that the intervention does not significantly affect unrelated capabilities.**"

> "All models show low average KL divergence, **especially the larger variants. This suggests that the discovered truth directions are highly specific and do not interfere with general instruction-following behavior.** The effectiveness of L_retain as a regularization objective is empirically supported by this result."

---

## Layer Localization (Consistent with Refusal)

### Experiment 1 Findings
> "Across both model families and sizes, we find that truth-related directions reliably emerge in the **middle to later layers (specifically, between 60–75 percent of the normalized layer depth).** As shown in Figure 2, ASR increases sharply in this range before decreasing sharply again in the very last layers. Additionally, we find that the **final token position consistently yields the strongest interventions, consistent with prior work** showing that high-level decisions often accumulate at the end of the sequence [Arditi et al. 2024, Burger et al. 2024]."

---

## Framework Limitations & Boundary Cases

### Appendix B.1 - Sentiment Experiment
> "Previous literature [Tigges et al. 2023] suggests that sentiment has a linear representation, similar to other concepts such as refusal [Arditi et al. 2024]. **We tried to extend our methodology to sentiment to determine whether it has a concept cone representation... We failed to find a meaningful concept cone for sentiment.** Further work could explore alternative techniques for finding a higher-dimensional representation for sentiment."

### Appendix B.2 - Toxicity Experiment
> "We explored the existence of higher-dimensional representations for toxicity... **Testing this direction using ablation failed, as the resulting output was unintelligible.** Since we were unable to obtain a valid linear direction for toxicity, we were unable to generate high-quality targets and as a result, were unable to train a valid cone."

### Limitation 7.3 - Subspace Understanding
> "Although we demonstrate that a low-dimensional subspace (or 'cone') can causally mediate truth behavior, **our method does not guarantee the discovery of a maximally informative or interpretable subspace.** We leave for future work the development of principled methods – for example using sparsity constraints, disentanglement metrics, or unsupervised clustering – to assign semantic meaning to individual cone axes."

---

## Framework Generality Validation

### Discussion - Pattern Recognition
> "Our findings reveal that, **beyond a single 'truth direction,' there exists a robust subspace of activation vectors whose positive combinations consistently modulate factuality. These findings show increasing promise for concept cones as an interpretability toolkit** and underscore new avenues and risks for alignment, calibration, and adversarial manipulation of model truthfulness."

> "The success of both DIM and cone-based interventions suggests that **truth may be linearly separable in the model's representation space.** While the directions that independently modulate truthful behavior can imply that this structure may be richer than a single linear axis, it does not necessarily prove that the underlying representation of truth is nonlinear."

---

## Future Directions

### Section 6 - Conclusion
> "Several promising avenues remain. **Concept cone search reliably uncovers a subspace, but we are yet to find semantically meaningful labels for the basis vectors.** Future work could pair cones with automated clustering or sparse autoencoding so that each basis vector corresponds to an interpretable facet of truth (e.g. temporal facts, geographic facts, or commonsense). Extending the method to larger, instruction tuned models and to multimodal settings would also test its robustness and reveal whether these semantic dimensions persist across scale and modality."

---

## Citation of Foundation Work

Complete reference from bibliography:

> **Wollschläger, T., Elstner, J., Geisler, S., Cohen-Addad, V., Günnemann, S., and Gasteiger, J.** The geometry of refusal in large language models: Concept cones and representational independence. *arXiv preprint arXiv:2502.17420*, 2025. URL https://arxiv.org/abs/2502.17420.
