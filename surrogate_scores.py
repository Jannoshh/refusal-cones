# %%
import glob
import json
import numpy as np
import os

dim = 4
model_path = "/ceph/hdd/students/elsj/paper_results/subspace_samples_eval/join_subspace_gemma-2-2b-it/dim_4"
gemma_sample_performances = []
for sample_num in range(1, 257):
    # Check for attack sample
    attack_file = os.path.join(model_path, "completions", f"jailbreakbench_ablation_sample_{sample_num}_evaluations.json")
    attack_files = glob.glob(attack_file)
    if attack_files:
        with open(attack_files[0]) as f:
            data = json.load(f)
        scores = [d["is_jailbreak_strongreject"] for d in data["completions"]]
        gemma_sample_performances.append(scores)
np.array(gemma_sample_performances).shape
# %%

# %%
import torch
results_dir = f"results/refusal_scores/gemma-2-2b-it/"
surrogate_scores = torch.load(os.path.join(results_dir, "refusal_scores.pt"))
surrogate_scores.shape
# %%
import numpy as np
import matplotlib.pyplot as plt
import torch  # if not already imported
if isinstance(surrogate_scores, torch.Tensor):
    surrogate_scores_np = surrogate_scores.detach().cpu().numpy()
else:
    surrogate_scores_np = np.array(surrogate_scores)

# We only have surrogate scores for 64 samples, so we restrict our test scores accordingly.
# (Assuming that in gemma_sample_performances the first 64 rows correspond to the ones which
#  you have a surrogate score for.)
test_scores = np.array(gemma_sample_performances)[:64, :]

# We'll evaluate using different numbers of samples (from 1 to 64)
max_samples = surrogate_scores_np.shape[0]  # should be 64 here
sample_counts = np.arange(1, max_samples + 1)
mean_scores = []

# For each sample count, pick the best vector (per instruction) based on the surrogate heuristic,
# then average the corresponding test scores.
for n in sample_counts:
    # For each instruction (each column), we look at the surrogate scores for the first 'n' samples.
    # Assume that lower surrogate scores are "better". Use np.argmin if the reverse is true.
    best_idxs = np.argmin(surrogate_scores_np[:n, :], axis=0)  # shape (100,)
    
    # Now, for each instruction, pick the candidate (vector) test score corresponding to the best surrogate.
    # The advanced indexing below picks one test score for each instruction.
    selected_test_scores = test_scores[best_idxs, np.arange(test_scores.shape[1])]
    
    # Compute the mean test score over all instructions.
    mean_score = selected_test_scores.mean()
    mean_scores.append(mean_score)

# Plot the mean test score versus the number of samples selected.
plt.figure(figsize=(8, 6))
plt.plot(sample_counts, mean_scores, marker="o")
plt.xlabel("Number of Samples")
plt.ylabel("Mean Test Score")
plt.title("Mean Test Score vs. Number of Samples")
plt.grid(True)
plt.show()

# %%
# from scipy.stats import pearsonr
# import numpy as np

# # Compute Pearson correlation for all samples
# corr_all, p_val_all = pearsonr(sample_counts, mean_scores)
# print(f"Pearson correlation (all samples): {corr_all:.3f} (p-value: {p_val_all:.3f})")

# # Compute Pearson correlation for the top 20% quantile of sample counts
# # Determine the threshold (80th percentile) for sample_counts
# threshold = np.quantile(sample_counts, 0.8)
# # Filter for samples in the top 20% quantile (i.e. sample_counts greater than or equal to the threshold)
# mask = sample_counts >= threshold
# top_sample_counts = sample_counts[mask]
# top_mean_scores = np.array(mean_scores)[mask]

# corr_top, p_val_top = pearsonr(top_sample_counts, top_mean_scores)
# print(f"Pearson correlation (top 20% samples, sample_counts >= {threshold:.1f}): {corr_top:.3f} (p-value: {p_val_top:.3f})")

# %%
import matplotlib.pyplot as plt

# Check that we have enough instructions; test_scores is assumed to be of shape (num_candidates, num_instructions)
num_instructions = test_scores.shape[1]
selected_instructions = list(range(min(5, num_instructions)))  # select the first 5 instructions

plt.figure(figsize=(10, 6))
for ins in selected_instructions:
    # For each candidate (row) the pair (surrogate_scores_np, test_scores) for instruction 'ins' is plotted.
    plt.scatter(surrogate_scores_np[:, ins], test_scores[:, ins], label=f"Instruction {ins}")

plt.xlabel("Refusal Score (Surrogate Score)")
plt.ylabel("Test Score")
plt.title("Test Score vs. Refusal Score for 5 Instructions")
plt.legend()
plt.grid(True)
plt.show()