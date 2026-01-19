# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from matplotlib import rc
import json
import wandb
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
import seaborn as sns
import matplotlib
#%%
# Style configuration
# plt.style.use('default')
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)

sns.set_style("white")
sns.set_context("paper", font_scale=1, rc={
        "lines.linewidth": 1.2,
        "xtick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.major.size": 0,
        "ytick.minor.size": 0
    })

matplotlib.rcParams["mathtext.fontset"] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['figure.autolayout'] = True

plt.rc('font', size=13)
plt.rc('font', size=13)
plt.rc('axes', titlesize=13)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', title_fontsize=13)
plt.rc('legend', fontsize=13)
plt.rc('figure', titlesize=16)

colors = sns.color_palette('colorblind')
colors[0], colors[-1] = colors[-1], colors[0]

# %%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths to the original images
pics = [
    "results/plots/crossovereffects/orthogonal_ablation_gemma_2_2b.png",
    "results/plots/crossovereffects/orthogonal_ablation_qwen_7b.png",
    "results/plots/crossovereffects/orthogonal_ablation_llama_3_8b.png",
]

# Open all images and get their dimensions
images = [Image.open(pic) for pic in pics]
widths, heights = zip(*(img.size for img in images))

# Make sure all images have the same width
max_width = max(widths)
for i in range(len(images)):
    if widths[i] < max_width:
        # Resize to match width if needed
        ratio = max_width / widths[i]
        new_height = int(heights[i] * ratio)
        images[i] = images[i].resize((max_width, new_height), Image.LANCZOS)

# Process second and third images to remove top legend
for i in range(1, len(images)):
    width, height = images[i].size
    cut_height = int(height * 0.23)  # Cut 23% from top
    images[i] = images[i].crop((0, cut_height, width, height))

# Get updated heights after cropping
heights = [img.height for img in images]
total_height = sum(heights)

# Create a new blank image to hold all combined images
combined_image = Image.new('RGB', (max_width, total_height), color='white')

# Paste images one below the other
y_offset = 0
for img in images:
    combined_image.paste(img, (0, y_offset))
    y_offset += img.height

# Save and display the final image
combined_image.save('combined_plot.png')

# Display with matplotlib
plt.figure(figsize=(10, 15))
plt.imshow(np.array(combined_image))
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

# %%
# overrefusal
from collections import defaultdict
import os
import json

base_dir = "/ceph/hdd/students/elsj/paper_results/dim_overrefusal/Qwen2.5-3B-Instruct/completions/"
# defaultdict with tuple doesn't work as expected since tuples are immutable
# Use a dictionary with default values instead
asr_overrefusal_tradeoffs = {}

for file in os.listdir(base_dir):
    if "actadd" in file and file.endswith("_evaluations.json"):
        path = os.path.join(base_dir, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("substring_matching_success_rate") is None:
                    print(f"Missing success rate in file: {file}")
                    print(f"File contents: {data.keys()}")
                    continue
        except UnicodeDecodeError:
            print(f"Unicode decode error in file: {file}")
            continue
        
        parts = file.split("_")
        alpha = int(parts[-2])
        
        # Initialize the dictionary entry if it doesn't exist
        if alpha not in asr_overrefusal_tradeoffs:
            asr_overrefusal_tradeoffs[alpha] = [None, None]
        
        
        if "jailbreakbench" in file:
            asr = 1 - data.get("substring_matching_success_rate")
            asr_overrefusal_tradeoffs[alpha][0] = asr
        elif "xstest" in file:
            overrefusal_score = 1 - data.get("substring_matching_success_rate")
            asr_overrefusal_tradeoffs[alpha][1] = overrefusal_score

base_dir = "/ceph/hdd/students/elsj/paper_results/rdo_overrefusal/Qwen2.5-3B-Instruct/completions/"
# defaultdict with tuple doesn't work as expected since tuples are immutable
# Use a dictionary with default values instead
rdo_asr_overrefusal_tradeoffs = {}

for file in os.listdir(base_dir):
    if "actadd" in file and file.endswith("_evaluations.json"):
        path = os.path.join(base_dir, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("substring_matching_success_rate") is None:
                    print(f"Missing success rate in file: {file}")
                    print(f"File contents: {data.keys()}")
                    continue
        except UnicodeDecodeError:
            print(f"Unicode decode error in file: {file}")
            continue
        parts = file.split("_")
        alpha = int(parts[-2])
        
        # Initialize the dictionary entry if it doesn't exist
        if alpha not in rdo_asr_overrefusal_tradeoffs:
            rdo_asr_overrefusal_tradeoffs[alpha] = [None, None]
        
        
        if "jailbreakbench" in file:
            asr = 1 - data.get("substring_matching_success_rate")
            rdo_asr_overrefusal_tradeoffs[alpha][0] = asr
        elif "xstest" in file:
            overrefusal_score = 1 - data.get("substring_matching_success_rate")
            rdo_asr_overrefusal_tradeoffs[alpha][1] = overrefusal_score

# %%
# remove values with none
asr_overrefusal_tradeoffs = {k: v for k, v in asr_overrefusal_tradeoffs.items() if v[0] is not None and v[1] is not None}
rdo_asr_overrefusal_tradeoffs = {k: v for k, v in rdo_asr_overrefusal_tradeoffs.items() if v[0] is not None and v[1] is not None}

# %%
# plot the overrefusal tradeoffs with overrefusal score on the x-axis and asr on the y-axis
plt.figure(figsize=(10, 5))
# Create lists for x and y values
x_values = [asr_overrefusal_tradeoffs[alpha][1] for alpha in sorted(asr_overrefusal_tradeoffs.keys())]
y_values = [asr_overrefusal_tradeoffs[alpha][0] for alpha in sorted(asr_overrefusal_tradeoffs.keys())]
plt.plot(x_values, y_values, 'o-', label="DIM: JailbreakBench ASR vs XSTest Overrefusal")
plt.ylabel("JailbreakBench Refusal Score")
plt.xlabel("XSTest Refusal Score")

# Add RDO line
rdo_x_values = [rdo_asr_overrefusal_tradeoffs[alpha][1] for alpha in sorted(rdo_asr_overrefusal_tradeoffs.keys())]
rdo_y_values = [rdo_asr_overrefusal_tradeoffs[alpha][0] for alpha in sorted(rdo_asr_overrefusal_tradeoffs.keys())]
plt.plot(rdo_x_values, rdo_y_values, 'o-', label="RDO: JailbreakBench ASR vs XSTest Overrefusal")

plt.legend()
plt.show()

# %%
# Plot with alpha on the x-axis
plt.figure(figsize=(10, 5))

# Sort alphas for both methods
dim_alphas = sorted(asr_overrefusal_tradeoffs.keys())
rdo_alphas = sorted(rdo_asr_overrefusal_tradeoffs.keys())

# Plot JailbreakBench refusal scores
plt.plot(dim_alphas, [asr_overrefusal_tradeoffs[alpha][0] for alpha in dim_alphas], 
         'o-', label="DIM: JailbreakBench Refusal")
plt.plot(rdo_alphas, [rdo_asr_overrefusal_tradeoffs[alpha][0] for alpha in rdo_alphas], 
         's-', label="RDO: JailbreakBench Refusal")

# Plot XSTest overrefusal scores
plt.plot(dim_alphas, [asr_overrefusal_tradeoffs[alpha][1] for alpha in dim_alphas], 
         'o--', label="DIM: XSTest Refusal")
plt.plot(rdo_alphas, [rdo_asr_overrefusal_tradeoffs[alpha][1] for alpha in rdo_alphas], 
         's--', label="RDO: XSTest Refusal")

plt.xlabel("Alpha (Intervention Strength)")
plt.ylabel("Refusal Score")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
# KL ablation
path = "results/refusal_dir/Qwen2.5-14B-Instruct/select_direction/direction_evaluations.json"
with open(path, "r") as f:
    data = json.load(f)

# Convert data into a more manageable format
positions = sorted(list(set(d['position'] for d in data)))
layers = sorted(list(set(d['layer'] for d in data)))

# Filter layers to only include those < 0.8 * max layer
max_layer = max(layers)
layers = [l for l in layers]

# Create a dictionary to store KL scores for each position and layer
kl_scores = {pos: [] for pos in positions}
induce_scores = {pos: [] for pos in positions}
for d in data:
    if d['layer'] in layers:  # Only include data for filtered layers
        kl_scores[d['position']].append(d['kl_div_score'])
        induce_scores[d['position']].append(d.get('steering_score', 0))

# Create the plot
plt.figure(figsize=(10, 5))

# First plot faded points (induce_score <= 0)
for pos_idx, pos in enumerate(positions):
    # Plot a single point with alpha=1 for the legend
    plt.plot([], [], 'o-',
            color=colors[pos_idx],
            label=f"Position {pos}",
            alpha=1)
            
    for i, layer in enumerate(layers):
        induce_score = induce_scores[pos][i]
        if induce_score <= 0:
            plt.plot(layer, kl_scores[pos][i], 'o-',
                    alpha=0.15,
                    markersize=4,
                    color=colors[pos_idx],
                    markeredgecolor='white',
                    markeredgewidth=0.5,
                    zorder=1)
            
            if i < len(layers)-1:
                plt.plot([layer, layers[i+1]], 
                        [kl_scores[pos][i], kl_scores[pos][i+1]],
                        alpha=0.1,
                        color=colors[pos_idx],
                        zorder=1)

# Then plot surviving points (induce_score > 0) with enhanced visibility
for pos_idx, pos in enumerate(positions):
    for i, layer in enumerate(layers):
        induce_score = induce_scores[pos][i]
        if induce_score > 0:
            plt.plot(layer, kl_scores[pos][i], 'o',
                    markersize=8,
                    color=colors[pos_idx],
                    markeredgecolor='black',
                    markeredgewidth=1,
                    zorder=4)
            
            if i < len(layers)-1:
                plt.plot([layer, layers[i+1]], 
                        [kl_scores[pos][i], kl_scores[pos][i+1]],
                        alpha=0.8,
                        color=colors[pos_idx],
                        linewidth=2,
                        zorder=3)

# Customize axes and labels
plt.xlabel('Source Layer')
plt.ylabel('KL Score')
plt.legend(title='Source Position', 
          loc='upper left',
          frameon=True,
          fancybox=False,
          edgecolor='black')

# Improve grid appearance
plt.grid(True, alpha=0.2, color='gray', linestyle='-', linewidth=0.5)

# Set y-axis to log scale with better limits
plt.yscale('log')
plt.ylim(0, 1e1)

# Add baseline and threshold with improved styling
plt.axhline(y=0.1, color="red", linestyle='--', alpha=0.5, linewidth=1, zorder=1)
plt.text(0, 0.07, 'KL Filter Threshold (0.1)', color="red", alpha=1)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1, zorder=1)

# Clean up spines
for spine in plt.gca().spines.values():
    spine.set_linewidth(0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adjust tick parameters
plt.tick_params(axis='both', which='major', width=0.5, length=4)
plt.tick_params(axis='both', which='minor', width=0.5, length=2)

plt.tight_layout()
os.makedirs("results/plots/ablations", exist_ok=True)
plt.savefig("results/plots/ablations/direction_kl_divergence_qwen14b.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# loss weight ablation
# Create a figure with two subplots stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

# First subplot - No retain loss
group = "loss_ablation_fixed_no_retain_Meta-Llama-3-8B-Instruct"
api = wandb.Api()
runs = api.runs("refusal-representations/robust_refusal_vector", {"group": group})

for run in runs:
    print(run.name)
    break

# Organize data by loss weight
loss_weight_data = {}
for run in runs:
    weight = run.config["addition_lambda"]
    ablation_score = run.summary.get("jailbreakbench_ablation_StrongREJECT_score", None)
    actadd_score = run.summary.get("jailbreakbench_actadd_StrongREJECT_score", None)
    # print(weight, ablation_score, actadd_score)
    
    if weight not in loss_weight_data:
        loss_weight_data[weight] = {"ablation": [], "actadd": []}
    
    if ablation_score is not None:
        loss_weight_data[weight]["ablation"].append(ablation_score)
    else:
        print(f"No ablation score for {run.name}")
    if actadd_score is not None:
        loss_weight_data[weight]["actadd"].append(actadd_score)
    
# Calculate mean and standard error of the mean for each loss weight
# Sort weights for consistent plotting
unique_weights = sorted(loss_weight_data.keys())
ablation_means = []
ablation_sems = []
actadd_means = []
actadd_sems = []

for weight in unique_weights:
    ablation_means.append(np.mean(loss_weight_data[weight]["ablation"]))
    # Calculate standard error of the mean (SEM = std / sqrt(n))
    ablation_sems.append(np.std(loss_weight_data[weight]["ablation"]) / np.sqrt(len(loss_weight_data[weight]["ablation"])))
    actadd_means.append(np.mean(loss_weight_data[weight]["actadd"]))
    actadd_sems.append(np.std(loss_weight_data[weight]["actadd"]) / np.sqrt(len(loss_weight_data[weight]["actadd"])))

# Plot with error bars using standard error of the mean
axes[0].errorbar(unique_weights, ablation_means, yerr=ablation_sems, fmt='o-', 
             capsize=5, label='Directional Ablation', color=colors[0])
axes[0].errorbar(unique_weights, actadd_means, yerr=actadd_sems, fmt='s-', 
             capsize=5, label='Activation Subtraction', color=colors[1])

axes[0].set_ylabel('Attack Success Rate')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_title('No Retain Loss')
axes[0].set_ylim(0, 1.0)  # Set y-axis to start at 0

# Add x-axis annotations directly under the ticks
min_weight = min(unique_weights)
max_weight = max(unique_weights)

# Second subplot - With retain loss
group = "loss_ablation_fixed_Meta-Llama-3-8B-Instruct"
api = wandb.Api()
runs = api.runs("refusal-representations/robust_refusal_vector", {"group": group})

# Organize data by loss weight
loss_weight_data = {}
for run in runs:
    weight = run.config["addition_lambda"]
    ablation_score = run.summary.get("jailbreakbench_ablation_StrongREJECT_score", None)
    actadd_score = run.summary.get("jailbreakbench_actadd_StrongREJECT_score", None)
    
    if weight not in loss_weight_data:
        loss_weight_data[weight] = {"ablation": [], "actadd": []}
    
    if ablation_score is not None:
        loss_weight_data[weight]["ablation"].append(ablation_score)
    else:
        print(f"No ablation score for {run.name}")
    if actadd_score is not None:
        loss_weight_data[weight]["actadd"].append(actadd_score)
    

# Calculate mean and standard error of the mean for each loss weight
# Sort weights for consistent plotting
unique_weights = sorted(loss_weight_data.keys())
ablation_means = []
ablation_sems = []
actadd_means = []
actadd_sems = []

for weight in unique_weights:
    ablation_means.append(np.mean(loss_weight_data[weight]["ablation"]))
    # Calculate standard error of the mean (SEM = std / sqrt(n))
    ablation_sems.append(np.std(loss_weight_data[weight]["ablation"]) / np.sqrt(len(loss_weight_data[weight]["ablation"])))
    actadd_means.append(np.mean(loss_weight_data[weight]["actadd"]))
    actadd_sems.append(np.std(loss_weight_data[weight]["actadd"]) / np.sqrt(len(loss_weight_data[weight]["actadd"])))

# Plot with error bars using standard error of the mean
axes[1].errorbar(unique_weights, ablation_means, yerr=ablation_sems, fmt='o-', 
             capsize=5, color=colors[0])
axes[1].errorbar(unique_weights, actadd_means, yerr=actadd_sems, fmt='s-', 
             capsize=5, color=colors[1])

axes[1].set_xlabel('Addition Loss Weight (1 - Ablation Loss Weight)', labelpad=20)
axes[1].set_ylabel('Attack Success Rate')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].set_title('With Retain Loss ($\lambda_{ret} = 1$)')
axes[1].set_ylim(0, 1.0)  # Set y-axis to start at 0

# Clean up spines
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add x-axis annotations directly under the ticks
min_weight = min(unique_weights)
max_weight = max(unique_weights)
axes[1].text(min_weight, -0.14, '($\lambda_{add} = 0$, $\lambda_{abl} = 1$)', ha='center', transform=axes[1].get_xaxis_transform(), fontsize=12)
axes[1].text(max_weight, -0.14, '($\lambda_{add} = 1$, $\lambda_{abl} = 0$)', ha='center', transform=axes[1].get_xaxis_transform(), fontsize=12)

# Add a single legend for both subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=1)

plt.tight_layout()
os.makedirs("results/plots/ablations", exist_ok=True)
plt.savefig("results/plots/ablations/loss_weights_comparison.png", dpi=300, bbox_inches='tight')
plt.show()


# %%
# Compare topk RepInd directions with DIM direction
repind_scores = [0.42305508613586423, 0.6420154309272766, 0.8002064847946166, 0.8072359848022461, 0.8158208751678466]
dim_score = 0.7858  # DIM direction score from eval_results[0]

# Create figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot DIM score as horizontal line
ax.axhline(y=dim_score, color=colors[0], linestyle='--', label='DIM Direction')
# Plot RepInd scores
ax.plot(range(1, len(repind_scores) + 1), repind_scores, 'o-', color=colors[1], label='RepInd Directions')

# Add value annotations
# Add DIM score annotation
ax.text(len(repind_scores) - 0.16, dim_score - 0.06, f'{dim_score:.3f}', 
        va='center', ha='left', color=colors[0], fontweight='bold')

# Add RepInd score annotations
for i, score in enumerate(repind_scores):
    ax.text(i + 1, score + 0.03, f'{score:.3f}', 
            va='bottom', ha='center', color=colors[1], fontweight='bold')

# Add labels and styling
ax.set_xlabel('Number of RepInd Directions Ablated (Top-k)')
ax.set_ylabel('Attack Success Rate')
ax.set_xticks(range(1, len(repind_scores) + 1))
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, linestyle='--')

# Clean up spines
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc="lower right", bbox_to_anchor=(1, 0.02))

plt.tight_layout()
os.makedirs("results/plots/ablations", exist_ok=True)
plt.savefig("results/plots/topk_ablation.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
# topk ablation
models = ["google/gemma-2-9b-it", "meta-llama/Meta-Llama-3-8B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]
model_display_names = ["Gemma 2 9B", "Llama 3 8B", "Qwen 2.5 14B"]
k_values = list(range(1, 6))

# Create a single figure for both metrics
plt.figure(figsize=(10, 6))

# Use different line styles for ASR and substring matching
line_styles = ['-', '--']
markers = ['o', 's']

# Create empty lists to store legend handles
model_handles = []
metric_handles = []

for i, (model, display_name) in enumerate(zip(models, model_display_names)):
    asrs = []
    substring_matches = []
    for k in k_values:
        model_name = model.split("/")[-1]
        path = f"results/refusal_dir/{model_name}/completions/jailbreakbench_ablationtop_{k}_evaluations.json"
        
        if os.path.exists(path):
            results = json.load(open(path, "r"))
            asr = results.get("StrongREJECT_score")
            template_matching = results.get("substring_matching_success_rate")
            if asr is None:
                print(f"Warning: 'StrongREJECT_score' not found in {path}")
                # Try to find alternative keys if the expected one isn't present
                print(f"Available keys: {list(results.keys())}")
            asrs.append(asr)
            substring_matches.append(template_matching)
        else:
            print(f"File not found: {path}")
            asrs.append(None)
            substring_matches.append(None)
    
    # Filter out None values for plotting
    valid_k_asr = [k for k, asr in zip(k_values, asrs) if asr is not None]
    valid_asrs = [asr for asr in asrs if asr is not None]
    
    valid_k_substring = [k for k, match in zip(k_values, substring_matches) if match is not None]
    valid_substring_matches = [match for match in substring_matches if match is not None]
    
    # Plot ASR
    if valid_asrs:
        line_asr, = plt.plot(valid_k_asr, valid_asrs, 
                 marker=markers[0], 
                 linestyle=line_styles[0], 
                 color=colors[i % len(colors)],
                 label=f"{display_name}")
        if i == 0:
            metric_handles.append(line_asr)
    
    # Plot Substring Matching
    if valid_substring_matches:
        line_substring, = plt.plot(valid_k_substring, valid_substring_matches, 
                 marker=markers[1], 
                 linestyle=line_styles[1], 
                 color=colors[i % len(colors)],
                 label=f"{display_name}")
    
    # Add a handle for the model legend (only once per model)
    if valid_asrs or valid_substring_matches:
        model_handles.append(plt.Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=display_name))

# Add dummy lines for metric legend
if len(metric_handles) < 2:  # If we didn't already add both metric types
    metric_handles = [
        plt.Line2D([0], [0], marker=markers[0], linestyle=line_styles[0], color='gray', label='Attack Success Rate'),
        plt.Line2D([0], [0], marker=markers[1], linestyle=line_styles[1], color='gray', label='1 − Refusal Score')
    ]

# Configure plot
plt.xlabel('Top-k DIM Directions Ablated')
plt.ylabel('Success Rate')
plt.grid(True, alpha=0.3, linestyle='--')

# Create two separate legend boxes
first_legend = plt.legend(handles=model_handles, title="Models", loc='upper left', bbox_to_anchor=(1.05, 1))
for spine in plt.gca().spines.values():
    spine.set_linewidth(0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().add_artist(first_legend)  # Add the first legend and keep it
plt.legend(handles=metric_handles, title="Metrics", loc='upper left', bbox_to_anchor=(1.05, 0.6))

plt.xticks(k_values)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("results/plots/topk_combined.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
models = [
    dict(model_name="Qwen 2.5 3B",
         model_id="Qwen/Qwen2.5-3B-Instruct"),
    # dict(model_name="Llama 3 8B",
    #      model_id="meta-llama/Meta-Llama-3-8B-Instruct"),
]
model = models[0]  
model_id = model["model_id"].split("/")[-1]
group = f"kl_ablation_{model_id}" if model_id != "Qwen2.5-3B-Instruct" else f"sb_data_kl_ablation_{model_id}"
# group = f"sb_data_kl_ablation_{model_id}"
lambdas = [0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16] if model_id == "Qwen2.5-3B-Instruct" else [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]
# Load scores for each lambda value

api = wandb.Api()
group_runs = api.runs("refusal-representations/robust_refusal_vector", {"group": group})

# Initialize dictionaries to store scores for each lambda
ablation_by_lambda = {}
actadd_by_lambda = {}
ablation_by_val_lambda = {}
actadd_by_val_lambda = {}

# Store pairs of (ablation score, benchmark score)
ablation_benchmark_pairs = []

# Collect scores for each lambda value
for run in group_runs:
    summary = run.summary._json_dict
    retain_lambda = run.config["retain_lambda"]
    
    # Get scores
    ablation_score = summary.get("jailbreakbench_ablation_StrongREJECT_score", None)
    actadd_score = summary.get("jailbreakbench_actadd_StrongREJECT_score", None)
    val_lambda = summary.get("val/kl_div_score", None)
    if val_lambda is not None and val_lambda > 0.3:
        continue
    
    # Add scores to respective lambda lists
    if retain_lambda not in ablation_by_lambda:
        ablation_by_lambda[retain_lambda] = []
        actadd_by_lambda[retain_lambda] = []
        
    # Round val_lambda to 3 decimal places to handle floating point imprecision
    if val_lambda is not None:
        val_lambda = round(val_lambda, 3)
        if val_lambda not in ablation_by_val_lambda:
            ablation_by_val_lambda[val_lambda] = []
            actadd_by_val_lambda[val_lambda] = []
        
    if ablation_score is not None:
        ablation_by_lambda[retain_lambda].append(ablation_score)
    if actadd_score is not None:
        actadd_by_lambda[retain_lambda].append(actadd_score)
    if val_lambda is not None and ablation_score is not None:
        ablation_by_val_lambda[val_lambda].append(ablation_score)
    if val_lambda is not None and actadd_score is not None:
        actadd_by_val_lambda[val_lambda].append(actadd_score)

    run_name = run.name
    benchmark_folder = os.path.join(f"lm-evaluation-harness/kl_ablation_results/{run_name}/meta-llama__Meta-Llama-3-8B-Instruct") if model_id != "Qwen2.5-3B-Instruct" else os.path.join(f"lm-evaluation-harness/chat_results/{run_name}/Qwen__Qwen2.5-3B-Instruct")
    if os.path.exists(benchmark_folder):
        for file in os.listdir(benchmark_folder)[::-1]:
            if file.startswith("results_"):
                with open(os.path.join(benchmark_folder, file), "r") as f:
                    data = json.load(f)
                truthfulqa_mc2_acc = data["results"]["truthfulqa_mc2"]["acc,none"]
                arc_challenge_acc = data["results"]["arc_challenge"]["acc,none"]
                gsm8k_acc = data["results"]["gsm8k"]["exact_match,flexible-extract"]
                mmlu_acc = data["results"]["mmlu"]["acc,none"]

                avg_acc = (truthfulqa_mc2_acc + arc_challenge_acc + mmlu_acc + gsm8k_acc) / 4
                print(run_name, avg_acc)
                
                if ablation_score is not None:
                    ablation_benchmark_pairs.append((ablation_score, avg_acc, retain_lambda))
                break
                    
        
# Calculate means and stds for retain lambda
ablation_stats = []
actadd_stats = []
for lambda_val in sorted(ablation_by_lambda.keys()):
    abl_scores = ablation_by_lambda[lambda_val]
    act_scores = actadd_by_lambda[lambda_val]
    
    abl_mean = np.mean(abl_scores) if abl_scores else None
    abl_std = np.std(abl_scores) if abl_scores else None
    act_mean = np.mean(act_scores) if act_scores else None 
    act_std = np.std(act_scores) if act_scores else None
    
    ablation_stats.append((lambda_val, abl_mean, abl_std))
    actadd_stats.append((lambda_val, act_mean, act_std))
    
    # print(f"Retain Lambda {lambda_val}:")
    # print(f"  Ablation - Mean: {abl_mean:.3f}, Std: {abl_std:.3f}")
    # print(f"  ActAdd - Mean: {act_mean:.3f}, Std: {act_std:.3f}")

# Calculate means and stds for validation lambda
val_ablation_stats = []
val_actadd_stats = []
for val_lambda in sorted(ablation_by_val_lambda.keys()):
    val_abl_scores = ablation_by_val_lambda[val_lambda]
    val_act_scores = actadd_by_val_lambda[val_lambda]
    
    val_abl_mean = np.mean(val_abl_scores) if val_abl_scores else None
    val_abl_std = np.std(val_abl_scores) if val_abl_scores else None
    val_act_mean = np.mean(val_act_scores) if val_act_scores else None
    val_act_std = np.std(val_act_scores) if val_act_scores else None
    
    val_ablation_stats.append((val_lambda, val_abl_mean, val_abl_std))
    val_actadd_stats.append((val_lambda, val_act_mean, val_act_std))
    
model["ablation_stats"] = ablation_stats
model["actadd_stats"] = actadd_stats
model["val_ablation_stats"] = val_ablation_stats
model["val_actadd_stats"] = val_actadd_stats
model["ablation_benchmark_pairs"] = ablation_benchmark_pairs

# %%
# save the ablation_benchmark_pairs to a json file
# with open("results/ablation_benchmark_pairs.json", "w") as f:
#     json.dump(ablation_benchmark_pairs, f)

# loaded_ablation_benchmark_pairs = json.load(open("results/ablation_benchmark_pairs.json", "r"))
# loaded_ablation_benchmark_pairs

# %%
# Create a figure with two subplots side by side
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# bar_width = 0.35

# # First subplot - Bar plot
# for model in models:
#     ablation_stats = model["ablation_stats"]
#     actadd_stats = model["actadd_stats"]
    
#     # Ensure we only plot data for the lambda values we have
#     valid_lambdas = [x[0] for x in ablation_stats]
#     x = np.arange(len(valid_lambdas))
    
#     # Plot ablation bars
#     ablation_means = [mean for _, mean, _ in ablation_stats if mean is not None]
#     ablation_stds = [std for _, _, std in ablation_stats if std is not None]
#     ax1.bar(x - bar_width/2, ablation_means, bar_width, 
#             label=f"Directional Ablation",
#             color=colors[0],
#             alpha=0.8,
#             yerr=ablation_stds,
#             capsize=5,
#             zorder=2)

#     # Plot activation addition bars
#     actadd_means = [mean for _, mean, _ in actadd_stats if mean is not None]
#     actadd_stds = [std for _, _, std in actadd_stats if std is not None]
#     ax1.bar(x + bar_width/2, actadd_means, bar_width,
#             label=f"Activation Subtraction", 
#             color=colors[1],
#             alpha=0.8,
#             yerr=actadd_stds,
#             capsize=5,
#             zorder=1)

# # Customize first subplot
# ax1.set_xlabel("Retain Loss Weight")
# ax1.set_ylabel("Attack Success Rate")
# ax1.set_xticks(x)
# ax1.set_xticklabels(['$2^{' + str(int(np.log2(lam))) + '}$' if lam > 0 else '$0$' for lam in valid_lambdas])
# ax1.tick_params(axis='both', which='major')
# ax1.grid(True, alpha=0.2, linestyle='-', axis='y')
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.set_ylim(0, 1)

# # Second subplot - Scatter plot
# for model in models:
#     # Get stats for ablation and activation addition
#     ablation_stats = model["val_ablation_stats"]
#     actadd_stats = model["val_actadd_stats"]
    
#     # Get valid lambda values
#     valid_lambdas = [x[0] for x in ablation_stats]
    
#     # Plot ablation points
#     ablation_means = [mean for _, mean, _ in ablation_stats if mean is not None]
#     ax2.plot(valid_lambdas, ablation_means,
#              label=f"Directional Ablation",
#              color=colors[0],
#              marker='o',
#              markersize=5,
#              linestyle='-',
#              alpha=0.8)

#     # Plot activation addition points
#     actadd_means = [mean for _, mean, _ in actadd_stats if mean is not None]
#     ax2.plot(valid_lambdas, actadd_means,
#              label=f"Activation Subtraction",
#              color=colors[1],
#              marker='o',
#              markersize=5,
#              linestyle='-',
#              alpha=0.8)

# # Customize second subplot
# ax2.set_xlabel("Validation KL-score")
# ax2.set_xscale('log')
# ax2.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2])  # Add more x-ticks
# ax2.set_xticklabels(['0.01', '0.02', '0.05', '0.1', '0.2'])  # Label the ticks
# ax2.grid(True, alpha=0.3)
# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)

# # Add legend to the right of both subplots
# handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels,
#           loc='upper center',
#           bbox_to_anchor=(0.5, 1.08),
#           ncol=2,
#           frameon=True,
#           fancybox=False,
#           edgecolor='black')

# # Adjust layout
# plt.tight_layout()
# os.makedirs("results/plots/ablations", exist_ok=True)
# plt.savefig("results/plots/ablations/ablation_vs_actadd.png", dpi=300, bbox_inches='tight')
# plt.show()

# %%
# Create a plot showing the tradeoff between ablation performance and benchmark scores
benchmark_model_id = model["model_id"].replace("/", "__")
plt.figure(figsize=(8, 6))
baseline_benchmark_path = f"lm-evaluation-harness/chat_results/baseline/{benchmark_model_id}/"
for file in os.listdir(baseline_benchmark_path)[::-1]:
    if file.startswith("results_"):
        with open(os.path.join(baseline_benchmark_path, file), "r") as f:
            baseline_data = json.load(f)
            break
baseline_score = baseline_data["results"]["truthfulqa_mc2"]["acc,none"] + \
    baseline_data["results"]["arc_challenge"]["acc,none"] + \
    baseline_data["results"]["mmlu"]["acc,none"] + \
    baseline_data["results"]["gsm8k"]["exact_match,flexible-extract"]
baseline_score = baseline_score / 4

dim_benchmark_path = f"lm-evaluation-harness/chat_results/original/{benchmark_model_id}/"
for file in os.listdir(dim_benchmark_path):
    if file.startswith("results_"):
        with open(os.path.join(dim_benchmark_path, file), "r") as f:
            dim_data = json.load(f)
            break
dim_score = dim_data["results"]["truthfulqa_mc2"]["acc,none"] + \
    dim_data["results"]["arc_challenge"]["acc,none"] + \
    dim_data["results"]["mmlu"]["acc,none"] + \
    dim_data["results"]["gsm8k"]["exact_match,flexible-extract"]
dim_score = dim_score / 4
print(dim_score)

# Collect all unique lambda values to create a categorical colormap
all_lambdas = []
for model in models:
    pairs = model["ablation_benchmark_pairs"]
    all_lambdas.extend([lam for _, _, lam in pairs])
unique_lambdas = sorted(list(set(all_lambdas)))

# Create a categorical colormap
# Use a more distinct colormap for better clarity
cmap = plt.cm.get_cmap('plasma', len(unique_lambdas))
# Create a dictionary mapping lambda values to colors with higher contrast
lambda_to_color = {lam: cmap(i) for i, lam in enumerate(unique_lambdas)}

# Create a scatter plot with points colored by lambda value as categories
for model in models:
    pairs = model["ablation_benchmark_pairs"]
    
    # Extract data points
    ablation_scores = [abl for abl, _, _ in pairs]
    benchmark_scores = [100 * (bench - baseline_score) for _, bench, _ in pairs]
    # benchmark_scores = [bench for _, bench, _ in pairs]
    retain_lambdas = [lam for _, _, lam in pairs]
    
    # Plot each lambda value as a separate category
    for lam in unique_lambdas:
        # Filter points for this lambda value
        indices = [i for i, l in enumerate(retain_lambdas) if l == lam]
        if indices:
            plt.scatter(
                [benchmark_scores[i] for i in indices],
                [ablation_scores[i] for i in indices],
                s=80,
                color=lambda_to_color[lam],
                alpha=0.8,
                edgecolors='black',
                linewidths=1,
                label=f"λ = {lam}" if model == models[0] else "",  # Only add to legend once
                zorder=3
            )

# Add a point for the DIM score
dim_performance = 0.6819
print(dim_score - baseline_score)
plt.scatter(100 * (dim_score - baseline_score), dim_performance, color=colors[0], s=200, marker='*', 
            label='DIM', zorder=5, edgecolors='black', linewidths=1)
# plt.annotate(f'DIM: {dim_score:.3f}', (100 * (dim_score - baseline_score), dim_performance), 
            #  xytext=(5, 5), textcoords='offset points', color='#1e88e5', fontsize=12, fontweight='bold')

# Add legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(list(by_label.values()), list(by_label.keys()), 
           title="Retain Loss Weight", 
           loc='best',
           frameon=True,
           framealpha=0.9,
           edgecolor='gray',
           borderaxespad=1)

# Customize plot
# plt.xlabel('Average Benchmark Score')
plt.xlabel('Difference in Avg. Benchmark Score Compared to No Intervention (%)')
plt.ylabel('Attack Success Rate')
# plt.title('Tradeoff: Attack Success vs. Benchmark Performance')
plt.grid(True, alpha=0.2, linestyle='-')
# plt.xlim(0, 1)
plt.ylim(0, 0.8)

# Clean up spines
for spine in plt.gca().spines.values():
    spine.set_linewidth(0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
os.makedirs("results/plots/ablations", exist_ok=True)
plt.savefig("results/plots/ablations/ablation_benchmark_tradeoff.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
from scipy.stats import norm

def combined_score(data):
    standardized = (data - data.mean(axis=0)) / data.std(axis=0)
    cdfed = norm.cdf(standardized)
    all_points = []
    for sample in cdfed:
        points = []
        for alpha in np.linspace(0, 1):
            points.append(alpha*sample[0] + (1-alpha)*sample[1])
        all_points.append(points)
    return np.array(all_points).sum(axis=1)

fig, ax = plt.subplots(figsize=(8, 6))

# Collect all data points for combined scoring
all_ablation_scores = []
all_benchmark_scores = []
all_model_indices = []
all_lambda_values = []

for i, model in enumerate(models):
    if "ablation_benchmark_pairs" in model:
        pairs = model["ablation_benchmark_pairs"]
        
        # Extract data points
        ablation_scores = [abl for abl, _, _ in pairs]
        benchmark_scores = [bench for _, bench, _ in pairs]
        retain_lambdas = [lam for _, _, lam in pairs]
        
        all_ablation_scores.extend(ablation_scores)
        all_benchmark_scores.extend(benchmark_scores)
        all_model_indices.extend([i] * len(pairs))
        all_lambda_values.extend(retain_lambdas)

# Add DIM point
all_ablation_scores.append(dim_performance)
all_benchmark_scores.append(dim_score)
all_model_indices.append(-1)  # Special index for DIM
all_lambda_values.append(-1)  # Special value for DIM

# Combine into a numpy array for scoring
data = np.array([all_ablation_scores, all_benchmark_scores]).T
scores = combined_score(data)

# Create a colormap for the scores
cmap = plt.cm.viridis
norm = plt.Normalize(scores.min(), scores.max())

# Plot each point with color based on combined score
for i, (abl, bench, model_idx, lam, score) in enumerate(zip(all_ablation_scores, all_benchmark_scores, 
                                                           all_model_indices, all_lambda_values, scores)):
    if model_idx == -1:  # DIM point
        ax.scatter(100 * (bench - baseline_score), abl, color=colors[0], s=200, marker='*', 
                   label='DIM', zorder=5, edgecolors='black', linewidths=1)
    else:
        marker_style = 'o'
        size = 80
        ax.scatter(100 * (bench - baseline_score), abl, s=size, 
                   color=cmap(norm(score)), 
                   marker=marker_style,
                   alpha=0.8, edgecolors='black', linewidths=1,
                   label=f"λ = {lam}" if lam not in [all_lambda_values[:i]] else "",
                   zorder=3)

# Add a colorbar with explicit axes reference
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Combined Score')

# Customize plot
ax.set_xlabel('Difference in Avg. Benchmark Score Compared to No Intervention (%)')
ax.set_ylabel('Attack Success Rate')
ax.grid(True, alpha=0.2, linestyle='-')
ax.set_ylim(0, 0.8)

# Clean up spines
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend for lambda values
handles, labels = [], []
for lam in sorted(set(all_lambda_values)):
    if lam != -1:  # Skip DIM
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=cmap(0.5), markersize=10,
                                 markeredgecolor='black', markeredgewidth=1))
        labels.append(f"λ = {lam}")
# Add DIM to legend
handles.append(plt.Line2D([0], [0], marker='*', color='w', 
                         markerfacecolor=colors[0], markersize=15,
                         markeredgecolor='black', markeredgewidth=1))
labels.append('DIM')

# ax.legend(handles, labels, title="Methods", loc='best',
        #   frameon=True, framealpha=0.9, edgecolor='gray', borderaxespad=1)

plt.tight_layout()
os.makedirs("results/plots/ablations", exist_ok=True)
plt.savefig("results/plots/ablations/combined_score_tradeoff.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# models = [
#     dict(model_name="Gemma 2 2B", 
#          model_id="google/gemma-2-2b-it"),
#     dict(model_name="Gemma 2 9B",
#          model_id="google/gemma-2-9b-it"),
#     dict(model_name="Llama 3 8B",
#          model_id="meta-llama/Meta-Llama-3-8B-Instruct"),
#     dict(model_name="Qwen 2.5 1.5B",
#          model_id="Qwen/Qwen2.5-1.5B-Instruct"),
#     dict(model_name="Qwen 2.5 3B",
#          model_id="Qwen/Qwen2.5-3B-Instruct"),
#     dict(model_name="Qwen 2.5 7B",
#          model_id="Qwen/Qwen2.5-7B-Instruct"),
#     dict(model_name="Qwen 2.5 14B",
#          model_id="Qwen/Qwen2.5-14B-Instruct"),
# ]

benchmarks_path = 'lm-evaluation-harness/chat_results'
folder_names = ["original", "baseline", "wandb"]
benchmarks = ["truthfulqa_mc2", "arc_challenge", "gsm8k", "mmlu"]
for model in models:
    family, model_id = model["model_id"].split("/")
    for folder_name in folder_names:
        model_path = os.path.join(benchmarks_path, folder_name, f"{family}__{model_id}")
        if os.path.exists(model_path):
            # get the file that starts with results_ and load the json
            for file in os.listdir(model_path)[::-1]:
                if file.startswith("results_"):
                    with open(os.path.join(model_path, file), "r") as f:
                        data = json.load(f)
                    
                    # Extract all benchmark scores
                    results = data["results"]

                    if "arc_challenge" not in results:
                        continue
                    
                    # TruthfulQA
                    if "truthfulqa_mc2" in results:
                        model[f"{folder_name}_truthfulqa_mc2_acc"] = results["truthfulqa_mc2"]["acc,none"]
                    
                    # ARC Challenge
                    if "arc_challenge" in results:
                        model[f"{folder_name}_arc_challenge_acc"] = results["arc_challenge"]["acc,none"]
                    
                    # GSM8K (using flexible-extract for better accuracy)
                    if "gsm8k" in results:
                        model[f"{folder_name}_gsm8k_acc"] = results["gsm8k"]["exact_match,flexible-extract"]
                    
                    # MMLU
                    if "mmlu" in results:
                        model[f"{folder_name}_mmlu_acc"] = results["mmlu"]["acc,none"]
                    
                    avg_acc = (model[f"{folder_name}_truthfulqa_mc2_acc"] + \
                        model[f"{folder_name}_gsm8k_acc"] + \
                        model[f"{folder_name}_arc_challenge_acc"] + \
                        model[f"{folder_name}_mmlu_acc"]) / 4
                    model[f"{folder_name}_avg_acc"] = avg_acc
                    
                    break
        else:
            print(f"Folder {folder_name} does not exist for model {model_id}")
            continue
# %%
from tabulate import tabulate

def format_value(value):
    return f"{value*100:.1f}%" if isinstance(value, (int, float)) else str(value)

# Create a table for each benchmark
benchmarks = ["truthfulqa_mc2", "arc_challenge", "gsm8k", "mmlu", "avg"]
headers = ["Model", "DIM", "RDO", "Baseline"]

# Create directory for saving tables
os.makedirs("results/tables", exist_ok=True)

# Open a single file to save all tables
with open("results/tables/benchmark_results.txt", "w") as f:
    for benchmark in benchmarks:
        print(f"\n{benchmark.upper()} Results:")
        f.write(f"\n{benchmark.upper()} Results:\n")
        
        table_data = []
        
        for model in models:
            row = [model["model_name"]]
            
            # Add values for each mode, leaving cell blank if data doesn't exist
            dim_value = None
            rdo_value = None
            
            for mode in ["original", "wandb", "baseline"]:
                key = f"{mode}_{benchmark}_acc"
                if model.get(key) is not None:
                    if mode == "original":
                        dim_value = model[key]
                        row.append(format_value(dim_value))
                    elif mode == "wandb":
                        rdo_value = model[key]
                        # Include difference between DIM and RDO
                        if dim_value is not None and rdo_value is not None:
                            diff = rdo_value - dim_value
                            sign = "+" if diff >= 0 else "-"
                            row.append(f"{format_value(rdo_value)} ({sign}{abs(diff)*100:.1f}%)")
                        else:
                            row.append(format_value(rdo_value))
                    else:
                        row.append(format_value(model[key]))
                else:
                    row.append("")
            
            # Only add row if at least one value exists
            if any(cell != "" for cell in row[1:]):
                table_data.append(row)
        
        # Create and print table for this benchmark
        if table_data:
            table = tabulate(table_data, headers=headers, tablefmt="grid")
            print(table)
            f.write(table + "\n\n")
        else:
            print(f"No data available for {benchmark}")
            f.write(f"No data available for {benchmark}\n\n")

# %%
with open("results/tables/benchmark_diffs.txt", "w") as f:
    for benchmark in benchmarks:
        print(f"\n{benchmark.upper()} Results:")
        f.write(f"\n{benchmark.upper()} Results:\n")
        
        table_data = []
        headers = ["Model", "DIM", "RDO"]
        
        for model in models:
            row = [model["model_name"]]
            
            # Get baseline value for comparison
            baseline_key = f"baseline_{benchmark}_acc"
            baseline_value = model.get(baseline_key)
            
            # Calculate differences from baseline for DIM and RDO
            dim_key = f"original_{benchmark}_acc"
            rdo_key = f"wandb_{benchmark}_acc"
            
            # Add DIM difference
            dim_diff = None
            if model.get(dim_key) is not None and baseline_value is not None:
                dim_diff = model[dim_key] - baseline_value
                row.append(f"{abs(dim_diff)*100:.1f}%")
            else:
                row.append("")
                
            # Add RDO difference
            if model.get(rdo_key) is not None and baseline_value is not None:
                rdo_diff = model[rdo_key] - baseline_value
                rdo_dim_diff = ""
                # Add difference between RDO and DIM
                if model.get(dim_key) is not None and dim_diff is not None:
                    diff = abs(rdo_diff) - abs(dim_diff)
                    sign = "+" if diff >= 0 else "-"
                    rdo_dim_diff = f" ({sign}{abs(diff)*100:.1f}%)"
                row.append(f"{abs(rdo_diff)*100:.1f}%{rdo_dim_diff}")
            else:
                row.append("")
            
            # Only add row if at least one value exists
            if any(cell != "" for cell in row[1:]):
                table_data.append(row)
        
        # Create and print table for this benchmark
        if table_data:
            table = tabulate(table_data, headers=headers, tablefmt="grid")
            print(table)
            f.write(table + "\n\n")
        else:
            print(f"No data available for {benchmark}")
            f.write(f"No data available for {benchmark}\n\n")


# %%
models = [
    dict(model_name="Gemma 2 2B", 
         model_id="google/gemma-2-2b-it"),
    dict(model_name="Gemma 2 9B",
         model_id="google/gemma-2-9b-it"),
    dict(model_name="Llama 3 8B",
         model_id="meta-llama/Meta-Llama-3-8B-Instruct"),
    dict(model_name="Qwen 2.5 1.5B",
         model_id="Qwen/Qwen2.5-1.5B-Instruct"),
    dict(model_name="Qwen 2.5 3B",
         model_id="Qwen/Qwen2.5-3B-Instruct"),
    dict(model_name="Qwen 2.5 7B",
         model_id="Qwen/Qwen2.5-7B-Instruct"),
    dict(model_name="Qwen 2.5 14B",
         model_id="Qwen/Qwen2.5-14B-Instruct"),
]

# load baseline scores
dim_dir = os.path.join("/ceph/hdd/students/elsj/paper_results/dim_directions")
for model in models:
    print(model)
    model_id = model["model_id"].split("/")[-1]
    datasets = ["jailbreakbench", "strongreject", "sorrybench", "xstest"]
    for dataset in datasets:
        key = 'StrongREJECT_score' if dataset != "xstest" else "xstest_judgements"
        path = os.path.join(dim_dir, f"{model_id}/completions/{dataset}_ablation_evaluations.json")
        with open(path, "r") as f:
            data = json.load(f)
        model[f"{dataset}_ablation_asr"] = data[key]
        path = os.path.join(dim_dir, f"{model_id}/completions/{dataset}_baseline_evaluations.json")
        with open(path, "r") as f:
            data = json.load(f)
        model[f"{dataset}_baseline_asr"] = data[key]
        path = os.path.join(dim_dir, f"{model_id}/completions/{dataset}_actadd_evaluations.json")
        with open(path, "r") as f:
            data = json.load(f)
        model[f"{dataset}_actadd_asr"] = data[key]
    path = os.path.join(dim_dir, f"{model_id}/completions/harmless_actadd_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["harmless_actadd_asr"] = data["substring_matching_success_rate"]
    path = os.path.join(dim_dir, f"{model_id}/completions/harmless_baseline_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["harmless_baseline_asr"] = data["substring_matching_success_rate"]
    path = os.path.join(dim_dir, f"{model_id}/direction.pt")
    refusal_direction = torch.load(path, map_location=torch.device('cpu'))
    model["refusal_direction"] = refusal_direction.clone()

    api = wandb.Api()
    groups = [
        dict(group="sb_data_with_retain", name="retain"),
    ]
    for group in groups:
        group_name = f"{group['group']}_{model_id}"
        run_name = "run_4" if (model_id == "Qwen2.5-3B-Instruct" and "retain" in group_name) else "run_5"
        runs = api.runs("refusal-representations/robust_refusal_vector", {"group": group_name, "display_name": run_name})
        if runs:
            run = runs[0]
            summary = run.summary._json_dict
            key = "retain_summary"
            model[f"{key}"] = summary
# %%
def plot_scores(y_label: str, dataset: str):
    # Extract scores from models
    dataset_key = dataset.lower().replace("-", "")
    ablation_scores = [model[f"{dataset_key}_ablation_asr"] for model in models]
    actadd_scores = [model[f"{dataset_key}_actadd_asr"] for model in models]
    baseline_scores = [model[f"{dataset_key}_baseline_asr"] for model in models]
    if 'xstest' in dataset_key: 
        retain_ablation_scores = [model[f"retain_summary"][f"{dataset_key}_ablation_xstest_judgements"] for model in models]
        retain_actadd_scores = [model[f"retain_summary"][f"{dataset_key}_actadd_xstest_judgements"] for model in models]
    else:
        retain_ablation_scores = [model[f"retain_summary"][f"{dataset_key}_ablation_StrongREJECT_score"] for model in models]
        retain_actadd_scores = [model[f"retain_summary"][f"{dataset_key}_actadd_StrongREJECT_score"] for model in models]
    model_names = [model["model_name"] for model in models]

    x = np.arange(len(models))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.12  # Reduced width to accommodate more bars

    # Create proxy artists for the legend
    from matplotlib.patches import Patch, Rectangle
    
    # Method labels (solid colors)
    method_patches = [
        Rectangle((0,0), 1, 1, facecolor='gray', label='Baseline\n(No Intervention)', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[0], label='DIM', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[1], label='RDO (Ours)', edgecolor='black', linewidth=0.5)
    ]
    
    # Condition labels (patterns)
    condition_patches = [
        Rectangle((0,0), 1, 1, facecolor='white', label='Directional\nAblation', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor='white', label='Activation\nSubtraction', edgecolor='black', linewidth=0.5, hatch='////')
    ]

    # Plot baseline scores
    ax.bar(x - 3*bar_width/2, baseline_scores, bar_width,
        color='gray',
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Plot ablation scores side by side
    ax.bar(x - bar_width/2, ablation_scores, bar_width,
        color=colors[0],
        alpha=1.0, edgecolor='black', linewidth=0.5)
    
    ax.bar(x + bar_width/2, retain_ablation_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Plot actadd scores side by side with hatching
    ax.bar(x + 3*bar_width/2, actadd_scores, bar_width,
        color=colors[0],
        alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

    ax.bar(x + 5*bar_width/2, retain_actadd_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

    # Customization
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='center')  # Rotated labels
    ax.tick_params(axis='both', which='major')

    # Grid styling
    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray', linewidth=0.5)

    # Adjust spines to be thinner
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Make tick marks thinner and shorter
    ax.tick_params(axis='both', width=0.5, length=3)

    # Y-axis limits and spines
    ax.set_ylim(0, 1)  # Slightly higher to accommodate error bars
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Create two-part legend with better positioning
    legend1 = ax.legend(handles=method_patches, title='Method', 
                    bbox_to_anchor=(1.0, 1.0),  # Position to the right
                    loc='upper left', ncol=1,
                    frameon=True, fancybox=False, 
                    edgecolor='black',
                    borderaxespad=0)
    ax.add_artist(legend1)  # Add first legend
    
    ax.legend(handles=condition_patches, title='Operation',
              bbox_to_anchor=(1.0, 0.5),  # Position to the right, below the first legend
              loc='upper left', ncol=1,
              frameon=True, fancybox=False,
              edgecolor='black', 
              borderaxespad=0)
    ax.set_title(f"{dataset}")
    # Save
    # plt.tight_layout()
    os.makedirs("results/plots/rdo/combined", exist_ok=True)
    path = f"results/plots/rdo/combined/{dataset}_scores.png"
    plt.savefig(path, dpi=300)
    plt.show()

plot_scores(y_label="Attack Success Rate", dataset="JailbreakBench")
plot_scores(y_label="Attack Success Rate", dataset="StrongREJECT")
plot_scores(y_label="Attack Success Rate", dataset="SORRY-Bench")
# plot_scores(y_label="Attack Success Rate", dataset="XSTest")

# %% 
# plot the asr but reversed and call it safety score
def plot_safety_scores(y_label: str):
    # Extract scores from models
    ablation_scores = [1 - model["ablation_asr"] for model in models]
    actadd_scores = [1 - model["actadd_asr"] for model in models]
    baseline_scores = [1 - model["baseline_asr"] for model in models]
    retain_ablation_scores = [1 - model[f"retain_summary"]["jailbreakbench_ablation_StrongREJECT_score"] for model in models]
    retain_actadd_scores = [1 - model[f"retain_summary"]["jailbreakbench_actadd_StrongREJECT_score"] for model in models]
    model_names = [model["model_name"] for model in models]

    x = np.arange(len(models))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.12  # Reduced width to accommodate more bars

    # Create proxy artists for the legend
    from matplotlib.patches import Patch, Rectangle
    
    # Method labels (solid colors)
    method_patches = [
        Rectangle((0,0), 1, 1, facecolor='gray', label='Baseline\n(No Intervention)', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[0], label='DIM', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[1], label='RDO (Ours)', edgecolor='black', linewidth=0.5)
    ]
    
    # Condition labels (patterns)
    condition_patches = [
        Rectangle((0,0), 1, 1, facecolor='white', label='Directional\nAblation', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor='white', label='Activation\nSubtraction', edgecolor='black', linewidth=0.5, hatch='////')
    ]

    # Plot baseline scores
    ax.bar(x - 3*bar_width/2, baseline_scores, bar_width,
        color='gray',
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Plot ablation scores side by side
    ax.bar(x - bar_width/2, ablation_scores, bar_width,
        color=colors[0],
        alpha=1.0, edgecolor='black', linewidth=0.5)
    
    ax.bar(x + bar_width/2, retain_ablation_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Plot actadd scores side by side with hatching
    ax.bar(x + 3*bar_width/2, actadd_scores, bar_width,
        color=colors[0],
        alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

    ax.bar(x + 5*bar_width/2, retain_actadd_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

    # Customization
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='center')  # Rotated labels
    ax.tick_params(axis='both', which='major')

    # Grid styling
    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray', linewidth=0.5)

    # Adjust spines to be thinner
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Make tick marks thinner and shorter
    ax.tick_params(axis='both', width=0.5, length=3)

    # Y-axis limits and spines
    ax.set_ylim(0, 1)  # Slightly higher to accommodate error bars
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Create two-part legend with better positioning
    legend1 = ax.legend(handles=method_patches, title='Method', 
                    bbox_to_anchor=(1.0, 1.0),  # Position to the right
                    loc='upper left', ncol=1,
                    frameon=True, fancybox=False, 
                    edgecolor='black',
                    borderaxespad=0)
    ax.add_artist(legend1)  # Add first legend
    
    ax.legend(handles=condition_patches, title='Operation',
              bbox_to_anchor=(1.0, 0.5),  # Position to the right, below the first legend
              loc='upper left', ncol=1,
              frameon=True, fancybox=False,
              edgecolor='black', 
              borderaxespad=0)

    # Save
    # plt.tight_layout()
    os.makedirs("results/plots/single_direction/combined", exist_ok=True)
    path = "results/plots/single_direction/combined/jailbreak_scores.png"
    plt.savefig(path, dpi=300)
    plt.show()

plot_safety_scores(y_label="Safety Score")

# %%
from tabulate import tabulate

def create_combined_table():
    headers = ["", "Safety - Directional Ablation", "", "", "Safety - Activation Subtraction", "", "", "General Capability", "", "", "", ""]
    subheaders = ["Model", "JBB\nASR ↑", "Strong\nREJECT\nASR ↑", "SORRY-Bench\nASR ↑", "JBB\nASR ↑", "Strong\nREJECT\nASR ↑", "SORRY-Bench\nASR ↑", "MMLU\nAcc ↑", "ARC-C\nAcc ↑", "GSM8K\nAcc ↑", "TruthfulQA\nAcc ↑", "Avg\nBenchmark ↑"]
    
    table_data = []
    
    # Add subheaders as first row
    table_data.append(subheaders)
    
    for model in models:
        model_name = model["model_name"]
        
        # Base row for the model
        row = [model_name]
        
        # Safety metrics - Directional Ablation (baseline)
        row.append(f"{model.get('jailbreakbench_baseline_asr', 0)*100:.1f}")
        row.append(f"{model.get('strongreject_baseline_asr', 0)*100:.1f}")
        row.append(f"{model.get('sorrybench_baseline_asr', 0)*100:.1f}")
        
        # Safety metrics - Activation Subtraction (baseline)
        # Leave these columns empty for the baseline row
        row.append("")
        row.append("")
        row.append("")
        
        # General capability metrics (baseline)
        mmlu = model.get('baseline_mmlu_acc', 0)
        arc = model.get('baseline_arc_challenge_acc', 0)
        gsm8k = model.get('baseline_gsm8k_acc', 0)
        truthfulqa = model.get('baseline_truthfulqa_mc2_acc', 0)
        
        # Calculate average benchmark score
        avg_benchmark = (mmlu + arc + gsm8k + truthfulqa) / 4
        
        row.append(f"{mmlu*100:.1f}")
        row.append(f"{arc*100:.1f}")
        row.append(f"{gsm8k*100:.1f}")
        row.append(f"{truthfulqa*100:.1f}")
        row.append(f"{avg_benchmark*100:.1f}")
        
        table_data.append(row)
        
        # Row for system prompt (DIM)
        row_dim = ["DIM"]
        
        # DIM safety metrics - Directional Ablation
        row_dim.append(f"{model.get('jailbreakbench_ablation_asr', 0)*100:.1f}")
        row_dim.append(f"{model.get('strongreject_ablation_asr', 0)*100:.1f}")
        row_dim.append(f"{model.get('sorrybench_ablation_asr', 0)*100:.1f}")
        
        # DIM safety metrics - Activation Subtraction
        row_dim.append(f"{model.get('jailbreakbench_actadd_asr', 0)*100:.1f}")
        row_dim.append(f"{model.get('strongreject_actadd_asr', 0)*100:.1f}")
        row_dim.append(f"{model.get('sorrybench_actadd_asr', 0)*100:.1f}")
        
        # DIM capability metrics
        dim_mmlu = model.get('original_mmlu_acc', 0)
        dim_arc = model.get('original_arc_challenge_acc', 0)
        dim_gsm8k = model.get('original_gsm8k_acc', 0)
        dim_truthfulqa = model.get('original_truthfulqa_mc2_acc', 0)
        
        # Calculate average benchmark score for DIM
        dim_avg_benchmark = (dim_mmlu + dim_arc + dim_gsm8k + dim_truthfulqa) / 4
        
        row_dim.append(f"{dim_mmlu*100:.1f}")
        row_dim.append(f"{dim_arc*100:.1f}")
        row_dim.append(f"{dim_gsm8k*100:.1f}")
        row_dim.append(f"{dim_truthfulqa*100:.1f}")
        row_dim.append(f"{dim_avg_benchmark*100:.1f}")
        
        table_data.append(row_dim)
        
        # Row for vector ablation (RDO)
        row_rdo = ["RDO"]
        
        # Get RDO safety metrics - Directional Ablation
        if "retain_summary" in model:
            row_rdo.append(f"{model['retain_summary'].get('jailbreakbench_ablation_StrongREJECT_score', 0)*100:.1f}")
            row_rdo.append(f"{model['retain_summary'].get('strongreject_ablation_StrongREJECT_score', 0)*100:.1f}")
            row_rdo.append(f"{model['retain_summary'].get('sorrybench_ablation_StrongREJECT_score', 0)*100:.1f}")
        else:
            row_rdo.extend(["N/A", "N/A", "N/A"])
        
        # Get RDO safety metrics - Activation Subtraction
        if "retain_summary" in model:
            row_rdo.append(f"{model['retain_summary'].get('jailbreakbench_actadd_StrongREJECT_score', 0)*100:.1f}")
            row_rdo.append(f"{model['retain_summary'].get('strongreject_actadd_StrongREJECT_score', 0)*100:.1f}")
            row_rdo.append(f"{model['retain_summary'].get('sorrybench_actadd_StrongREJECT_score', 0)*100:.1f}")
        else:
            row_rdo.extend(["N/A", "N/A", "N/A"])
        
        # RDO capability metrics
        rdo_mmlu = model.get('wandb_mmlu_acc', 0)
        rdo_arc = model.get('wandb_arc_challenge_acc', 0)
        rdo_gsm8k = model.get('wandb_gsm8k_acc', 0)
        rdo_truthfulqa = model.get('wandb_truthfulqa_mc2_acc', 0)
        
        # Calculate average benchmark score for RDO
        rdo_avg_benchmark = (rdo_mmlu + rdo_arc + rdo_gsm8k + rdo_truthfulqa) / 4
        
        row_rdo.append(f"{rdo_mmlu*100:.1f}")
        row_rdo.append(f"{rdo_arc*100:.1f}")
        row_rdo.append(f"{rdo_gsm8k*100:.1f}")
        row_rdo.append(f"{rdo_truthfulqa*100:.1f}")
        row_rdo.append(f"{rdo_avg_benchmark*100:.1f}")
        
        table_data.append(row_rdo)
        
        # Add separator line between models
        if model != models[-1]:
            table_data.append([""] * len(headers))
    
    # Create and print the table
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print(table)
    
    # Save the table to a file
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/combined_evaluation.txt", "w") as f:
        f.write(table)

# Call the function to create the table
create_combined_table()

# %%
def create_latex_table():
    headers = ["", "Safety - Directional Ablation", "", "", "Safety - Activation Subtraction", "", "", "General Capability", "", "", "", ""]
    subheaders = ["Model", "JBB\nASR ↑", "Strong\nREJECT\nASR ↑", "SORRY-Bench\nASR ↑", "JBB\nASR ↑", "Strong\nREJECT\nASR ↑", "SORRY-Bench\nASR ↑", "MMLU\nAcc ↑", "ARC-C\nAcc ↑", "GSM8K\nAcc ↑", "TruthfulQA\nAcc ↑", "Avg\nBenchmark ↑"]
    
    # Create column specification: first column left-aligned, rest centered
    column_spec = "|l|" + "c|" * 11
    
    # Start creating the LaTeX table
    latex_table = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",  # Use small font to fit the table
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\hline"
    ]
    
    # Add multicolumn headers
    multicolumn_headers = []
    
    # First cell is empty
    multicolumn_headers.append("")
    
    # Safety - Directional Ablation spans 3 columns
    multicolumn_headers.append("\\multicolumn{3}{c|}{Safety - Directional Ablation}")
    
    # Safety - Activation Subtraction spans 3 columns
    multicolumn_headers.append("\\multicolumn{3}{c|}{Safety - Activation Subtraction}")
    
    # General Capability spans 5 columns
    multicolumn_headers.append("\\multicolumn{5}{c|}{General Capability}")
    
    # Join the multicolumn headers with & and add to table
    latex_table.append(" & ".join(multicolumn_headers) + " \\\\")
    latex_table.append("\\hline")
    
    # Convert subheaders (replace newlines with LaTeX line breaks)
    formatted_subheaders = [subheader.replace("\n", " \\\\") for subheader in subheaders]
    latex_table.append(" & ".join(formatted_subheaders) + " \\\\")
    latex_table.append("\\hline")
    
    # Assume models is defined elsewhere in your code
    # For each model, create 3 rows: base model, DIM, and RDO
    for model in models:
        model_name = model["model_name"]
        
        # Base row for the model
        row = [model_name]
        
        # Safety metrics - Directional Ablation (baseline)
        row.append(f"{model.get('jailbreakbench_baseline_asr', 0)*100:.1f}")
        row.append(f"{model.get('strongreject_baseline_asr', 0)*100:.1f}")
        row.append(f"{model.get('sorrybench_baseline_asr', 0)*100:.1f}")
        
        # Safety metrics - Activation Subtraction (baseline)
        # Leave these columns empty for the baseline row
        row.append("")
        row.append("")
        row.append("")
        
        # General capability metrics (baseline)
        mmlu = model.get('baseline_mmlu_acc', 0)
        arc = model.get('baseline_arc_challenge_acc', 0)
        gsm8k = model.get('baseline_gsm8k_acc', 0)
        truthfulqa = model.get('baseline_truthfulqa_mc2_acc', 0)
        
        # Calculate average benchmark score
        avg_benchmark = (mmlu + arc + gsm8k + truthfulqa) / 4
        
        row.append(f"{mmlu*100:.1f}")
        row.append(f"{arc*100:.1f}")
        row.append(f"{gsm8k*100:.1f}")
        row.append(f"{truthfulqa*100:.1f}")
        row.append(f"{avg_benchmark*100:.1f}")
        
        latex_table.append(" & ".join(row) + " \\\\")
        
        # Row for system prompt (DIM)
        row_dim = ["DIM"]
        
        # DIM safety metrics - Directional Ablation
        row_dim.append(f"{model.get('jailbreakbench_ablation_asr', 0)*100:.1f}")
        row_dim.append(f"{model.get('strongreject_ablation_asr', 0)*100:.1f}")
        row_dim.append(f"{model.get('sorrybench_ablation_asr', 0)*100:.1f}")
        
        # DIM safety metrics - Activation Subtraction
        row_dim.append(f"{model.get('jailbreakbench_actadd_asr', 0)*100:.1f}")
        row_dim.append(f"{model.get('strongreject_actadd_asr', 0)*100:.1f}")
        row_dim.append(f"{model.get('sorrybench_actadd_asr', 0)*100:.1f}")
        
        # DIM capability metrics
        dim_mmlu = model.get('original_mmlu_acc', 0)
        dim_arc = model.get('original_arc_challenge_acc', 0)
        dim_gsm8k = model.get('original_gsm8k_acc', 0)
        dim_truthfulqa = model.get('original_truthfulqa_mc2_acc', 0)
        
        # Calculate average benchmark score for DIM
        dim_avg_benchmark = (dim_mmlu + dim_arc + dim_gsm8k + dim_truthfulqa) / 4
        
        row_dim.append(f"{dim_mmlu*100:.1f}")
        row_dim.append(f"{dim_arc*100:.1f}")
        row_dim.append(f"{dim_gsm8k*100:.1f}")
        row_dim.append(f"{dim_truthfulqa*100:.1f}")
        row_dim.append(f"{dim_avg_benchmark*100:.1f}")
        
        latex_table.append(" & ".join(row_dim) + " \\\\")
        
        # Row for vector ablation (RDO)
        row_rdo = ["RDO"]
        
        # Get RDO safety metrics - Directional Ablation
        if "retain_summary" in model:
            row_rdo.append(f"{model['retain_summary'].get('jailbreakbench_ablation_StrongREJECT_score', 0)*100:.1f}")
            row_rdo.append(f"{model['retain_summary'].get('strongreject_ablation_StrongREJECT_score', 0)*100:.1f}")
            row_rdo.append(f"{model['retain_summary'].get('sorrybench_ablation_StrongREJECT_score', 0)*100:.1f}")
        else:
            row_rdo.extend(["N/A", "N/A", "N/A"])
        
        # Get RDO safety metrics - Activation Subtraction
        if "retain_summary" in model:
            row_rdo.append(f"{model['retain_summary'].get('jailbreakbench_actadd_StrongREJECT_score', 0)*100:.1f}")
            row_rdo.append(f"{model['retain_summary'].get('strongreject_actadd_StrongREJECT_score', 0)*100:.1f}")
            row_rdo.append(f"{model['retain_summary'].get('sorrybench_actadd_StrongREJECT_score', 0)*100:.1f}")
        else:
            row_rdo.extend(["N/A", "N/A", "N/A"])
        
        # RDO capability metrics
        rdo_mmlu = model.get('wandb_mmlu_acc', 0)
        rdo_arc = model.get('wandb_arc_challenge_acc', 0)
        rdo_gsm8k = model.get('wandb_gsm8k_acc', 0)
        rdo_truthfulqa = model.get('wandb_truthfulqa_mc2_acc', 0)
        
        # Calculate average benchmark score for RDO
        rdo_avg_benchmark = (rdo_mmlu + rdo_arc + rdo_gsm8k + rdo_truthfulqa) / 4
        
        row_rdo.append(f"{rdo_mmlu*100:.1f}")
        row_rdo.append(f"{rdo_arc*100:.1f}")
        row_rdo.append(f"{rdo_gsm8k*100:.1f}")
        row_rdo.append(f"{rdo_truthfulqa*100:.1f}")
        row_rdo.append(f"{rdo_avg_benchmark*100:.1f}")
        
        latex_table.append(" & ".join(row_rdo) + " \\\\")
        
        # Add hline between models
        if model != models[-1]:
            latex_table.append("\\hline")
    
    # Complete the table
    latex_table.append("\\hline")
    latex_table.append("\\end{tabular}")
    latex_table.append("\\caption{Combined evaluation of safety and capability metrics across different models.}")
    latex_table.append("\\label{tab:combined_evaluation}")
    latex_table.append("\\end{table}")
    
    # Join all lines to create the final LaTeX table
    final_latex = "\n".join(latex_table)
    
    # Print and save the LaTeX table
    print(final_latex)
    
    # Optional: Save to file
    import os
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/combined_evaluation.tex", "w") as f:
        f.write(final_latex)

# Call the function to create the LaTeX table
create_latex_table()

# %%
def create_latex_table():
    # Start of the LaTeX table
    latex_output = [
        "\\begin{table*}[htpb]",
        "\\small",
        "\\centering",
        "\\begin{tabular}{ l c c c c c c c }",
        "\\toprule"
    ]
    
    # Multirow headers
    latex_output.append("\\multirow{2}{*}{ } & \\multicolumn{3}{c}{\\textbf{Jailbreaking}} & " +
                       "\\multicolumn{4}{c}{\\textbf{General Capability}} \\\\")
    
    # Cmidrule separators
    latex_output.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-8}")
    
    # Subheaders
    latex_output.append(" & \\textbf{JailbreakBench} & \\textbf{StrongREJECT} & \\textbf{SORRY-Bench} & " +
                       "\\textbf{MMLU} & \\textbf{ARC-C} & \\textbf{GSM8K} & \\textbf{TruthfulQA} \\\\")
    
    # Direction indicators
    latex_output.append(" & ASR \\(\\uparrow\\) & ASR \\(\\uparrow\\) & ASR \\(\\uparrow\\) & " +
                       "Acc \\(\\uparrow\\) & Acc \\(\\uparrow\\) & Acc \\(\\uparrow\\) & Acc \\(\\uparrow\\) \\\\")
    
    # Midrule
    latex_output.append("\\midrule")
    
    for model in models:
        model_name = model["model_name"]
        
        # Base model row
        base_mmlu = model.get('baseline_mmlu_acc', 0)*100
        base_arc = model.get('baseline_arc_challenge_acc', 0)*100
        base_gsm8k = model.get('baseline_gsm8k_acc', 0)*100
        base_truthfulqa = model.get('baseline_truthfulqa_mc2_acc', 0)*100
        avg_benchmark = (base_mmlu + base_arc + base_gsm8k + base_truthfulqa) / 4
        
        latex_output.append(f"\\textsc{{{model_name}}} & " +
                          f"{model.get('jailbreakbench_baseline_asr', 0)*100:.1f} & " +
                          f"{model.get('strongreject_baseline_asr', 0)*100:.1f} & " +
                          f"{model.get('sorrybench_baseline_asr', 0)*100:.1f} & " +
                          f"{base_mmlu:.1f} & " +
                          f"{base_arc:.1f} & " +
                          f"{base_gsm8k:.1f} & " +
                          f"{base_truthfulqa:.1f} \\\\")
        
        # DIM row (system prompt)
        dim_jbb_dir = model.get('jailbreakbench_ablation_asr', 0)*100
        dim_jbb_act = model.get('jailbreakbench_actadd_asr', 0)*100
        dim_strong_dir = model.get('strongreject_ablation_asr', 0)*100
        dim_strong_act = model.get('strongreject_actadd_asr', 0)*100
        dim_sorry_dir = model.get('sorrybench_ablation_asr', 0)*100
        dim_sorry_act = model.get('sorrybench_actadd_asr', 0)*100
        
        dim_mmlu = model.get('original_mmlu_acc', 0)*100
        dim_arc = model.get('original_arc_challenge_acc', 0)*100
        dim_gsm8k = model.get('original_gsm8k_acc', 0)*100
        dim_truthfulqa = model.get('original_truthfulqa_mc2_acc', 0)*100
        
        # RDO values for comparison
        rdo_jbb_dir = 0
        rdo_jbb_act = 0
        rdo_strong_dir = 0
        rdo_strong_act = 0
        rdo_sorry_dir = 0
        rdo_sorry_act = 0
        rdo_mmlu = model.get('wandb_mmlu_acc', 0)*100
        rdo_arc = model.get('wandb_arc_challenge_acc', 0)*100
        rdo_gsm8k = model.get('wandb_gsm8k_acc', 0)*100
        rdo_truthfulqa = model.get('wandb_truthfulqa_mc2_acc', 0)*100
        
        if "retain_summary" in model:
            rdo_jbb_dir = model['retain_summary'].get('jailbreakbench_ablation_StrongREJECT_score', 0)*100
            rdo_jbb_act = model['retain_summary'].get('jailbreakbench_actadd_StrongREJECT_score', 0)*100
            rdo_strong_dir = model['retain_summary'].get('strongreject_ablation_StrongREJECT_score', 0)*100
            rdo_strong_act = model['retain_summary'].get('strongreject_actadd_StrongREJECT_score', 0)*100
            rdo_sorry_dir = model['retain_summary'].get('sorrybench_ablation_StrongREJECT_score', 0)*100
            rdo_sorry_act = model['retain_summary'].get('sorrybench_actadd_StrongREJECT_score', 0)*100
        
        # Bold DIM values if better than RDO for safety metrics
        dim_jbb_dir_text = f"\\textbf{{{dim_jbb_dir:.1f}}}" if dim_jbb_dir > rdo_jbb_dir else f"{dim_jbb_dir:.1f}"
        dim_jbb_act_text = f"\\textbf{{{dim_jbb_act:.1f}}}" if dim_jbb_act > rdo_jbb_act else f"{dim_jbb_act:.1f}"
        dim_strong_dir_text = f"\\textbf{{{dim_strong_dir:.1f}}}" if dim_strong_dir > rdo_strong_dir else f"{dim_strong_dir:.1f}"
        dim_strong_act_text = f"\\textbf{{{dim_strong_act:.1f}}}" if dim_strong_act > rdo_strong_act else f"{dim_strong_act:.1f}"
        dim_sorry_dir_text = f"\\textbf{{{dim_sorry_dir:.1f}}}" if dim_sorry_dir > rdo_sorry_dir else f"{dim_sorry_dir:.1f}"
        dim_sorry_act_text = f"\\textbf{{{dim_sorry_act:.1f}}}" if dim_sorry_act > rdo_sorry_act else f"{dim_sorry_act:.1f}"
        
        # For capability metrics, bold the highest score
        dim_mmlu_text = f"\\textbf{{{dim_mmlu:.1f}}}" if dim_mmlu > rdo_mmlu else f"{dim_mmlu:.1f}"
        dim_arc_text = f"\\textbf{{{dim_arc:.1f}}}" if dim_arc > rdo_arc else f"{dim_arc:.1f}"
        dim_gsm8k_text = f"\\textbf{{{dim_gsm8k:.1f}}}" if dim_gsm8k > rdo_gsm8k else f"{dim_gsm8k:.1f}"
        dim_truthfulqa_text = f"\\textbf{{{dim_truthfulqa:.1f}}}" if dim_truthfulqa > rdo_truthfulqa else f"{dim_truthfulqa:.1f}"
        
        latex_output.append(f"\\multicolumn{{1}}{{c}}{{\\: DIM}} & " +
                          f"{dim_jbb_dir_text} / {dim_jbb_act_text} & " +
                          f"{dim_strong_dir_text} / {dim_strong_act_text} & " +
                          f"{dim_sorry_dir_text} / {dim_sorry_act_text} & " +
                          f"{dim_mmlu_text} & " +
                          f"{dim_arc_text} & " +
                          f"{dim_gsm8k_text} & " +
                          f"{dim_truthfulqa_text} \\\\")
        
        # RDO row (vector ablation)
        if "retain_summary" in model:
            jbb_dir_text = f"\\textbf{{{rdo_jbb_dir:.1f}}}" if rdo_jbb_dir > dim_jbb_dir else f"{rdo_jbb_dir:.1f}"
            jbb_act_text = f"\\textbf{{{rdo_jbb_act:.1f}}}" if rdo_jbb_act > dim_jbb_act else f"{rdo_jbb_act:.1f}"
            strong_dir_text = f"\\textbf{{{rdo_strong_dir:.1f}}}" if rdo_strong_dir > dim_strong_dir else f"{rdo_strong_dir:.1f}"
            strong_act_text = f"\\textbf{{{rdo_strong_act:.1f}}}" if rdo_strong_act > dim_strong_act else f"{rdo_strong_act:.1f}"
            sorry_dir_text = f"\\textbf{{{rdo_sorry_dir:.1f}}}" if rdo_sorry_dir > dim_sorry_dir else f"{rdo_sorry_dir:.1f}"
            sorry_act_text = f"\\textbf{{{rdo_sorry_act:.1f}}}" if rdo_sorry_act > dim_sorry_act else f"{rdo_sorry_act:.1f}"
            
            jbb_cell = f"{jbb_dir_text} / {jbb_act_text}"
            strong_cell = f"{strong_dir_text} / {strong_act_text}"
            sorry_cell = f"{sorry_dir_text} / {sorry_act_text}"
        else:
            jbb_cell = "N/A"
            strong_cell = "N/A"
            sorry_cell = "N/A"
        
        mmlu_text = f"\\textbf{{{rdo_mmlu:.1f}}}" if rdo_mmlu > dim_mmlu else f"{rdo_mmlu:.1f}"
        arc_text = f"\\textbf{{{rdo_arc:.1f}}}" if rdo_arc > dim_arc else f"{rdo_arc:.1f}"
        gsm8k_text = f"\\textbf{{{rdo_gsm8k:.1f}}}" if rdo_gsm8k > dim_gsm8k else f"{rdo_gsm8k:.1f}"
        truthfulqa_text = f"\\textbf{{{rdo_truthfulqa:.1f}}}" if rdo_truthfulqa > dim_truthfulqa else f"{rdo_truthfulqa:.1f}"
        
        latex_output.append(f"\\multicolumn{{1}}{{c}}{{\\: RDO}} & " +
                           f"{jbb_cell} & {strong_cell} & {sorry_cell} & " +
                           f"{mmlu_text} & {arc_text} & {gsm8k_text} & {truthfulqa_text} \\\\")
        
        # Add midrule between models (except after the last one)
        if model != models[-1]:
            latex_output.append("\\midrule")
    
    # Bottom rule and table end
    latex_output.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "\\caption{Combined evaluation results showing safety metrics (DirectionalAblation/ActivationSubtraction) and general capability benchmarks. Each safety cell contains two values: the first for directional ablation and the second for activation subtraction.}",
        "\\label{tab:combined_evaluation}",
        "\\end{table*}"
    ])
    
    # Join and print the LaTeX code
    latex_table = "\n".join(latex_output)
    print(latex_table)
    
    # Save the LaTeX table to a file
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/combined_evaluation.tex", "w") as f:
        f.write(latex_table)

# Call the function to create the LaTeX table
create_latex_table()

# %%
def plot_harmless_scores(y_label: str):
    # Extract scores from models
    ablation_scores = [1 - model["harmless_actadd_asr"] for model in models]  # Changed to harmless scores
    retain_ablation_scores = [1 - model[f"retain_summary"]["harmless_actadd_substring_matching_success_rate"] for model in models]

    model_names = [model["model_name"] for model in models]

    x = np.arange(len(models))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.15  # Slightly wider since we have fewer bars

    # Create proxy artists for the legend
    from matplotlib.patches import Patch, Rectangle
    
    # Method labels (solid colors)
    method_patches = [
        Rectangle((0,0), 1, 1, facecolor=colors[0], label='DIM', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[1], label='RDO (Ours)', edgecolor='black', linewidth=0.5)
    ]

    # Plot bars without transparency
    ax.bar(x - bar_width/2, ablation_scores, bar_width, 
        color=colors[0], 
        alpha=1.0, edgecolor='black', linewidth=0.5)
    
    ax.bar(x + bar_width/2, retain_ablation_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Customization
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='center')  # Rotated labels
    ax.tick_params(axis='both', which='major')

    # Grid styling
    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray', linewidth=0.5)

    # Adjust spines to be thinner
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Make tick marks thinner and shorter
    ax.tick_params(axis='both', width=0.5, length=3)

    # Y-axis limits and spines
    ax.set_ylim(0, 1)  # Slightly higher to accommodate error bars
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Create legend
    legend = ax.legend(handles=method_patches,
                    bbox_to_anchor=(0.5, 1.15),
                    loc='center', ncol=2,
                    frameon=True, fancybox=False, 
                    edgecolor='black',
                    borderaxespad=0)

    # Save
    os.makedirs("results/plots/single_direction/combined", exist_ok=True)
    path = "results/plots/single_direction/combined/harmless_scores.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')

# Call the function
plot_harmless_scores(y_label="Refusal Score")
# %%
print(models[-1])
# %%
for model in models:
    print(model["model_name"], model["harmless_actadd_asr"])
# %%
# Create table of cosine similarities between trained and baseline directions
# from tabulate import tabulate

# similarities = []
# for model in models:
#     sim = torch.nn.functional.cosine_similarity(
#         model["trained_refusal_direction"], 
#         model["refusal_direction"], 
#         dim=-1
#     )
#     similarities.append([model["model_name"], sim.item()])

# print(tabulate(similarities, 
#               headers=["Model", "Cosine Similarity"],
#               tablefmt="grid",
#               floatfmt=".3f"))

# %%
# %%
models = [
    # dict(model_name="Gemma 2 2B", 
    #      model_id="google/gemma-2-2b-it"),
    # dict(model_name="Gemma 2 9B",
    #      model_id="google/gemma-2-9b-it"),
    dict(model_name="Qwen 2.5 1.5B",
         model_id="Qwen/Qwen2.5-1.5B-Instruct"),
    dict(model_name="Qwen 2.5 3B",
         model_id="Qwen/Qwen2.5-3B-Instruct"),
    dict(model_name="Qwen 2.5 7B",
         model_id="Qwen/Qwen2.5-7B-Instruct"),
    dict(model_name="Qwen 2.5 14B",
         model_id="Qwen/Qwen2.5-14B-Instruct"),
    # dict(model_name="Llama 3 8B",
    #      model_id="meta-llama/Meta-Llama-3-8B-Instruct"),
]
base_dir = "/ceph/hdd/students/elsj/paper_results/subspace_samples_eval"
api = wandb.Api()

# Get all runs once
base_runs = {}
for model in models:
    model_id = model["model_id"].split("/")[-1]
    base_group_name = "join_subspace"
    # base_group_name = "repind_subspace_retain1_repind200_fixed_sum_no_dim"
    base_runs[model_id] = api.runs("refusal-representations/robust_refusal_subspace", 
                                 {"group": f"{base_group_name}_{model_id}"})

for model in models:
    model_id = model["model_id"].split("/")[-1]
    for dim in range(2, 9):
        matching_runs = []
        for r in base_runs[model_id]:
            if r.display_name == f"dim_{dim}":
                matching_runs.append(r)
        if not matching_runs:
            print(f"No runs found for dim_{dim} for model {model_id}")
            continue
            
        run = matching_runs[-1]
        summary = run.summary._json_dict
        if "candidate_idx" not in summary and "use_repind_loss" not in summary:
            print(f"No candidate_idx found for dim_{dim} for model {model_id}")
            continue
            
        basis_scores = []
        basis_scores_harmless = []
        for basis_dim in range(1, dim + 1):
            basis_scores.append(summary[f"jailbreakbench_ablation_basis_{basis_dim}_StrongREJECT_score"])
            basis_scores_harmless.append(summary[f"harmless_actadd_basis_{basis_dim}_substring_matching_success_rate"])
        model[f"dim_{dim}_basis"] = basis_scores
        model[f"dim_{dim}_basis_harmless"] = basis_scores_harmless

# %%
import os
import json
from concurrent.futures import ThreadPoolExecutor

def read_file_field(filepath, field):
    with open(filepath) as f:
        data = json.load(f)
    return data[field]

def load_evaluations(model_path):
    """Load all evaluation files for a model path concurrently and efficiently."""
    model_data = {}
    missing_samples = {}
    
    with ThreadPoolExecutor() as executor:
        futures = []
        # Iterate through dimensions we expect to find (dim 2 to 8)
        for dim in range(2, 9):
            attack_key = f"dim_{dim}_samples"
            harmless_key = f"dim_{dim}_samples_harmless"
            model_data[attack_key] = [None] * 256
            model_data[harmless_key] = [None] * 256
            missing_samples[attack_key] = set(range(1, 257))
            missing_samples[harmless_key] = set(range(1, 257))
            
            # List the completions directory once per dimension
            completions_dir = os.path.join(model_path, f"dim_{dim}", "completions")
            if not os.path.isdir(completions_dir):
                continue
            try:
                files = os.listdir(completions_dir)
            except Exception as e:
                print(f"Error listing {completions_dir}: {e}")
                continue
            
            for file in files:
                # Process attack samples
                if file.startswith("jailbreakbench_ablation_sample_") and file.endswith("_evaluations.json"):
                    sample_str = file[len("jailbreakbench_ablation_sample_"):-len("_evaluations.json")]
                    try:
                        sample_num = int(sample_str)
                    except ValueError:
                        continue
                    file_path = os.path.join(completions_dir, file)
                    futures.append((executor.submit(read_file_field, file_path, "StrongREJECT_score"), attack_key, sample_num))
                # Process harmless samples
                elif file.startswith("harmless_actadd_sample_") and file.endswith("_evaluations.json"):
                    sample_str = file[len("harmless_actadd_sample_"):-len("_evaluations.json")]
                    try:
                        sample_num = int(sample_str)
                    except ValueError:
                        continue
                    file_path = os.path.join(completions_dir, file)
                    futures.append((executor.submit(read_file_field, file_path, "substring_matching_success_rate"), harmless_key, sample_num))
        
        # Gather results from all concurrent file reads
        for future, key, sample_num in futures:
            try:
                value = future.result()
            except Exception:
                value = None
            if value is not None:
                if 1 <= sample_num <= len(model_data[key]):
                    model_data[key][sample_num - 1] = value
                    missing_samples[key].discard(sample_num)
                else:
                    print(f"Warning: sample number {sample_num} is out of range for {key} in {model_path}. Skipping assignment.")
    
    # Remove dimensions with no data and filter out missing entries
    keys_to_remove = []
    for key in list(model_data.keys()):
        if all(x is None for x in model_data[key]):
            print(f"No data found for {key} for {model_path}")
            keys_to_remove.append(key)
            missing_samples.pop(key, None)
        else:
            model_data[key] = [s for s in model_data[key] if s is not None]
    for key in keys_to_remove:
        model_data.pop(key, None)
    
    # Report any missing samples
    for key, missing in missing_samples.items():
        if missing:
            model_name = os.path.basename(model_path)
            print(f"Model {model_name} - Missing samples for {key}: {sorted(missing)}")
            
    return model_data

# Process each model
for model in models:
    model_id = model["model_id"].split("/")[-1]
    dir_name = f"join_subspace_{model_id}" if "repind" not in base_group_name else f"repind_subspace_retain1_repind200_fixed_sum_no_dim_{model_id}"
    samples_folder = os.path.join(base_dir, dir_name)
    
    # Load all evaluations for this model concurrently
    model_data = load_evaluations(samples_folder)
    
    if not any(key.endswith("_samples") or key.endswith("_samples_harmless") for key in model_data.keys()):
        print(f"Model {model['model_name']} Warning: No samples found")
        continue
        
    model.update(model_data)
    
    for key, value in model.items():
        if key.endswith("_samples") or key.endswith("_samples_harmless"):
            if len(value) != 256:
                print(f"Model {model['model_name']} Warning: {key} has {len(value)} samples instead of expected 256")

# %%
for model in models:
    print(model)
    model_id = model["model_id"].split("/")[-1]
    path = os.path.join(f"results/refusal_dir/{model_id}/completions/jailbreakbench_ablation_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["ablation_asr"] = data["StrongREJECT_score"]
    model["dim_1_basis"] = [data["StrongREJECT_score"]]  # Add dim 1 baseline
    path = os.path.join(f"results/refusal_dir/{model_id}/completions/jailbreakbench_baseline_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["baseline_asr"] = data["StrongREJECT_score"]
    path = os.path.join(f"results/refusal_dir/{model_id}/completions/jailbreakbench_actadd_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["actadd_asr"] = data["StrongREJECT_score"]
    path = os.path.join(f"results/refusal_dir/{model_id}/completions/harmless_actadd_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["harmless_actadd_asr"] = data["substring_matching_success_rate"]
    model["dim_1_basis_harmless"] = [data["substring_matching_success_rate"]]  # Add dim 1 harmless baseline
    path = os.path.join(f"results/refusal_dir/{model_id}/completions/harmless_baseline_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["harmless_baseline_asr"] = data["substring_matching_success_rate"]
    path = os.path.join(f"results/refusal_dir/{model_id}/direction.pt")
    refusal_direction = torch.load(path, map_location=torch.device('cpu'))
    model["refusal_direction"] = refusal_direction.clone()

    api = wandb.Api()
    groups = [
        dict(group="sb_data_with_retain", name="retain"),
    ]
    for group in groups:
        group_name = f"{group['group']}_{model_id}"
        run_name = "run_4" if (model_id == "Qwen2.5-3B-Instruct" and "retain" in group_name) else "run_5"
        runs = api.runs("refusal-representations/robust_refusal_vector", {"group": group_name, "display_name": run_name})
        if runs:
            run = runs[0]
            summary = run.summary._json_dict
            key = "retain_summary"
            model[f"{key}"] = summary

# %%
for key in models[0].keys():
    print(key)
# %%
# Find min and max dimensions from available data
# Group models by family
# Configure font sizes

model_families = {}
for model in models:
    family = model["model_id"].split("/")[0]
    # Rename google family to Gemma
    if family == "google":
        family = "Gemma"
    elif family == "meta-llama":
        family = "Llama 3.1"
    if family not in model_families:
        model_families[family] = []
    model_families[family].append(model)

# Create a plot for each model family
min_dim = 2
qwen_max_dim = 7
gemma_max_dim = 5
llama_max_dim = 5
for family, family_models in model_families.items():
    # Find min and max dimensions from available data across all models in family
    if family == "Qwen":
        max_dim = qwen_max_dim
    elif family == "Gemma":
        max_dim = gemma_max_dim
    elif family == "Llama 3.1":
        max_dim = llama_max_dim
    subspace_dims = list(range(min_dim, max_dim + 1))

    # Define colors
    # model_colors = ['#E67E22'] 

    # Create attack success plot - make it wider
    plot_size = (12, 5)  # Increased width to accommodate right legend
    fig1 = plt.figure(figsize=plot_size)
    ax1 = fig1.add_subplot(111)

    # Create harmless scores plot
    fig2 = plt.figure(figsize=plot_size)
    ax2 = fig2.add_subplot(111)

    # Increase group spacing
    group_spacing = 1.25
    box_spacing = 0.2

    # Plot each model in the family
    for model_idx, model in enumerate(family_models):
        # Calculate offset for this model's boxes
        offset = model_idx * box_spacing - (len(family_models)-1) * box_spacing/2
        positions = [x * group_spacing + offset for x in range(len(subspace_dims))]

        if "dim_1_basis" in model and "repind" not in base_group_name:
            # Plot dim 1 star with model color and black edge
            dim1_pos = -group_spacing + offset
            ax1.scatter([dim1_pos], model["dim_1_basis"], 
                    color=colors[model_idx],
                    marker='o',
                    s=10,
                    zorder=2,
                    alpha=1.0,
                    edgecolors='black',
                    linewidth=0.5)
            ax2.scatter([dim1_pos], [1 - x for x in model["dim_1_basis_harmless"]], 
                    color=colors[model_idx],
                    marker='o',
                    s=10,
                    zorder=2,
                    alpha=1.0,
                    edgecolors='black',
                    linewidth=0.5)

        # Extract performances for both plots
        performances = []
        performances_harmless = []
        basis_performances = []
        basis_performances_harmless = []
        for dim in range(min_dim, max_dim + 1):
            # Get basis data
            basis = model.get(f"dim_{dim}_basis", [])
            basis_harmless = [1 - x for x in model.get(f"dim_{dim}_basis_harmless", [])]
            basis_performances.append(basis)
            basis_performances_harmless.append(basis_harmless)
            
            # Get sample data if available
            samples = model.get(f"dim_{dim}_samples", [])
            samples_harmless = [1 - x for x in model.get(f"dim_{dim}_samples_harmless", [])]
            performances.append(samples)
            performances_harmless.append(samples_harmless)

        # Only plot boxplots if we have sample data
        if any(len(p) > 0 for p in performances):
            # Plot on first subplot (attack success) - reduce outlier visibility
            bp1 = ax1.boxplot(performances, positions=positions, 
                            widths=0.12,  # Slightly reduced width
                            patch_artist=True,
                            medianprops={'color': 'black', 'linewidth': 1},
                            flierprops={'marker': '.', 
                                      'markerfacecolor': colors[model_idx], 
                                      'markersize': 4,
                                      'alpha': 0.3,
                                      'markeredgecolor': 'none'},
                            whiskerprops={'linewidth': 1},
                            boxprops={'facecolor': colors[model_idx], 
                                    'alpha': 0.8,
                                    'linewidth': 1})

            # Plot on second subplot (harmless scores)
            bp2 = ax2.boxplot(performances_harmless, positions=positions, 
                            widths=0.12,
                            patch_artist=True,
                            medianprops={'color': 'black', 'linewidth': 1},
                            flierprops={'marker': '.', 
                                      'markerfacecolor': colors[model_idx], 
                                      'markersize': 4,
                                      'alpha': 0.3,
                                      'markeredgecolor': 'none'},
                            whiskerprops={'linewidth': 1},
                            boxprops={'facecolor': colors[model_idx], 
                                    'alpha': 0.8,
                                    'linewidth': 1})

        # Plot basis vectors on both subplots
        basis_scatter1 = None
        basis_scatter2 = None
        for i, (basis, basis_harmless) in enumerate(zip(basis_performances, basis_performances_harmless)):
            if not basis:  # Skip if no basis data for this dimension
                continue
            scatter1 = ax1.scatter([positions[i]] * len(basis), basis, 
                       color=colors[model_idx],
                       marker='o', 
                       s=10,  
                       zorder=2,
                       alpha=1.0,
                       edgecolors='black',
                       linewidth=0.5)
            scatter2 = ax2.scatter([positions[i]] * len(basis_harmless), basis_harmless,
                       color=colors[model_idx],
                       marker='o', 
                       s=10,  
                       zorder=2,
                       alpha=1.0,
                       edgecolors='black',
                       linewidth=0.5)
            if i == 0:
                basis_scatter1 = scatter1
                basis_scatter2 = scatter2

        # Collect handles and labels for models
        if any(len(p) > 0 for p in performances):
            if model_idx == 0:
                handles = [bp1['boxes'][0]]
                labels = [f'{model["model_name"]}']
            else:
                handles.append(bp1['boxes'][0])
                labels.append(f'{model["model_name"]}')

    # Create a color-neutral basis vector marker for the legend
    basis_legend_marker = ax1.scatter([], [], 
                                    color='gray',
                                    marker='o',
                                    s=10,
                                    zorder=2,
                                    alpha=1.0,
                                    edgecolors='black',
                                    linewidth=0.5)

    # Add basis vectors to legend last
    if 'handles' in locals():
        handles.append(basis_legend_marker)
        labels.append('Basis Vectors')

    # Customize both plots
    for ax, fig in [(ax1, fig1), (ax2, fig2)]:
        ax.set_xlabel("Cone Dimension")
        # Adjust x-axis ticks for wider spacing
        if "repind" in base_group_name:
            ax.set_xticks([x * group_spacing for x in range(len(subspace_dims))])  # Only subspace dims for repind
            ax.set_xticklabels(subspace_dims)  # Only subspace dims for repind
        else:
            ax.set_xticks([-group_spacing] + [x * group_spacing for x in range(len(subspace_dims))])  # Add dim 1 tick
            ax.set_xticklabels([1] + subspace_dims)  # Add dim 1 label
        ax.tick_params(axis='both', which='major')
        ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', width=0.5, length=3)
        ax.set_ylim(0, 1)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # Add light vertical gridlines to separate dimensions
        ax.grid(axis='x', linestyle='-', alpha=0.1, color='gray', linewidth=0.5)
        
        # Place legend on the right
        if 'handles' in locals():
            legend = ax.legend(handles=handles, labels=labels,
                          frameon=True, fancybox=False, edgecolor='black',
                          loc='center left', bbox_to_anchor=(1.02, 0.5),
                          borderaxespad=0, borderpad=0.5)

        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

    # Set specific y-labels
    ax1.set_ylabel("Attack Success Rate")
    ax2.set_ylabel("Refusal Score")
    
    # Show plots
    plt.figure(fig1.number)
    plt.show()
    plt.figure(fig2.number) 
    plt.show()
    
    # Save plots with extra space for legend
    if "repind" in base_group_name:
        os.makedirs(os.path.dirname(f"results/plots/subspace_performance/{family.lower()}_repind"), exist_ok=True)
        fig1.savefig(f"results/plots/subspace_performance/{family.lower()}_repind_dim_attack.png", dpi=300, bbox_inches='tight')
        fig2.savefig(f"results/plots/subspace_performance/{family.lower()}_repind_dim_harmless.png", dpi=300, bbox_inches='tight')
    else:
        os.makedirs(os.path.dirname(f"results/plots/subspace_performance/{family.lower()}"), exist_ok=True)
        fig1.savefig(f"results/plots/subspace_performance/{family.lower()}_attack.png", dpi=300, bbox_inches='tight')
        fig2.savefig(f"results/plots/subspace_performance/{family.lower()}_harmless.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    plt.close(fig1)

# %%
# Plot comparison between Gemma 9B, Qwen 14B and Llama 8B
selected_models = [
    model for model in models 
    if model["model_name"] in ["Gemma 2 9B", "Qwen 2.5 7B", "Llama 3 8B"]
]

max_dim = 5

min_dim = 2

subspace_dims = list(range(min_dim, max_dim + 1))

# Create attack success plot
figsize = (10, 5)
fig1 = plt.figure(figsize=figsize)
ax1 = fig1.add_subplot(111)

# Create harmless scores plot
fig2 = plt.figure(figsize=figsize)
ax2 = fig2.add_subplot(111)

# Spacing parameters
group_spacing = 1
box_spacing = 0.2

# Plot each model
for model_idx, model in enumerate(selected_models):
    # Calculate offset for this model's boxes
    offset = model_idx * box_spacing - (len(selected_models)-1) * box_spacing/2
    positions = [-group_spacing + offset] + [x * group_spacing + offset for x in range(len(subspace_dims))]  # Add dim 1 position

    # Extract performances for both plots
    performances = []
    performances_harmless = []
    basis_performances = []
    basis_performances_harmless = []
    
    # Add dim 1 basis data
    basis_performances.append(model.get("dim_1_basis", []))
    basis_performances_harmless.append([1 - x for x in model.get("dim_1_basis_harmless", [])])
    
    for dim in range(min_dim, max_dim + 1):
        # Get basis data
        basis = model.get(f"dim_{dim}_basis", [])
        basis_harmless = [1 - x for x in model.get(f"dim_{dim}_basis_harmless", [])]
        basis_performances.append(basis)
        basis_performances_harmless.append(basis_harmless)
        
        # Get sample data if available
        samples = model.get(f"dim_{dim}_samples", [])
        samples_harmless = [1 - x for x in model.get(f"dim_{dim}_samples_harmless", [])]
        performances.append(samples)
        performances_harmless.append(samples_harmless)

    # Only plot boxplots if we have sample data
    if any(len(p) > 0 for p in performances):
        # Plot on first subplot (attack success)
        bp1 = ax1.boxplot(performances, positions=positions[1:], 
                        widths=0.12,
                        patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 1},
                        flierprops={'marker': '.', 
                                  'markerfacecolor': colors[model_idx], 
                                  'markersize': 4,
                                  'alpha': 0.3,
                                  'markeredgecolor': 'none'},
                        whiskerprops={'linewidth': 1},
                        boxprops={'facecolor': colors[model_idx], 
                                'alpha': 0.8,
                                'linewidth': 1})

        # Plot on second subplot (harmless scores)
        bp2 = ax2.boxplot(performances_harmless, positions=positions[1:], 
                        widths=0.12,
                        patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 1},
                        flierprops={'marker': '.', 
                                  'markerfacecolor': colors[model_idx], 
                                  'markersize': 4,
                                  'alpha': 0.3,
                                  'markeredgecolor': 'none'},
                        whiskerprops={'linewidth': 1},
                        boxprops={'facecolor': colors[model_idx], 
                                'alpha': 0.8,
                                'linewidth': 1})

    # Plot basis vectors on both subplots
    basis_scatter1 = None
    basis_scatter2 = None
    for i, (basis, basis_harmless) in enumerate(zip(basis_performances, basis_performances_harmless)):
        if not basis:
            continue
        scatter1 = ax1.scatter([positions[i]] * len(basis), basis, 
                   color='gray',
                   marker='o', 
                   s=10,  
                   zorder=2,
                   alpha=1.0,
                   edgecolors='black',
                   linewidth=0.5)
        scatter2 = ax2.scatter([positions[i]] * len(basis_harmless), basis_harmless,
                   color='gray',
                   marker='o', 
                   s=10,  
                   edgecolors='black',
                   linewidth=0.5,
                   zorder=2,
                   alpha=1.0)
        if i == 0:
            basis_scatter1 = scatter1
            basis_scatter2 = scatter2

    # Collect handles and labels for models
    if any(len(p) > 0 for p in performances):
        if model_idx == 0:
            handles = [bp1['boxes'][0]]
            labels = [f'{model["model_name"]}']
        else:
            handles.append(bp1['boxes'][0])
            labels.append(f'{model["model_name"]}')
        
        if model_idx == 0:
            basis_legend_scatter = basis_scatter1

# Add basis vectors to legend last
if 'handles' in locals():
    handles.append(basis_legend_scatter)
    labels.append('Basis Vectors')

# Customize both plots
for ax, fig in [(ax1, fig1), (ax2, fig2)]:
    ax.set_xlabel("Cone Dimension")
    ax.set_xticks([-group_spacing] + [x * group_spacing for x in range(len(subspace_dims))])  # Add dim 1 tick
    ax.set_xticklabels([1] + subspace_dims)  # Add dim 1 label
    ax.tick_params(axis='both', which='major')
    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(axis='both', width=0.5, length=3)
    ax.set_ylim(0, 1)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Add light vertical gridlines
    ax.grid(axis='x', linestyle='-', alpha=0.1, color='gray', linewidth=0.5)
    
    # Adjust legend position and layout
    if 'handles' in locals():
        legend = ax.legend(handles=handles, labels=labels,
                      frameon=True, fancybox=False, edgecolor='black',
                      loc='center left', bbox_to_anchor=(1.02, 0.5),
                      borderaxespad=0, borderpad=0.5)

    plt.tight_layout()

# Set specific y-labels
ax1.set_ylabel("Attack Success Rate")
ax2.set_ylabel("Refusal Score")

# Save plots
plt.figure(fig1.number)
plt.tight_layout()
plt.show()
plt.figure(fig2.number)
plt.tight_layout()
plt.show()
os.makedirs("results/plots/subspace_performance/model_comparison", exist_ok=True)
fig1.savefig("results/plots/subspace_performance/model_comparison_attack.png", dpi=300, bbox_inches='tight')
fig2.savefig("results/plots/subspace_performance/model_comparison_harmless.png", dpi=300, bbox_inches='tight')

plt.close(fig2)
plt.close(fig1)

# %%
# Plot ASR distribution across quantiles for multiple models
fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# Define colors for different dimensions
dim_colors = sns.color_palette('viridis', 6)

# Create shared handles and labels for legend
handles = []
labels = []

# Plot for model_idx 0 (Qwen 14B)
model_idx = 0
model_name = models[model_idx]["model_name"]
ax = axes[0]

# Add baseline for dimension 1
if "dim_1_basis" in models[model_idx]:
    baseline = models[model_idx]["dim_1_basis"][0]
    baseline_line = ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, 
               label=f"Dimension 1 (RDO)")
    # Add baseline to legend handles and labels
    handles.append(baseline_line)
    labels.append("Dimension 1 (RDO)")

# Loop through dimensions 2-7
for i, dim in enumerate(range(2, 8)):
    key = f"dim_{dim}_samples"
    if key not in models[model_idx] or not models[model_idx][key]:
        continue
        
    # Get samples data
    samples = np.array(models[model_idx][key])
    if len(samples) == 0:
        continue
    
    # Sort samples to compute all quantiles (in descending order for highest values first)
    sorted_samples = np.sort(samples)[::-1]  # Reversed to have highest values first
    # Create quantile values (0 to 1)
    quantile_values = np.linspace(0, 1, len(sorted_samples))
    # Convert to percentages (0 to 100)
    percentiles = 100 * quantile_values
    
    # Plot the line for this dimension
    line, = ax.plot(
        percentiles,
        sorted_samples,
        '-', 
        linewidth=2,
        color=dim_colors[i],
        label=f"Dimension {dim}"
    )
    
    # Only add to handles/labels once
    if model_idx == 0:
        handles.append(line)
        labels.append(f"Dimension {dim}")

# Customize plot
ax.set_ylabel("Attack Success Rate by Percentile")
ax.set_title(f"{model_name}")
ax.set_ylim(0, 1)
ax.grid(axis='both', linestyle='-', alpha=0.15)

# Remove top and right spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Plot for model_idx 3
model_idx = 3
model_name = models[model_idx]["model_name"]
ax = axes[1]

# Add baseline for dimension 1
if "dim_1_basis" in models[model_idx]:
    baseline = models[model_idx]["dim_1_basis"][0]
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, 
               label=f"Dimension 1 (RDO)")

# Loop through dimensions 2-7
for i, dim in enumerate(range(2, 8)):
    key = f"dim_{dim}_samples"
    if key not in models[model_idx] or not models[model_idx][key]:
        continue
        
    # Get samples data
    samples = np.array(models[model_idx][key])
    if len(samples) == 0:
        continue
    
    # Sort samples to compute all quantiles (in descending order for highest values first)
    sorted_samples = np.sort(samples)[::-1]  # Reversed to have highest values first
    # Create quantile values (0 to 1)
    quantile_values = np.linspace(0, 1, len(sorted_samples))
    # Convert to percentages (0 to 100)
    percentiles = 100 * quantile_values
    
    # Plot the line for this dimension
    ax.plot(
        percentiles,
        sorted_samples,
        '-', 
        linewidth=2,
        color=dim_colors[i],
        label=f"Dimension {dim}"
    )

# Customize plot
ax.set_xlabel("Percentile")
ax.set_ylabel("Attack Success Rate by Percentile")
ax.set_title(f"{model_name}")
# Set x-ticks at key percentiles
ax.set_xticks([0, 10, 25, 50, 75, 90, 100])
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.grid(axis='both', linestyle='-', alpha=0.15)

# Remove top and right spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Add shared legend at the top of the figure
fig.legend(handles=handles, labels=labels,
           frameon=True, fancybox=True, edgecolor='black',
           loc='upper right', bbox_to_anchor=(1.0, 0.95))

plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Make room for the legend

# Save the plot
os.makedirs("results/plots/subspace_performance/quantiles", exist_ok=True)
plt.savefig(f"results/plots/subspace_performance/quantiles/model_comparison_asr_quantiles.png", 
            dpi=300, bbox_inches='tight')
plt.show()


            
#%%
np.array(models[0]["dim_3_samples"]).shape
# %%
# Set style and figure size
def plot_performance_vs_samples(models, model_idx, model_name, path):
    """
    Plot performance vs number of samples for different dimensions,
    using professional font sizes and styling.
    """
    import os
    import json
    import glob
    import numpy as np
    import matplotlib.pyplot as plt

    # Define markers for each dimension (ensure at least as many markers as dimensions)
    markers = ['o', 'x', 's', 'D', '^']

    # Check if the model has any samples for dimensions 2 through 6
    if not any(f"dim_{dim}_samples" in models[model_idx] for dim in range(2, 7)):
        print(f"Warning: No sample data found for {model_name}")
        return

    # Use a smaller figure size as specified
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Loop over dimensions 2 to 6 (or skip if not available)
    for i, dim in enumerate(range(2, 7)):
        if f"dim_{dim}_samples" not in models[model_idx]:
            continue
        # Assemble the path and initialize list for sample performance data
        model_path = f"/ceph/hdd/students/elsj/paper_results/subspace_samples_eval/join_subspace_gemma-2-2b-it/dim_{dim}"
        sample_performances = []
        for sample_num in range(1, 257):
            # Look for the attack sample file
            attack_file = os.path.join(model_path, "completions", f"jailbreakbench_ablation_sample_{sample_num}_evaluations.json")
            attack_files = glob.glob(attack_file)
            if attack_files:
                with open(attack_files[0]) as f:
                    data = json.load(f)
                # Get the list of scores for this sample
                scores = [d["is_jailbreak_strongreject"] for d in data["completions"]]
                sample_performances.append(scores)

        # Skip plotting if no sample performances were retrieved
        if not sample_performances:
            continue
        sample_performances = np.array(sample_performances)
        # Compute maximum performance for increasing number of samples starting at 4
        max_perf_for_n_samples = [
            sample_performances[:j].max(axis=0).mean() 
            for j in range(4, len(sample_performances) + 1)
        ]
        # Plot using a semilogx line with the assigned marker and color
        ax.semilogx(
            range(4, len(max_perf_for_n_samples) + 4),
            max_perf_for_n_samples,
            marker=markers[i % len(markers)],
            markersize=4,
            linewidth=1.5,
            label=str(dim),
            color=colors[i],
            markevery=30,
            alpha=0.9
        )

    # If no data was plotted, exit after warning
    if not ax.lines:
        print(f"Warning: No valid dimensions to plot for {model_name}")
        plt.close(fig)
        return

    # Customize axes labels using the specified font sizes
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Performance')

    # Set a dashed grid with slight transparency
    ax.grid(True, linestyle='--', alpha=0.4, which='both')

    # Add a customized legend
    ax.legend(
        title='Cone Dimension',
        frameon=True,
        fancybox=False,
        edgecolor='black',
        bbox_to_anchor=(0.97, 0.45),
        loc='best'
    )

    # Set the x-axis scaling and ticks
    ax.set_xscale('log')
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels(['4', '8', '16', '32', '64', '128', '256'])
    ax.xaxis.set_minor_locator(plt.NullLocator())  # Remove minor ticks

    # Set y-axis limits from the data (with a slight margin)
    ydata = [line.get_ydata() for line in ax.lines if len(line.get_ydata()) > 0]
    if ydata:
        ydata_flat = np.concatenate(ydata)
        ax.set_ylim(ydata_flat.min() - 0.01, ydata_flat.max() + 0.01)

    # Set tick parameters for a cleaner look
    ax.tick_params(axis='both', which='major')

    # Clean up spines for a professional appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Finalize layout and save the figure
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

plot_performance_vs_samples(models, 0, "Gemma 2 2B", "results/plots/subspace_performance/samples/max_asrs.png")
# plot_performance_vs_samples(models, 1, "Qwen 2.5 1.5B", "results/plots/subspace_performance/samples/qwen_2_5_1_5b.png")
# plot_performance_vs_samples(models, 2, "Qwen 2.5 3B", "results/plots/subspace_performance/samples/qwen_2_5_3b.png")
# plot_performance_vs_samples(models, 3, "Qwen 2.5 7B", "results/plots/subspace_performance/qwen_2_5_7b_performance_vs_samples.png")

# %%
temperatures = [0.2, 0.5, 1, 1.2]
sample_performances_by_temp = {temp: [] for temp in temperatures}
path = "/ceph/hdd/students/elsj/paper_results/temperature_completions/temperature_completions/completions"
for file in os.listdir(path):
    for temperature in temperatures:
        if file.endswith("evaluations.json") and f"temp{temperature}" in file:
            with open(os.path.join(path, file), "r") as f:
                data = json.load(f)
            scores = [d["is_jailbreak_strongreject"] for d in data["completions"]]
            sample_performances_by_temp[temperature].append(scores)

# %%
print(sample_performances_by_temp)
# %%
import glob
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

# Plot performance by temperature and compare with Gemma samples
# plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# Store original font sizes
original_sizes = {k: v for k, v in plt.rcParams.items() if 'size' in k.lower()}

# Plot Gemma sample performance with increased visibility
gemma_performances = np.array(gemma_sample_performances)
num_batches = len(gemma_performances) // 64
batch_performances = []

# Compute max performance for each batch
gemma_performance_batches = [gemma_performances[i:i+64] for i in range(0, len(gemma_performances), 64)]
for batch in gemma_performance_batches:
    batch_max_perf = [batch[:i].max(axis=0).mean() for i in range(4, min(64, len(batch))+1)]
    batch_performances.append(batch_max_perf)

# Average across batches and compute std
batch_performances = np.array(batch_performances)
max_perf_for_n_samples = np.mean(batch_performances, axis=0)
max_perf_std = np.std(batch_performances, axis=0)

# Plot temperature sampling results first
for i, (temp, performances) in enumerate(sample_performances_by_temp.items()):
    performances = np.array(performances)
    max_perf_for_n_samples_temp = [performances[:i].max(axis=0).mean() for i in range(4, min(64, len(performances))+1)]
    
    plt.semilogx(range(4, len(max_perf_for_n_samples_temp) + 4), max_perf_for_n_samples_temp,
                linewidth=2,
                color=colors[i+1],
                alpha=0.8,
                label=f'RDO with T={temp}')

# Plot mean performance with dashed line for cone
plt.semilogx(range(4, len(max_perf_for_n_samples) + 4), max_perf_for_n_samples,
            linewidth=2.5,
            color=colors[0],
            alpha=1.0,  # Increased opacity
            label='Cone Sampling',
            zorder=10,  # Ensure it's drawn on top
            linestyle='--')  # Make line dashed

# Plot standard deviation as shaded region
x_values = range(4, len(max_perf_for_n_samples) + 4)
# plt.fill_between(x_values, 
#                  max_perf_for_n_samples - max_perf_std,
#                  max_perf_for_n_samples + max_perf_std,
#                  color=colors[0],
#                  alpha=0.2)

# Customize plot
font_scale = 1.5
plt.xlabel('Number of Samples', fontsize=plt.rcParams['font.size'] * font_scale)
plt.ylabel('Attack Success Rate', fontsize=plt.rcParams['font.size'] * font_scale)

# Grid and legend
plt.grid(True, linestyle='--', alpha=0.6, which='both')
plt.legend(frameon=True,
          fancybox=False,
          edgecolor='black',
          loc='lower right', 
          ncol=1,  # Changed to 1 column
          fontsize=plt.rcParams['font.size'] * font_scale,
          handlelength=1.5) # Reduce width of color indicators

# Axes settings
plt.gca().set_xscale('log')
plt.gca().set_xticks([4, 8, 16, 32, 64])
plt.gca().set_xticklabels(['4', '8', '16', '32', '64'], fontsize=plt.rcParams['font.size'] * font_scale)
plt.gca().xaxis.set_minor_locator(plt.NullLocator())
plt.ylim(min(min(line.get_ydata()) for line in plt.gca().get_lines()) - 0.01, 
         max(max(line.get_ydata()) for line in plt.gca().get_lines()) + 0.017)
plt.tick_params(axis='both', which='major', labelsize=plt.rcParams['font.size'] * font_scale)

# Clean up spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.5)
plt.gca().spines['bottom'].set_linewidth(0.5)

# Save plot
plt.tight_layout()

# Save plot with consistent parameters
os.makedirs("results/plots/temperature_sampling", exist_ok=True)
plt.savefig("results/plots/temperature_sampling/temperature_vs_subspace.png", 
            dpi=300, 
            bbox_inches='tight', 
            facecolor='white')
plt.show()

# Restore original font sizes
plt.rcParams.update(original_sizes)

# %%
# heuristic_scores = torch.load("results/refusal_scores/gemma-2-2b-it/refusal_scores.pt")
# heuristic_scores = -np.array(gemma_sample_performances) # for debugging
# # Plot performance using refusal score heuristic with varying number of samples
# plt.style.use('seaborn-v0_8-paper')
# plt.figure(figsize=(10, 6))

# # Define colors
# colors = ['#59a14f', '#4e79a7']  # Green and blue

# # Use refusal scores as heuristic for Gemma samples
# gemma_performances = np.array(gemma_sample_performances)  # Shape: [256, num_instructions]
# refusal_scores = heuristic_scores

# # Calculate performance for different numbers of samples
# n_samples = range(4, 64)  # Start from 4 samples up to 256

# # Calculate performance using heuristic
# heuristic_performances = []
# for n in n_samples:
#     # For each instruction, select from top n samples with lowest refusal scores
#     top_n_indices = np.argpartition(refusal_scores[:n], 0, axis=0)[0]  # Get best sample index from first n samples
#     selected_performances = gemma_performances[top_n_indices, np.arange(len(top_n_indices))]
#     mean_perf = np.mean(selected_performances)
#     heuristic_performances.append(mean_perf)

# # Calculate performance using best sample directly
# direct_performances = []
# for n in n_samples:
#     # For each instruction, take the best performing sample from first n samples
#     best_performances = np.max(gemma_performances[:n], axis=0)
#     mean_perf = np.mean(best_performances)
#     direct_performances.append(mean_perf)

# # Plot both approaches
# plt.semilogx(n_samples, heuristic_performances,
#             marker='o', markersize=4, linewidth=2,
#             color=colors[0],
#             markevery=30,  # Show fewer markers
#             alpha=1.0,
#             label='Using refusal score heuristic',
#             zorder=2)

# plt.semilogx(n_samples, direct_performances,
#             marker='s', markersize=4, linewidth=2,
#             color=colors[1], 
#             markevery=30,
#             alpha=1.0,
#             label='Using best performing sample',
#             zorder=2)

# # Customize plot
# plt.xlabel('Number of Samples', fontsize=11)
# plt.ylabel('Attack Success Rate', fontsize=11)

# # Grid and legend
# plt.grid(True, linestyle='--', alpha=0.4, which='both')
# plt.legend(frameon=True,
#           fancybox=False,
#           edgecolor='black', 
#           fontsize=10,
#           loc='lower right')

# # Axes settings
# plt.gca().set_xscale('log')
# plt.gca().set_xticks([4, 8, 16, 32, 64])
# plt.gca().set_xticklabels(['4', '8', '16', '32', '64'])
# plt.gca().xaxis.set_minor_locator(plt.NullLocator())
# plt.ylim(0.84, 0.92)
# plt.tick_params(axis='both', which='major', labelsize=10)

# # Clean up spines
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_linewidth(0.5)
# plt.gca().spines['bottom'].set_linewidth(0.5)

# # Save plot
# plt.tight_layout()
# plt.show()
# os.makedirs("results/plots/heuristic_sampling", exist_ok=True)
# plt.savefig("results/plots/heuristic_sampling/refusal_score_heuristic_by_samples.png", 
#             dpi=300, bbox_inches='tight', facecolor='white')
# plt.close()

# %%
# Plot samples for Qwen 14B for dimensions 2 and 3
qwen_14b = dict(model_name="Qwen 2.5 14B",
         model_id="Qwen/Qwen2.5-14B-Instruct")
base_dir = "/ceph/hdd/students/elsj/paper_results/subspace_samples_eval"
api = wandb.Api()
model_id = qwen_14b["model_id"].split("/")[-1]
base_group_name = "join_subspace"
samples_folder = os.path.join(base_dir, f"{base_group_name}_{model_id}")
run_names = os.listdir(samples_folder)
run_paths = [os.path.join(samples_folder, run_name) for run_name in run_names]
for run_path in run_paths:
    dim = int(run_path.split("_")[-1])
    if dim not in [2, 3]:
        continue
    completions_path = os.path.join(run_path, "completions")

    runs = api.runs("refusal-representations/robust_refusal_subspace", {"group": f"{base_group_name}_{model_id}", "display_name": f"dim_{dim}"})
    run = runs[0]
    summary = run.summary._json_dict
    basis_scores = []
    for basis_dim in range(1, dim + 1):
        basis_scores.append(summary[f"jailbreakbench_ablation_basis_{basis_dim}_StrongREJECT_score"])
    qwen_14b[f"dim_{dim}_basis"] = basis_scores
    for sample_idx in range(1, 257):
        file = f"jailbreakbench_ablation_sample_{sample_idx}_evaluations.json"
        with open(os.path.join(completions_path, file), "r") as f:
            data = json.load(f)
        qwen_14b[f"dim_{dim}_samples"] = qwen_14b.get(f"dim_{dim}_samples", []) + [data["StrongREJECT_score"]]

print(qwen_14b)
# %%
def sample_hypersphere_vectors(batch_size, dim):
    print("Sampling hypersphere vectors")
    rng_state = torch.get_rng_state()
    torch.manual_seed(42)
    samples = torch.randn(batch_size, dim).abs()
    samples = samples / torch.norm(samples, dim=1, keepdim=True)
    torch.set_rng_state(rng_state)
    return samples

samples_2d = sample_hypersphere_vectors(512, 2)[:256].numpy()
samples_3d = sample_hypersphere_vectors(512, 3)[:256].numpy()
# %%
# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with subplots
fig = make_subplots(rows=1, cols=2, 
                    specs=[[{'type': 'scatter'}, {'type': 'surface'}]],
                    subplot_titles=("2D Unit Circle (First Quadrant)", "3D Unit Sphere (First Octant)"))

# 2D Plot - Unit Circle First Quadrant
# Create dense grid of points along the arc
theta = np.linspace(0, np.pi/2, 100)
arc_x = np.cos(theta)
arc_y = np.sin(theta)
arc_points = np.column_stack((arc_x, arc_y))

# Interpolate performance values along the arc
arc_performances = griddata(samples_2d, qwen_14b["dim_2_samples"], arc_points, method='cubic')

# Add the arc with color gradient
fig.add_trace(
    go.Scatter(
        x=arc_x, y=arc_y,
        mode='lines',
        line=dict(
            width=6,
        ),
        marker=dict(
            color=arc_performances,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Performance Score")
        ),
        showlegend=False
    ),
    row=1, col=1
)

# Add scatter points
fig.add_trace(
    go.Scatter(
        x=samples_2d[:, 0], y=samples_2d[:, 1],
        mode='markers',
        marker=dict(
            size=6,
            color=qwen_14b["dim_2_samples"],
            colorscale='Viridis',
            showscale=False
        ),
        showlegend=False
    ),
    row=1, col=1
)

# Add basis vectors
fig.add_trace(
    go.Scatter(
        x=[1, 0], y=[0, 1],
        mode='markers+text',
        marker=dict(symbol='diamond', size=15, color='red'),
        text=['b₁', 'b₂'],
        textposition="top right",
        showlegend=False
    ),
    row=1, col=1
)

# 3D Plot - Unit Sphere First Octant
# Create a finer mesh for the sphere surface
phi = np.linspace(0, np.pi/2, 50)  # Only first octant
theta = np.linspace(0, np.pi/2, 50)  # Only first octant
phi, theta = np.meshgrid(phi, theta)

# Spherical to Cartesian coordinates
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Create points for interpolation
points_3d = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
grid_performances_3d = griddata(samples_3d, qwen_14b["dim_3_samples"], points_3d, method='linear')
grid_performances_3d = grid_performances_3d.reshape(x.shape)

# Add surface
fig.add_trace(
    go.Surface(
        x=x, y=y, z=z,
        surfacecolor=grid_performances_3d,
        colorscale='Viridis',
        showscale=False,
        opacity=0.8
    ),
    row=1, col=2
)

# Add scatter points
fig.add_trace(
    go.Scatter3d(
        x=samples_3d[:, 0], y=samples_3d[:, 1], z=samples_3d[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=qwen_14b["dim_3_samples"],
            colorscale='Viridis',
            showscale=False
        ),
        showlegend=False
    ),
    row=1, col=2
)

# Add basis vectors
fig.add_trace(
    go.Scatter3d(
        x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1],
        mode='markers+text',
        marker=dict(symbol='diamond', size=8, color='red'),
        text=['b₁', 'b₂', 'b₃'],
        showlegend=False
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title_text="Performance on Unit Circle/Sphere Surfaces",
    showlegend=False,
    width=1200,
    height=600,
    scene=dict(
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5),  # Adjust camera position
            up=dict(x=0, y=0, z=1)
        ),
        aspectmode='cube',  # Force equal aspect ratio
        xaxis=dict(range=[0, 1], showgrid=True, zeroline=False, showline=False),
        yaxis=dict(range=[0, 1], showgrid=True, zeroline=False, showline=False),
        zaxis=dict(range=[0, 1], showgrid=True, zeroline=False, showline=False),
    ),
    xaxis=dict(range=[0, 1], showgrid=True, zeroline=False, showline=False),
    yaxis=dict(range=[0, 1], showgrid=True, zeroline=False, showline=False),
    xaxis_title="x",
    yaxis_title="y"
)

# Save as PNG (static)
fig.write_image("results/plots/qwen_14b_surface.png", scale=4)

# %%
# run_names = [
#     "rep_ind_1_run_2",
#     "rep_ind_2_run_3", 
#     "rep_ind_3_run_2",
#     "rep_ind_4_run_3",
#     "rep_ind_5_run_1",
# ]
# group = "repind_symmetric_sum_loss_gemma-2-2b-it"
run_names = [
    "rep_ind_1_run_1",
    "rep_ind_2_run_3", 
    "rep_ind_3_run_1",
    "rep_ind_4_run_1",
    "rep_ind_5_run_1",
]
group = "repind_symmetric_sum_loss_diff_cutoff09_gemma-2-2b-it"
api = wandb.Api()
runs = []
matching_runs = api.runs("refusal-representations/robust_refusal_vector", 
                        filters={"group": group})
print(f"Found {len(matching_runs)} runs of the group")

# %%
eval_results = []
for run_name in run_names:
    for run in matching_runs:
        if run_name == run.name:
            print(run_name)
            # Download file from the run's files
            file_path = "completions/jailbreakbench_ablation_evaluations.json"
            download_dir = os.path.join("wandb_downloads", run.group, run.name)
            os.makedirs(download_dir, exist_ok=True)
            run.file(file_path).download(root=download_dir, exist_ok=True)
            file_path = os.path.join(download_dir, file_path)
            eval_results.append(json.load(open(file_path)))
# %%
eval_results.insert(0, json.load(open("results/refusal_dir/gemma-2-2b-it/completions/jailbreakbench_ablation_evaluations.json")))

# %%
# Calculate mean scores for each result
mean_scores = [np.mean([c["is_jailbreak_strongreject"] for c in result["completions"]]) for result in eval_results]

# Create labels and sort by performance
labels = []
if len(eval_results) == 6:  # If index 0 is included
    dim_score = mean_scores[0]
    dim_label = "DIM"
    # Remove DIM direction from lists to sort RepInd directions
    mean_scores = mean_scores[1:]
    labels = [f"RepInd {i}" for i in range(1, 6)]
else:
    labels = [f"RepInd {i}" for i in range(1, 6)]

# Sort RepInd scores and labels together
# sorted_pairs = sorted(zip(mean_scores, labels), reverse=True)
# mean_scores, labels = zip(*sorted_pairs)
mean_scores = sorted(mean_scores, reverse=True)

# Add DIM direction back if it exists
if len(eval_results) == 6:
    mean_scores = [dim_score] + list(mean_scores)
    labels = [dim_label] + list(labels)

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 4))

# Define professional colors - blue for DIM direction, orange for RepInd
plot_colors = [colors[0]] + [colors[1]]*(len(mean_scores)-1)

# Create bars with consistent styling but reduced spacing
bars = ax.bar(np.arange(len(mean_scores))*0.7, mean_scores, 
              color=plot_colors, 
              alpha=0.8, 
              width=0.5,
              edgecolor='black', 
              linewidth=0.5)

# Customize plot with consistent font sizes
ax.set_ylabel('Attack Success Rate')
ax.set_xticks(np.arange(len(mean_scores))*0.7)
ax.set_xticklabels(labels, rotation=45, ha='center')
ax.tick_params(axis='both', which='major')

# Grid styling with consistent parameters
ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray', linewidth=0.5)

# Adjust spines to be thinner
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

# Make tick marks thinner and shorter
ax.tick_params(axis='both', width=0.5, length=3)

# Y-axis limits and spines
ax.set_ylim(0, 1)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Add value labels on top of bars with consistent font size
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', 
            va='bottom')

# Adjust layout with consistent parameters
plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.3, w_pad=0.3)

# Save plot with consistent parameters
os.makedirs("results/plots/repind", exist_ok=True)
plt.savefig('results/plots/repind/performance.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white')
plt.show()


# %%
models = [
    dict(model_name="Gemma 2 2B", 
         model_id="google/gemma-2-2b-it"),
    dict(model_name="Gemma 2 9B",
         model_id="google/gemma-2-9b-it"),
    dict(model_name="Llama 3 8B",
         model_id="meta-llama/Meta-Llama-3-8B-Instruct"),
    dict(model_name="Qwen 2.5 1.5B",
         model_id="Qwen/Qwen2.5-1.5B-Instruct"),
    dict(model_name="Qwen 2.5 3B",
         model_id="Qwen/Qwen2.5-3B-Instruct"),
    dict(model_name="Qwen 2.5 7B",
         model_id="Qwen/Qwen2.5-7B-Instruct"),
    dict(model_name="Qwen 2.5 14B",
         model_id="Qwen/Qwen2.5-14B-Instruct"),
]

benchmarks_path = 'lm-evaluation-harness/chat_results'
folder_names = ["original", "baseline", "wandb"]
benchmarks = ["truthfulqa_mc2", "arc_challenge", "gsm8k", "mmlu"]
for model in models:
    family, model_id = model["model_id"].split("/")
    for folder_name in folder_names:
        model_path = os.path.join(benchmarks_path, folder_name, f"{family}__{model_id}")
        if os.path.exists(model_path):
            # get the file that starts with results_ and load the json
            for file in os.listdir(model_path)[::-1]:
                if file.startswith("results_"):
                    with open(os.path.join(model_path, file), "r") as f:
                        data = json.load(f)
                    
                    # Extract all benchmark scores
                    results = data["results"]

                    if "arc_challenge" not in results:
                        continue
                    
                    # TruthfulQA
                    if "truthfulqa_mc2" in results:
                        model[f"{folder_name}_truthfulqa_mc2_acc"] = results["truthfulqa_mc2"]["acc,none"]
                    
                    # ARC Challenge
                    if "arc_challenge" in results:
                        model[f"{folder_name}_arc_challenge_acc"] = results["arc_challenge"]["acc,none"]
                    
                    # GSM8K (using flexible-extract for better accuracy)
                    if "gsm8k" in results:
                        model[f"{folder_name}_gsm8k_acc"] = results["gsm8k"]["exact_match,flexible-extract"]
                    
                    # MMLU
                    if "mmlu" in results:
                        model[f"{folder_name}_mmlu_acc"] = results["mmlu"]["acc,none"]
                    
                    avg_acc = (model[f"{folder_name}_truthfulqa_mc2_acc"] + \
                        model[f"{folder_name}_gsm8k_acc"] + \
                        model[f"{folder_name}_arc_challenge_acc"] + \
                        model[f"{folder_name}_mmlu_acc"]) / 4
                    model[f"{folder_name}_avg_acc"] = avg_acc
                    
                    break
        else:
            print(f"Folder {folder_name} does not exist for model {model_id}")
            continue
# %%
from tabulate import tabulate

def format_value(value):
    return f"{value*100:.1f}%" if isinstance(value, (int, float)) else str(value)

# Create a table for each benchmark
benchmarks = ["truthfulqa_mc2", "arc_challenge", "gsm8k", "mmlu", "avg"]
headers = ["Model", "DIM", "RDO", "Baseline"]

# Create directory for saving tables
os.makedirs("results/tables", exist_ok=True)

# Open a single file to save all tables
with open("results/tables/benchmark_results.txt", "w") as f:
    for benchmark in benchmarks:
        print(f"\n{benchmark.upper()} Results:")
        f.write(f"\n{benchmark.upper()} Results:\n")
        
        table_data = []
        
        for model in models:
            row = [model["model_name"]]
            
            # Add values for each mode, leaving cell blank if data doesn't exist
            dim_value = None
            rdo_value = None
            
            for mode in ["original", "wandb", "baseline"]:
                key = f"{mode}_{benchmark}_acc"
                if model.get(key) is not None:
                    if mode == "original":
                        dim_value = model[key]
                        row.append(format_value(dim_value))
                    elif mode == "wandb":
                        rdo_value = model[key]
                        # Include difference between DIM and RDO
                        if dim_value is not None and rdo_value is not None:
                            diff = rdo_value - dim_value
                            sign = "+" if diff >= 0 else "-"
                            row.append(f"{format_value(rdo_value)} ({sign}{abs(diff)*100:.1f}%)")
                        else:
                            row.append(format_value(rdo_value))
                    else:
                        row.append(format_value(model[key]))
                else:
                    row.append("")
            
            # Only add row if at least one value exists
            if any(cell != "" for cell in row[1:]):
                table_data.append(row)
        
        # Create and print table for this benchmark
        if table_data:
            table = tabulate(table_data, headers=headers, tablefmt="grid")
            print(table)
            f.write(table + "\n\n")
        else:
            print(f"No data available for {benchmark}")
            f.write(f"No data available for {benchmark}\n\n")

# %%
with open("results/tables/benchmark_diffs.txt", "w") as f:
    for benchmark in benchmarks:
        print(f"\n{benchmark.upper()} Results:")
        f.write(f"\n{benchmark.upper()} Results:\n")
        
        table_data = []
        headers = ["Model", "DIM", "RDO"]
        
        for model in models:
            row = [model["model_name"]]
            
            # Get baseline value for comparison
            baseline_key = f"baseline_{benchmark}_acc"
            baseline_value = model.get(baseline_key)
            
            # Calculate differences from baseline for DIM and RDO
            dim_key = f"original_{benchmark}_acc"
            rdo_key = f"wandb_{benchmark}_acc"
            
            # Add DIM difference
            dim_diff = None
            if model.get(dim_key) is not None and baseline_value is not None:
                dim_diff = model[dim_key] - baseline_value
                row.append(f"{abs(dim_diff)*100:.1f}%")
            else:
                row.append("")
                
            # Add RDO difference
            if model.get(rdo_key) is not None and baseline_value is not None:
                rdo_diff = model[rdo_key] - baseline_value
                rdo_dim_diff = ""
                # Add difference between RDO and DIM
                if model.get(dim_key) is not None and dim_diff is not None:
                    diff = abs(rdo_diff) - abs(dim_diff)
                    sign = "+" if diff >= 0 else "-"
                    rdo_dim_diff = f" ({sign}{abs(diff)*100:.1f}%)"
                row.append(f"{abs(rdo_diff)*100:.1f}%{rdo_dim_diff}")
            else:
                row.append("")
            
            # Only add row if at least one value exists
            if any(cell != "" for cell in row[1:]):
                table_data.append(row)
        
        # Create and print table for this benchmark
        if table_data:
            table = tabulate(table_data, headers=headers, tablefmt="grid")
            print(table)
            f.write(table + "\n\n")
        else:
            print(f"No data available for {benchmark}")
            f.write(f"No data available for {benchmark}\n\n")


# %%
models = [
    dict(model_name="Gemma 2 2B", 
         model_id="google/gemma-2-2b-it"),
    dict(model_name="Gemma 2 9B",
         model_id="google/gemma-2-9b-it"),
    dict(model_name="Llama 3 8B",
         model_id="meta-llama/Meta-Llama-3-8B-Instruct"),
    dict(model_name="Qwen 2.5 1.5B",
         model_id="Qwen/Qwen2.5-1.5B-Instruct"),
    dict(model_name="Qwen 2.5 3B",
         model_id="Qwen/Qwen2.5-3B-Instruct"),
    dict(model_name="Qwen 2.5 7B",
         model_id="Qwen/Qwen2.5-7B-Instruct"),
    dict(model_name="Qwen 2.5 14B",
         model_id="Qwen/Qwen2.5-14B-Instruct"),
]

# load baseline scores
dim_dir = os.path.join("/ceph/hdd/students/elsj/paper_results/dim_directions")
for model in models:
    print(model)
    model_id = model["model_id"].split("/")[-1]
    datasets = ["jailbreakbench", "strongreject", "sorrybench", "xstest"]
    for dataset in datasets:
        key = 'StrongREJECT_score' if dataset != "xstest" else "xstest_judgements"
        path = os.path.join(dim_dir, f"{model_id}/completions/{dataset}_ablation_evaluations.json")
        with open(path, "r") as f:
            data = json.load(f)
        model[f"{dataset}_ablation_asr"] = data[key]
        path = os.path.join(dim_dir, f"{model_id}/completions/{dataset}_baseline_evaluations.json")
        with open(path, "r") as f:
            data = json.load(f)
        model[f"{dataset}_baseline_asr"] = data[key]
        path = os.path.join(dim_dir, f"{model_id}/completions/{dataset}_actadd_evaluations.json")
        with open(path, "r") as f:
            data = json.load(f)
        model[f"{dataset}_actadd_asr"] = data[key]
    path = os.path.join(dim_dir, f"{model_id}/completions/harmless_actadd_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["harmless_actadd_asr"] = data["substring_matching_success_rate"]
    path = os.path.join(dim_dir, f"{model_id}/completions/harmless_baseline_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["harmless_baseline_asr"] = data["substring_matching_success_rate"]
    path = os.path.join(dim_dir, f"{model_id}/direction.pt")
    refusal_direction = torch.load(path, map_location=torch.device('cpu'))
    model["refusal_direction"] = refusal_direction.clone()

    api = wandb.Api()
    groups = [
        dict(group="sb_data_with_retain", name="retain"),
    ]
    for group in groups:
        group_name = f"{group['group']}_{model_id}"
        run_name = "run_4" if (model_id == "Qwen2.5-3B-Instruct" and "retain" in group_name) else "run_5"
        runs = api.runs("refusal-representations/robust_refusal_vector", {"group": group_name, "display_name": run_name})
        if runs:
            run = runs[0]
            summary = run.summary._json_dict
            key = "retain_summary"
            model[f"{key}"] = summary
# %%
def plot_scores(y_label: str, dataset: str):
    # Extract scores from models
    dataset_key = dataset.lower().replace("-", "")
    ablation_scores = [model[f"{dataset_key}_ablation_asr"] for model in models]
    actadd_scores = [model[f"{dataset_key}_actadd_asr"] for model in models]
    baseline_scores = [model[f"{dataset_key}_baseline_asr"] for model in models]
    if 'xstest' in dataset_key: 
        retain_ablation_scores = [model[f"retain_summary"][f"{dataset_key}_ablation_xstest_judgements"] for model in models]
        retain_actadd_scores = [model[f"retain_summary"][f"{dataset_key}_actadd_xstest_judgements"] for model in models]
    else:
        retain_ablation_scores = [model[f"retain_summary"][f"{dataset_key}_ablation_StrongREJECT_score"] for model in models]
        retain_actadd_scores = [model[f"retain_summary"][f"{dataset_key}_actadd_StrongREJECT_score"] for model in models]
    model_names = [model["model_name"] for model in models]

    x = np.arange(len(models))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.12  # Reduced width to accommodate more bars

    # Create proxy artists for the legend
    from matplotlib.patches import Patch, Rectangle
    
    # Method labels (solid colors)
    method_patches = [
        Rectangle((0,0), 1, 1, facecolor='gray', label='Baseline\n(No Intervention)', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[0], label='DIM', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[1], label='RDO (Ours)', edgecolor='black', linewidth=0.5)
    ]
    
    # Condition labels (patterns)
    condition_patches = [
        Rectangle((0,0), 1, 1, facecolor='white', label='Directional\nAblation', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor='white', label='Activation\nSubtraction', edgecolor='black', linewidth=0.5, hatch='////')
    ]

    # Plot baseline scores
    ax.bar(x - 3*bar_width/2, baseline_scores, bar_width,
        color='gray',
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Plot ablation scores side by side
    ax.bar(x - bar_width/2, ablation_scores, bar_width,
        color=colors[0],
        alpha=1.0, edgecolor='black', linewidth=0.5)
    
    ax.bar(x + bar_width/2, retain_ablation_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Plot actadd scores side by side with hatching
    ax.bar(x + 3*bar_width/2, actadd_scores, bar_width,
        color=colors[0],
        alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

    ax.bar(x + 5*bar_width/2, retain_actadd_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

    # Customization
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='center')  # Rotated labels
    ax.tick_params(axis='both', which='major')

    # Grid styling
    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray', linewidth=0.5)

    # Adjust spines to be thinner
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Make tick marks thinner and shorter
    ax.tick_params(axis='both', width=0.5, length=3)

    # Y-axis limits and spines
    ax.set_ylim(0, 1)  # Slightly higher to accommodate error bars
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Create two-part legend with better positioning
    legend1 = ax.legend(handles=method_patches, title='Method', 
                    bbox_to_anchor=(1.0, 1.0),  # Position to the right
                    loc='upper left', ncol=1,
                    frameon=True, fancybox=False, 
                    edgecolor='black',
                    borderaxespad=0)
    ax.add_artist(legend1)  # Add first legend
    
    ax.legend(handles=condition_patches, title='Operation',
              bbox_to_anchor=(1.0, 0.5),  # Position to the right, below the first legend
              loc='upper left', ncol=1,
              frameon=True, fancybox=False,
              edgecolor='black', 
              borderaxespad=0)
    ax.set_title(f"{dataset}")
    # Save
    # plt.tight_layout()
    os.makedirs("results/plots/rdo/combined", exist_ok=True)
    path = f"results/plots/rdo/combined/{dataset}_scores.png"
    plt.savefig(path, dpi=300)
    plt.show()

plot_scores(y_label="Attack Success Rate", dataset="JailbreakBench")
plot_scores(y_label="Attack Success Rate", dataset="StrongREJECT")
plot_scores(y_label="Attack Success Rate", dataset="SORRY-Bench")