# %%
import torch
from nnsight import LanguageModel
import json
import numpy as np
from torch import Tensor
from jaxtyping import Float
import os

torch.set_grad_enabled(False)

#%%
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
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

colors = sns.color_palette('colorblind')
colors[0], colors[-1] = colors[-1], colors[0]
# %%
MODEL_PATH = 'google/gemma-2-2b-it'
# MODEL_PATH = 'meta-llama/Meta-Llama-3-8B-Instruct'
# MODEL_PATH = 'Qwen/Qwen2.5-7B-Instruct'
CACHE_DIR = '/ceph/hdd/students/elsj/huggingface'

model = LanguageModel(MODEL_PATH, cache_dir=CACHE_DIR, device_map='auto', torch_dtype=torch.bfloat16)

# %%
with model.trace("Hello") as _:
    pass
# %%
GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

QWEN25_CHAT_TEMPLATE = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

if "gemma" in MODEL_PATH.lower():
    TEMPLATE = GEMMA_CHAT_TEMPLATE
elif "qwen2.5" in MODEL_PATH.lower():
    TEMPLATE = QWEN25_CHAT_TEMPLATE
elif "llama-3" in MODEL_PATH.lower():
    TEMPLATE = LLAMA3_CHAT_TEMPLATE
else:
    raise ValueError(f"Model {MODEL_PATH} not supported")

SAVE_DIR = f"results/directopt/{MODEL_PATH.split('/')[-1]}/"
os.makedirs(SAVE_DIR, exist_ok=True)

# harmful_val = json.load(open('data/processed/jailbreakbench.json'))
harmful_val = json.load(open('data/saladbench_splits/harmful_val.json'))
harmful_val = json.load(open('data/processed/sorrybench.json')) # TODO dataset
harmless_val = json.load(open('data/saladbench_splits/harmless_val.json'))[:len(harmful_val)]
print(len(harmful_val), len(harmless_val))

# %%
from nnsight.envoy import Envoy #
import einops

def projection_einops(activation, direction):
    proj = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction
    )
    return proj

def intervene_with_fn_vector_ablation(
    model: LanguageModel,
    prompt: str,
    fn_vector: Float[Tensor, "d_model"],
    max_new_tokens: int = 150,
):
    fn_vector = fn_vector / fn_vector.norm()

    start_token = len(model.tokenizer([prompt], add_special_tokens=True, padding=False, truncation=False)["input_ids"][0])
        
    with model.generate(max_new_tokens=max_new_tokens, do_sample=False) as generator:
        with generator.invoke([prompt]) as invoker:
            tokens_intervention = model.generator.output.save()
            for n in range(max_new_tokens - 1):
                for layer in model.model.layers:
                    layer.input[:] -= projection_einops(layer.input[:], fn_vector)
                    layer.self_attn.output[0][:] -= projection_einops(layer.self_attn.output[0][:], fn_vector)
                    layer.mlp.output[:] -= projection_einops(layer.mlp.output[:], fn_vector)
                generator.next()

    completion = model.tokenizer.batch_decode(tokens_intervention.value[:, start_token:], skip_special_tokens=True)
    return completion[0]

# %%
module = model.model

# %%
raw_harmful_instructions = [d["instruction"] for d in harmful_val]
harmful_instructions = [TEMPLATE.format(instruction=d["instruction"]) for d in harmful_val]
raw_harmless_instructions = [d["instruction"] for d in harmless_val]
harmless_instructions = [TEMPLATE.format(instruction=d["instruction"]) for d in harmless_val]
print(len(raw_harmful_instructions), len(raw_harmless_instructions))

# %%
from scoring import get_refusal_scores
if "gemma" in MODEL_PATH.lower():
    refusal_tokens = [235285]
elif "qwen2.5" in MODEL_PATH.lower():
    refusal_tokens = [40, 2121]
elif "llama-3" in MODEL_PATH.lower():
    refusal_tokens = [40]
else:
    raise ValueError(f"Model {MODEL_PATH} not supported")

filter_data = True
batch_size = 16
if filter_data:
    print("Filtering data")
    harmful_scores = get_refusal_scores(model, harmful_instructions, refusal_tokens, batch_size=batch_size)
    harmless_scores = get_refusal_scores(model, harmless_instructions, refusal_tokens, batch_size=batch_size)
    filtered_harmful_instructions = [d for d, score in zip(harmful_instructions, harmful_scores) if score > 0]

    filtered_harmless_instructions = [d for d, score in zip(harmless_instructions, harmless_scores) if score < 0]
    print(f"Remaining harmful instances: {len(filtered_harmful_instructions)}")
    filtered_harmless_instructions = filtered_harmless_instructions[:len(filtered_harmful_instructions)]
    print(f"Remaining harmless instances: {len(filtered_harmless_instructions)}")

    harmful_instructions = filtered_harmful_instructions
    harmless_instructions = filtered_harmless_instructions

# %%
def dotproduct_einops(activation, direction):
    dot = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
    )
    return dot

def get_cosine_similarities(model, prompt, measure_vector, intervention_vector=None, measure="after"):
    cosine_similarities = []
    dot_products = []
    if intervention_vector is not None:
        norm_intervention_vector = intervention_vector / intervention_vector.norm()
    with model.trace([prompt]) as _:
        for layer in model.model.layers:
            act = layer.input
            if intervention_vector is not None:
                ablated_act = act - projection_einops(act, norm_intervention_vector)
                layer.input = ablated_act 
                layer.self_attn.output[0][:] -= projection_einops(layer.self_attn.output[0][:], norm_intervention_vector)
                layer.mlp.output[:] -= projection_einops(layer.mlp.output[:], norm_intervention_vector)
            else:
                ablated_act = act
            measure_act = ablated_act if measure == 'after' else act
            measure_act = measure_act[0, -1]
            cosine_sim = torch.nn.functional.cosine_similarity(measure_act, measure_vector, dim=-1).save()
            dot_product = dotproduct_einops(measure_act, measure_vector).save()
            cosine_similarities.append(cosine_sim)
            dot_products.append(dot_product)
    return [c.value.item() for c in cosine_similarities], [d.value.item() for d in dot_products]

from tqdm import tqdm
def new_get_cosine_similarities(model, prompts, measure_vector, intervention_vector=None, measure="after", batch_size=16):
    # Initialize list of lists to store cosine similarities per layer
    layer_cosine_sims = [[] for _ in range(len(model.model.layers))]
    layer_dot_products = [[] for _ in range(len(model.model.layers))]
    
    if intervention_vector is not None:
        norm_intervention_vector = intervention_vector / intervention_vector.norm()
        
    for i in range(0, len(prompts), batch_size):
        instructions = prompts[i:i+batch_size]
        with model.trace(instructions) as _:
            for layer_idx, layer in enumerate(model.model.layers):
                act = layer.input
                if intervention_vector is not None:
                    ablated_act = act - projection_einops(act, norm_intervention_vector)
                    layer.input = ablated_act 
                    layer.self_attn.output[0][:] -= projection_einops(layer.self_attn.output[0][:], norm_intervention_vector)
                    layer.mlp.output[:] -= projection_einops(layer.mlp.output[:], norm_intervention_vector)
                else:
                    ablated_act = act
                measure_act = ablated_act if measure == 'after' else act
                measure_act = measure_act[:, -1]
                cosine_sim = torch.nn.functional.cosine_similarity(measure_act, measure_vector, dim=-1).detach().save()
                layer_cosine_sims[layer_idx].append(cosine_sim)
                dot_product = dotproduct_einops(measure_act, measure_vector).detach().squeeze().save()
                layer_dot_products[layer_idx].append(dot_product)
    
    # Concatenate all values for each layer
    all_values = []
    for layer_sims in layer_cosine_sims:
        layer_values = torch.cat([l.value for l in layer_sims])
        all_values.append(layer_values)
    all_dot_products = []
    for layer_dot_products in layer_dot_products:
        layer_dot_products = torch.cat([l.value for l in layer_dot_products])
        all_dot_products.append(layer_dot_products)
    return torch.stack(all_values, dim=1).to(torch.float32).cpu().numpy(), torch.stack(all_dot_products, dim=1).to(torch.float32).cpu().numpy()

def new_get_cosine_similarities_single_layer(model, prompts, measure_vector, intervention_vector, measure="after", batch_size=16):
    # Initialize list of lists to store cosine similarities per layer
    before_layer_cosine_sims = [[] for _ in range(len(model.model.layers))]
    layer_cosine_sims = [[] for _ in range(len(model.model.layers))]
    
    if intervention_vector is not None:
        norm_intervention_vector = intervention_vector / intervention_vector.norm()
        
    for i in range(0, len(prompts), batch_size):
        instructions = prompts[i:i+batch_size]
        with model.trace(instructions) as _:
            for layer_idx, layer in enumerate(model.model.layers):
                act = layer.input.detach().clone()
                ablated_act = act - projection_einops(act, norm_intervention_vector)
                after_measure_act = ablated_act[:, -1]
                cosine_sim = torch.nn.functional.cosine_similarity(after_measure_act, measure_vector, dim=-1).detach().save()
                layer_cosine_sims[layer_idx].append(cosine_sim)

                before_measure_act = act[:, -1]
                cosine_sim = torch.nn.functional.cosine_similarity(before_measure_act, measure_vector, dim=-1).detach().save()
                before_layer_cosine_sims[layer_idx].append(cosine_sim)

                # nnsight.log("norm", after_measure_act.norm() - before_measure_act.norm())
    
    # Concatenate all values for each layer
    all_values = []
    for layer_sims in layer_cosine_sims:
        layer_values = torch.cat([l.value for l in layer_sims])
        all_values.append(layer_values)
    # before_values = []  
    # for layer_sims in before_layer_cosine_sims:
    #     layer_values = torch.cat([l.value for l in layer_sims])
    #     before_values.append(layer_values)
    return torch.stack(all_values, dim=1).to(torch.float32).cpu().numpy() #, torch.stack(before_values, dim=1).to(torch.float32).cpu().numpy()

def new_get_cosine_similarities_single_layer_sanity_check(model, prompts, measure_vector, intervention_vector, measure="after", batch_size=16):
    # Collect only the last-token activations for each layer
    all_last_token_acts = [[] for _ in range(len(model.model.layers))]
    norm_intervention_vector = intervention_vector / intervention_vector.norm()
    for i in range(0, len(prompts), batch_size):
        instructions = prompts[i:i+batch_size]
        with model.trace(instructions) as _:
            for layer_idx, layer in enumerate(model.model.layers):
                act = layer.input  # shape: (batch, seq_length, hidden)
                last_token_act = act[:, -1]  # only save the activation at the last token
                all_last_token_acts[layer_idx].append(last_token_act.detach().save())
    
    # Compute cosine similarities for each layer using the saved last-token activations
    cosine_sims_per_layer = []
    for layer_idx, acts_list in enumerate(all_last_token_acts):
        token_list_before = []
        token_list_after = []
        for saved_act in acts_list:
            act_tensor = saved_act.value  # shape: (batch, hidden)
            token_list_before.append(act_tensor)
            if measure == "after":
                act_tensor_after = act_tensor - projection_einops(act_tensor, norm_intervention_vector.cuda())
            else:
                act_tensor_after = act_tensor
            token_list_after.append(act_tensor_after)
        # Concatenate activations for this layer
        acts_cat_before = torch.cat(token_list_before, dim=0)
        acts_cat_after = torch.cat(token_list_after, dim=0)
        # Compute cosine similarities before and after intervention
        cs_before = torch.nn.functional.cosine_similarity(acts_cat_before, measure_vector.cuda(), dim=-1).detach()
        cs_after = torch.nn.functional.cosine_similarity(acts_cat_after, measure_vector.cuda(), dim=-1).detach()
        # Compute norms before and after intervention
        norm_before = torch.norm(acts_cat_before, dim=-1).detach()
        norm_after = torch.norm(acts_cat_after, dim=-1).detach()
        # Compute changes from before to after
        change_cs = cs_after - cs_before
        change_norm = norm_after - norm_before
        # Print the change in cosine similarity and norm for the current layer
        print(f"Layer {layer_idx}: Mean change in cosine similarity: {change_cs.mean().item():.4f}, Mean cosine similarity before: {cs_before.mean().item():.4f}, Mean cosine similarity after: {cs_after.mean().item():.4f}, Mean change in norm: {change_norm.mean().item():.4f}, mean norm before: {norm_before.mean().item():.4f}, mean norm after: {norm_after.mean().item():.4f}")
        # Append the after cosine similarities for later use
        cosine_sims_per_layer.append(cs_after)
    
    return torch.stack(cosine_sims_per_layer, dim=1).to(torch.float32).cpu().numpy() 


def get_cosine_similarities_batch(model, prompts, measure_vector, intervention_vector=None, measure="after"):
    cosine_similarities = []
    dot_products = []
    for prompt in prompts:
        cosine_sim, dot_product = get_cosine_similarities(model, prompt, measure_vector, intervention_vector, measure)
        cosine_similarities.append(cosine_sim)
        dot_products.append(dot_product)
    return np.mean(cosine_similarities, axis=0), np.mean(dot_products, axis=0)

def get_cosine_similarities_batch(model, prompts, measure_vector, intervention_vector=None, measure="after"):
    cosine_similarities = []
    dot_products = []
    for prompt in prompts:
        cosine_sim, dot_product = get_cosine_similarities(model, prompt, measure_vector, intervention_vector, measure)
        cosine_similarities.append(cosine_sim)
        dot_products.append(dot_product)
    return np.mean(cosine_similarities, axis=0), np.mean(dot_products, axis=0)

def get_dot_products_before_intervention(model, prompts, vector, layers=range(len(model.model.layers))):
    dot_products = []
    normalized_vector = vector / vector.norm()
    for prompt in prompts:
        prompt_dot_products = []
        with model.trace([prompt]) as _:
            for layer_idx in layers:
                last_token_act = model.model.layers[layer_idx].output[0][0, -1]
                dot_product = dotproduct_einops(last_token_act, normalized_vector).abs().save()
                prompt_dot_products.append(dot_product)

                act = model.model.layers[layer_idx].output[0][:]
                ablated_act = act - projection_einops(act, normalized_vector)
                model.model.layers[layer_idx].output[0][:] = ablated_act 

        dot_products.append(prompt_dot_products)
    dot_products = [[d.item() for d in prompt_dot_products] for prompt_dot_products in dot_products]
    return np.mean(dot_products, axis=0)

def get_cosine_similarities_during_generation(model, prompt, measure_vector, intervention_vector=None, measure="after", measure_layer=12, max_new_tokens=10):
    cosine_similarities = []
    if intervention_vector is not None:
        norm_intervention_vector = intervention_vector / intervention_vector.norm()
    with model.generate(max_new_tokens=max_new_tokens, do_sample=False) as generator:
        with generator.invoke([prompt]) as invoker:
            for n in range(max_new_tokens - 1):
                for layer_idx, layer in enumerate(model.model.layers):
                    act = layer.input
                    if intervention_vector is not None:
                        ablated_act = act - projection_einops(act, norm_intervention_vector)
                        layer.input = ablated_act 
                        layer.self_attn.output[0][:] -= projection_einops(layer.self_attn.output[0][:], norm_intervention_vector)
                        layer.mlp.output[:] -= projection_einops(layer.mlp.output[:], norm_intervention_vector)
                    else:
                        ablated_act = act
                    measure_act = ablated_act if measure == 'after' else act
                    measure_act = measure_act[0, -1]
                    cosine_sim = torch.nn.functional.cosine_similarity(measure_act, measure_vector, dim=-1).save()
                    if measure_layer == layer_idx:
                        cosine_similarities.append(cosine_sim)
                    generator.next()

    return [c.value.item() for c in cosine_similarities]


# %%
model_id = MODEL_PATH.split("/")[-1]
refusal_directions = torch.load(f"results/refusal_dir/{model_id}/generate_directions/mean_diffs.pt")
refusal_results = json.load(open(f"results/refusal_dir/{model_id}/direction_metadata.json"))
best_layer = refusal_results["layer"]
best_token = refusal_results["pos"]
best_refusal_direction = torch.load(f"results/refusal_dir/{model_id}/direction.pt").to(model.dtype)

# %%
import matplotlib.pyplot as plt

def plot_data(data_list, x, labels, title="Similarities Across Model Layers", y_label="Similarity", figsize=(12, 7)):
    from matplotlib import pyplot as plt

    plt.figure(figsize=figsize)
    for data, label in zip(data_list, labels):
        plt.plot(x, data, label=label)

    plt.xlabel("Layer")
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(x, x)
    plt.legend()
    plt.grid(True)
    plt.show()

#%% 
import jsonlines
instructions = []
suffixes = []
for i in range(8):
    gcg_files_path = f"results/repind_gcg/aux_50_newtargets/{i}.jsonl"
    if os.path.exists(gcg_files_path):
        with jsonlines.open(gcg_files_path) as reader:
            for item in reader:
                instructions.append(item["instruction"])
                suffixes.append(item["best_string"])

adversarial_instructions = [i + " " + s for i, s in zip(instructions, suffixes)]
adversarial_instructions = [TEMPLATE.format(instruction=instruction) for instruction in adversarial_instructions]

print(len(adversarial_instructions))
# %%
baseline_cosine_sims = {}
baseline_cosine_sims_adversarial = {}

vector = torch.load(f"vectors/lowest_loss_vector (2).pt").to(model.dtype) 
vector_names = ["repind"]
intervention_vectors = [vector]

for vector_name, vector in zip(vector_names, intervention_vectors):
    # Get similarities for harmful instructions
    cosine_sims, _ = new_get_cosine_similarities(model, harmful_instructions, measure_vector=vector, intervention_vector=None, batch_size=batch_size)
    baseline_cosine_sims[vector_name] = cosine_sims
    
    # Get similarities for adversarial instructions
    cosine_sims, _ = new_get_cosine_similarities(model, adversarial_instructions, measure_vector=vector, intervention_vector=None, batch_size=batch_size)
    baseline_cosine_sims_adversarial[vector_name] = cosine_sims

# %%
# Calculate means and standard deviations
baseline_means = baseline_cosine_sims[vector_name].mean(axis=0)
baseline_stds = baseline_cosine_sims[vector_name].std(axis=0)
adversarial_means = baseline_cosine_sims_adversarial[vector_name].mean(axis=0)
adversarial_stds = baseline_cosine_sims_adversarial[vector_name].std(axis=0)

# Create plot
MARKER_SIZE = 0

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))
x = range(model.config.num_hidden_layers)
y_min = min(min(baseline_means - baseline_stds), min(adversarial_means - adversarial_stds))
y_max = max(max(baseline_means + baseline_stds), max(adversarial_means + adversarial_stds))

# Plot original data
ax.plot(x, baseline_means, color=colors[0], marker='o', 
        markersize=MARKER_SIZE, label='RepInd on p$_{harm}$')
ax.fill_between(x, baseline_means - baseline_stds, baseline_means + baseline_stds, 
                color=colors[0], alpha=0.15)

# Plot adversarial data
ax.plot(x, adversarial_means, color=colors[0], marker='s', 
        markersize=MARKER_SIZE, linestyle='--', label='RepInd on p$_{adv}$')
ax.fill_between(x, adversarial_means - adversarial_stds, adversarial_means + adversarial_stds,
                color=colors[0], alpha=0.25)

# Configure axis styling
ax.grid(True, linestyle='--')

# Labels with larger fonts
ax.set_xlabel('Layer')
ax.set_ylabel('Cosine similarity')

# Spine styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tick parameters
ax.tick_params(width=0.5)
ax.set_xticks(range(0, len(x), 5))  # x ticks every 5 layers
ax.set_ylim(y_min - 0.01, y_max + 0.05)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))  # y ticks every 0.1

# Place legend above plot
plt.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95), ncol=1)

plt.tight_layout()

# Save figure before showing
plt.savefig(f"results/plots/repind/gcg_repind_new.png", dpi=300, bbox_inches='tight')
plt.show()

# %%

# %%
def get_completions(
    model: LanguageModel,
    dataset: list[str],
    max_new_tokens: int = 150,
    batch_size: int = 16,
):
    all_completions = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        instructions = dataset[i:i+batch_size]
        start_token = len(model.tokenizer(instructions, add_special_tokens=True, padding=True, truncation=False)["input_ids"][0])
        
        with model.generate(instructions, max_new_tokens=max_new_tokens, do_sample=False) as generator:
            tokens = model.generator.output.save()

        completion = model.tokenizer.batch_decode(tokens.value[:, start_token:], skip_special_tokens=True)
        all_completions.extend(completion)

    return all_completions

# Get completions for first 10 adversarial instructions
adversarial_completions = get_completions(model, adversarial_instructions, 
                                        max_new_tokens=512, batch_size=8)

# %%
with open('adversarial_results.txt', 'w') as f:
    for i, (instruction, completion) in enumerate(zip(adversarial_instructions[:10], adversarial_completions), 1):
        f.write(f"\nExample {i}:\n")
        f.write(f"Instruction: {instruction}\n")
        f.write(f"Completion: {completion}\n") 
        f.write("-" * 100 + "\n")

# %%
from strong_reject.evaluate import evaluate_dataset
from datasets import Dataset
EVALUATOR = "strongreject_finetuned"

# Create dataset with prompts in forbidden_prompt column
dataset = Dataset.from_dict({"forbidden_prompt": instructions})
responses_dataset = dataset.add_column("response", adversarial_completions)
# use the StrongREJECT evaluator to score the harmfulness of the responses
eval_dataset = evaluate_dataset(responses_dataset, [EVALUATOR], batch_size=8, empty_model_cache=False)

print(f"StrongREJECT score: {np.mean(eval_dataset['score'])}")
print(f"StrongREJECT score std: {np.std(eval_dataset['score'])}")
# %%
eval_dataset.to_json("evaluation_results.json")

# %%

run_names = [
    "rep_ind_1_run_2",
    "rep_ind_2_run_3", 
    "rep_ind_3_run_2",
    "rep_ind_4_run_3",
    "rep_ind_5_run_1",
]
group = "repind_symmetric_sum_loss_gemma-2-2b-it"
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
            print(run.name)
            # Download file from the run's files
            file_path = "completions/jailbreakbench_ablation_evaluations.json"
            download_dir = os.path.join("wandb_downloads", run.group, run.name)
            os.makedirs(download_dir, exist_ok=True)
            run.file(file_path).download(root=download_dir, exist_ok=True)
            file_path = os.path.join(download_dir, file_path)
            eval_results.append(json.load(open(file_path)))
# %%
eval_results.append(json.load(open("results/refusal_dir/gemma-2-2b-it/completions/jailbreakbench_ablation_evaluations.json")))

# %%
eval_scores = []
for result in eval_results:
    scores = []
    for completion in result["completions"]:
        score = completion["is_jailbreak_strongreject"]
        scores.append(score)
    eval_scores.append(scores)
    print(np.array(scores).mean())
# %%
for i in range(1, len(eval_scores)+1):
    print(np.array(eval_scores[:i]).max(axis=0).mean())

# %% # Get baseline similarities without intervention
baseline_cosine_sims = {}
baseline_cosine_sims_harmless = {}

for vector_name, vector in zip(vector_names, intervention_vectors):
    # Get similarities for harmful instructions
    cosine_sims = new_get_cosine_similarities(model, harmful_instructions, vector, batch_size=48)
    baseline_cosine_sims[vector_name] = cosine_sims
    
    # # Get similarities for harmless instructions
    cosine_sims = new_get_cosine_similarities(model, harmless_instructions, vector, batch_size=48)
    baseline_cosine_sims_harmless[vector_name] = cosine_sims

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle('Baseline Cosine Similarities Between Directions and Mean Activations Across Model Layers')

# Find global min/max for consistent y-axis
y_min = min(
    min([min(baseline_cosine_sims[v]) for v in vector_names]),
    min([min(baseline_cosine_sims_harmless[v]) for v in vector_names])
)
y_max = max(
    max([max(baseline_cosine_sims[v]) for v in vector_names]),
    max([max(baseline_cosine_sims_harmless[v]) for v in vector_names])
)

x = range(model.config.num_hidden_layers)
for i, vector_name in enumerate(vector_names):
    # Use same color for both plots but different line styles
    color = f'C{i}'
    ax1.plot(x, baseline_cosine_sims[vector_name], label=vector_name, color=color, linestyle='-', marker='o', markersize=4)
    ax2.plot(x, baseline_cosine_sims_harmless[vector_name], label=vector_name, color=color, linestyle='-', marker='o', markersize=4)

for ax in [ax1, ax2]:
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(y_min, y_max)  # Set same y-axis limits
    ax.set_xticks(x)  # Set x-axis ticks to match x values

ax1.set_title('Harmful Instructions')
ax2.set_title('Harmless Instructions')

plt.tight_layout()
plt.show()

# %% 
# plot the difference between the cosine similarities of harmful and harmless instructions
cosine_diff = [baseline_cosine_sims[v] - baseline_cosine_sims_harmless[v] for v in vector_names]
plot_data(cosine_diff, x, vector_names, f"Difference in Cosine Similarities Between Harmful and Harmless Instructions Across Model Layers", "Cosine Similarity Difference")

# %%
from matplotlib import rc
plt.style.use('default')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# %%
# Compute similarities after intervention
all_cosine_sims = {}
cosine_sim_ratios = {}
cosine_sim_changes = {}

# ratio_threshold = 0.02
for intervention_vector, vector_name in zip(intervention_vectors, vector_names):
    all_cosine_sims[vector_name] = {}
    cosine_sim_ratios[vector_name] = {}
    cosine_sim_changes[vector_name] = {}
    
    for name, direction in zip(vector_names, intervention_vectors):
        cosine_sims = new_get_cosine_similarities(model, harmful_instructions, direction, intervention_vector, batch_size=48)
        all_cosine_sims[vector_name][name] = cosine_sims
        
        # Calculate ratio and change of after/before
        ratio = [a/b if b != 0 else 0 for a, b in zip(cosine_sims, baseline_cosine_sims[name])]
        change = [a - b for a, b in zip(cosine_sims, baseline_cosine_sims[name])]
        cosine_sim_ratios[vector_name][name] = ratio
        cosine_sim_changes[vector_name][name] = change

# %%
# Plot the results
basis_vector_performances = [0.76, 0.66, 0.55, 0.2]
fig, axs = plt.subplots(len(vector_names), 3, figsize=(20, 4.5*len(vector_names)), sharex=True)

cosine_sim_min = float('inf')
cosine_sim_max = float('-inf')
change_min = float('inf')
change_max = float('-inf')
ratio_min = float('inf')
ratio_max = float('-inf')

for vector_name in vector_names:
    for name in vector_names:
        change = cosine_sim_changes[vector_name][name]
        ratio = cosine_sim_ratios[vector_name][name]
        
        change_min = min(change_min, min(change))
        change_max = max(change_max, max(change))
        ratio_min = min(ratio_min, min(ratio))
        ratio_max = max(ratio_max, max(ratio))

        cosine_sim_min = min(cosine_sim_min, min(all_cosine_sims[vector_name][name]))
        cosine_sim_max = max(cosine_sim_max, max(all_cosine_sims[vector_name][name]))

for vector_name in vector_names:
    for name in vector_names:
        change = cosine_sim_changes[vector_name][name]

        if change_max in change:
            max_change_idx = np.where(change == change_max)[0][0]
            print(f"Max change {change_max:.3f} when ablating {vector_name} measuring {name}")
            print(f"  Baseline cosine similarity: {baseline_cosine_sims[name][max_change_idx]:.3f}")
            print(f"  After ablation: {all_cosine_sims[vector_name][name][max_change_idx]:.3f}")
            print(f"  At layer: {max_change_idx}")
            
        if change_min in change:
            min_change_idx = np.where(change == change_min)[0][0]
            print(f"Min change {change_min:.3f} when ablating {vector_name} measuring {name}")
            print(f"  Baseline cosine similarity: {baseline_cosine_sims[name][min_change_idx]:.3f}")
            print(f"  After ablation: {all_cosine_sims[vector_name][name][min_change_idx]:.3f}")
            print(f"  At layer: {min_change_idx}")

# Add padding to the limits
cosine_sim_padding = (cosine_sim_max - cosine_sim_min) * 0.04
cosine_sim_min -= cosine_sim_padding
cosine_sim_max += cosine_sim_padding

change_padding = (change_max - change_min) * 0.04
change_min -= change_padding
change_max += change_padding

ratio_padding = (ratio_max - ratio_min) * 0.04
ratio_min -= ratio_padding
ratio_max += ratio_padding

# Store line objects for the legend
lines = []
labels = []

for i, vector_name in enumerate(vector_names):
    # Plot cosine similarities after intervention
    for name in vector_names:
        line = axs[i, 0].plot(range(model.config.num_hidden_layers),
                             all_cosine_sims[vector_name][name],
                             linestyle='-',
                             marker='o',
                             markersize=4)
        if i == 0:  # Only store lines and labels from first row for legend
            lines.append(line[0])
            labels.append(name)
            
    # Add performance to title if it's a basis vector
    if "Basis Vector" in vector_name:
        basis_idx = int(vector_name.split()[-1]) - 1
        perf = basis_vector_performances[basis_idx]
        title = f"Cosine Similarities after {vector_name} Ablation (Performance: {perf:.2f})"
    else:
        title = f"Cosine Similarities after {vector_name} Vector Ablation"
        
    axs[i, 0].set_title(title)
    axs[i, 0].set_xlabel("Layer")
    axs[i, 0].set_ylabel("Cosine Similarity")
    axs[i, 0].grid(True)
    axs[i, 0].set_xticks(range(model.config.num_hidden_layers))
    axs[i, 0].set_ylim(cosine_sim_min, cosine_sim_max)
    axs[i, 0].legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot cosine similarity changes
    for name in vector_names:
        change = cosine_sim_changes[vector_name][name]
        axs[i, 1].plot(range(model.config.num_hidden_layers),
                       change,
                       linestyle='-',
                       marker='o',
                       markersize=4)
                       
    # Add performance to title if it's a basis vector
    if "Basis Vector" in vector_name:
        basis_idx = int(vector_name.split()[-1]) - 1
        perf = basis_vector_performances[basis_idx]
        title = f"Change in Cosine Similarity after {vector_name} Ablation (Performance: {perf:.2f})"
    else:
        title = f"Change in Cosine Similarity after {vector_name} Vector Ablation"
        
    axs[i, 1].set_title(title)
    axs[i, 1].set_xlabel("Layer")
    axs[i, 1].set_ylabel("Change in Cosine Similarity")
    axs[i, 1].grid(True)
    axs[i, 1].set_xticks(range(model.config.num_hidden_layers))
    axs[i, 1].set_ylim(change_min, change_max)
    axs[i, 1].legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot cosine similarity ratios
    for name in vector_names:
        ratio = cosine_sim_ratios[vector_name][name]
        axs[i, 2].plot(range(model.config.num_hidden_layers),
                       ratio,
                       linestyle='-',
                       marker='o',
                       markersize=4)
                       
    # Add performance to title if it's a basis vector
    if "Basis Vector" in vector_name:
        basis_idx = int(vector_name.split()[-1]) - 1
        perf = basis_vector_performances[basis_idx]
        title = f"Ratio of Cosine Similarities after {vector_name} Ablation (Performance: {perf:.2f})"
    else:
        title = f"Ratio of Cosine Similarities after {vector_name} Vector Ablation"
        
    axs[i, 2].set_title(title)
    axs[i, 2].set_xlabel("Layer")
    axs[i, 2].set_ylabel("Ratio of Cosine Similarities")
    axs[i, 2].grid(True)
    axs[i, 2].set_xticks(range(model.config.num_hidden_layers))
    axs[i, 2].set_ylim(-2, 2)
    axs[i, 2].legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

# %%
import wandb
vector_configs = [
    {"run_id": "dlnphp08", "run_name": "RDO$_\perp$"},
    {"run_id": "y8sfmb74", "run_name": "Rep. Ind."},
    # {"run_id": "0z65jdvt", "run_name": "Rep. Ind."},
]
api = wandb.Api()
vectors = []
for vector_config in vector_configs:
    run_id = vector_config["run_id"]
    run = api.run(f"refusal-representations/robust_refusal_vector/{run_id}")
    candidate_idx = run.summary.get("val/candidate_idx")
    artifact = api.artifact(f'refusal-representations/robust_refusal_vector/trained_vectors_run_{run_id}:v0', type='vector')
    artifact_dir = artifact.download()
    print(f"Artifact dir: {artifact_dir}")
    print(f"Vector config: {vector_config}")
    if 'Ind' in vector_config["run_name"]:
        print("Loading lowest loss vector")
        direction = torch.load(os.path.join(artifact_dir, "lowest_loss_vector.pt"))
        direction = direction - projection_einops(direction, best_refusal_direction.cpu().to(direction.dtype) / best_refusal_direction.norm().cpu().to(direction.dtype)) # TODO: remove this
    else:
        direction = torch.load(os.path.join(artifact_dir, "vector.pt"))[-20+candidate_idx]
    vectors.append(direction.to(model.dtype))

# %%
# vectors[-1] = torch.load("vectors/gemma2repind1.pt")[-1].to(model.dtype)
vectors[-1] = torch.load("lowest_loss_vector (4).pt")[-1].to(model.dtype)

# vectors[-1] = torch.load("independent_vectors_cutoff09_repind1000_final.pt")[1].to(model.dtype)
# %%
vector_names = ["DIM"] + [c["run_name"] for c in vector_configs]
intervention_vectors = [best_refusal_direction] + vectors

 # %%
torch.nn.functional.cosine_similarity(intervention_vectors[-1].cpu(), intervention_vectors[0].cpu(), dim=-1)

# %%
batch_size = 16

baseline_cosine_sims = {}
baseline_cosine_sims_harmless = {}
baseline_dot_products = {}
baseline_dot_products_harmless = {}

for vector_name, vector in zip(vector_names, intervention_vectors):
    # Get similarities for harmful instructions
    cosine_sims, dot_products = new_get_cosine_similarities(model, harmful_instructions, vector, batch_size=batch_size)
    baseline_cosine_sims[vector_name] = cosine_sims
    baseline_dot_products[vector_name] = dot_products
    
    # # Get similarities for harmless instructions
    cosine_sims, dot_products = new_get_cosine_similarities(model, harmless_instructions, vector, batch_size=batch_size)
    baseline_cosine_sims_harmless[vector_name] = cosine_sims
    baseline_dot_products_harmless[vector_name] = dot_products

# %% harmful_means = {}
harmful_stds = {}
harmless_means = {}
harmless_stds = {}

for vector_name in vector_names:
    harmful_means[vector_name] = baseline_cosine_sims[vector_name].mean(axis=0)
    harmful_stds[vector_name] = baseline_cosine_sims[vector_name].std(axis=0)
    harmless_means[vector_name] = baseline_cosine_sims_harmless[vector_name].mean(axis=0)
    harmless_stds[vector_name] = baseline_cosine_sims_harmless[vector_name].std(axis=0)

# Create plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle('Baseline Cosine Similarities Between Directions and Mean Activations Across Model Layers')

y_min = min(
    min([min(harmful_means[v] - harmful_stds[v]) for v in vector_names]),
    min([min(harmless_means[v] - harmless_stds[v]) for v in vector_names])
)
y_max = max(
    max([max(harmful_means[v] + harmful_stds[v]) for v in vector_names]), 
    max([max(harmless_means[v] + harmless_stds[v]) for v in vector_names])
)

x = range(model.config.num_hidden_layers)
for i, vector_name in enumerate(vector_names):
    print(vector_name, harmful_means[vector_name])
    color = f'C{i}'
    
    ax1.plot(x, harmful_means[vector_name], label=vector_name, color=color, linestyle='-', marker='o', markersize=4)
    ax1.fill_between(x, harmful_means[vector_name] - harmful_stds[vector_name], 
                     harmful_means[vector_name] + harmful_stds[vector_name], color=color, alpha=0.2)
    
    ax2.plot(x, harmless_means[vector_name], label=vector_name, color=color, linestyle='-', marker='o', markersize=4)
    ax2.fill_between(x, harmless_means[vector_name] - harmless_stds[vector_name],
                     harmless_means[vector_name] + harmless_stds[vector_name], color=color, alpha=0.2)

for ax in [ax1, ax2]:
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)

ax1.set_title('Harmful Instructions')
ax2.set_title('Harmless Instructions')

plt.tight_layout()
plt.show()

# %% 
# plot the difference between the cosine similarities of harmful and harmless instructions
# cosine_diff = [baseline_cosine_sims[v] - baseline_cosine_sims_harmless[v] for v in vector_names]
# plot_data(cosine_diff, x, vector_names, f"Difference in Cosine Similarities Between Harmful and Harmless Instructions Across Model Layers", "Cosine Similarity Difference")

# %%
# Compute similarities after intervention
all_cosine_sims = {}
cosine_sim_changes = {}
cosine_sim_stds = {}

for intervention_vector, intervention_name in zip(intervention_vectors, vector_names):
    all_cosine_sims[intervention_name] = {}
    cosine_sim_changes[intervention_name] = {}
    cosine_sim_stds[intervention_name] = {}
    for measure_vector, measure_name in zip(intervention_vectors, vector_names):
        cosine_sims, dot_products = new_get_cosine_similarities(model, harmful_instructions, measure_vector=measure_vector, intervention_vector=intervention_vector, batch_size=batch_size)
        all_cosine_sims[intervention_name][measure_name] = cosine_sims.mean(axis=0)
        
        # Calculate change from baseline and std
        cosine_sim_changes[intervention_name][measure_name] = cosine_sims.mean(axis=0) - baseline_cosine_sims[measure_name].mean(axis=0)
        cosine_sim_stds[intervention_name][measure_name] = cosine_sims.std(axis=0)

# %%
# cosine_sims = new_get_cosine_similarities(model, harmful_instructions[:1], measure_vector=intervention_vectors[0], intervention_vector=intervention_vectors[0], batch_size=batch_size)
# cosine_sims = new_get_cosine_similarities_single_layer_sanity_check(model, harmful_instructions[:8], measure_vector=intervention_vectors[1], intervention_vector=intervention_vectors[0], batch_size=batch_size)
# cosine_sims2 = new_get_cosine_similarities_single_layer_sanity_check(model, harmful_instructions[:2], measure_vector=intervention_vectors[0], intervention_vector=intervention_vectors[0], batch_size=batch_size)
# cosine_sims = new_get_cosine_similarities(model, harmful_instructions[:1], measure_vector=intervention_vectors[2], batch_size=batch_size)
# test_cos = new_get_cosine_similarities_single_layer(model, harmful_instructions[:1], measure_vector=intervention_vectors[2], intervention_vector=intervention_vectors[0], batch_size=batch_size)
# cosine_sims, test_cos, cosine_sims - test_cos
# %%
# torch.nn.functional.cosine_similarity(intervention_vectors[0].cpu(), intervention_vectors[1].cpu(), dim=-1)
# %%
# all_single_layer_cosine_sims = {}
# single_layer_cosine_sim_changes = {}
# single_layer_cosine_sim_stds = {}

# for intervention_vector, intervention_name in zip(intervention_vectors, vector_names):
#     all_single_layer_cosine_sims[intervention_name] = {}
#     single_layer_cosine_sim_changes[intervention_name] = {}
#     single_layer_cosine_sim_stds[intervention_name] = {}

#     for measure_vector, measure_name in zip(intervention_vectors, vector_names):
#         cosine_sims = new_get_cosine_similarities_single_layer(model, harmful_instructions, measure_vector=measure_vector, intervention_vector=intervention_vector, batch_size=batch_size)
#         all_single_layer_cosine_sims[intervention_name][measure_name] = cosine_sims.mean(axis=0)
        
#         # Calculate change from baseline and std
#         single_layer_cosine_sim_changes[intervention_name][measure_name] = cosine_sims.mean(axis=0) - baseline_cosine_sims[measure_name].mean(axis=0)
#         single_layer_cosine_sim_stds[intervention_name][measure_name] = cosine_sims.std(axis=0)

# %%
# Create figure with more spacing between subplots
# Get min/max across all relevant data for consistent y-axis
dim_baseline = baseline_cosine_sims[vector_names[0]].mean(axis=0)
orthogonal_baseline = baseline_cosine_sims[vector_names[1]].mean(axis=0)
repind_baseline = baseline_cosine_sims[vector_names[2]].mean(axis=0)
orthogonal_dim_ablated = all_cosine_sims[vector_names[0]][vector_names[1]]  # DiM ablated by orthogonal
# orthogonal_dim_ablated_single_layer = all_single_layer_cosine_sims[vector_names[0]][vector_names[1]]
dim_orthogonal_ablated = all_cosine_sims[vector_names[1]][vector_names[0]]  # Orthogonal ablated by DiM
print(dim_baseline - dim_orthogonal_ablated)
# dim_orthogonal_ablated_single_layer = all_single_layer_cosine_sims[vector_names[1]][vector_names[0]]
repind_dim_ablated = all_cosine_sims[vector_names[0]][vector_names[2]]  # RepInd ablated by DiM
# repind_dim_ablated_single_layer = all_single_layer_cosine_sims[vector_names[0]][vector_names[2]]
dim_repind_ablated = all_cosine_sims[vector_names[2]][vector_names[0]]  # DiM ablated by RepInd
# dim_repind_ablated_single_layer = all_single_layer_cosine_sims[vector_names[2]][vector_names[0]]

# print(max(orthogonal_baseline - orthogonal_dim_ablated_single_layer))

# max between dim_baseline and dim_repind_ablated
diffs = dim_baseline - dim_repind_ablated
print(max(diffs), np.argmax(diffs))

y_min = min(min(dim_baseline), min(orthogonal_baseline), min(repind_baseline),
            min(orthogonal_dim_ablated), min(dim_orthogonal_ablated),
            min(repind_dim_ablated), min(dim_repind_ablated))
y_max = max(max(dim_baseline), max(orthogonal_baseline), max(repind_baseline),
            max(orthogonal_dim_ablated), max(dim_orthogonal_ablated),
            max(repind_dim_ablated), max(dim_repind_ablated))
y_padding = (y_max - y_min) * 0.04
y_min -= y_padding
y_max += y_padding

x = range(model.config.num_hidden_layers)

# %%
# Styling parameters
MARKER_SIZE = 0  # Increased marker size
LINE_WIDTH = 2

# Create second version without first subplot
# figsize was 6,6 for paper
fig3, ((ax2, ax3), (ax4, ax5)) = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=True)

# Ax2: RDO$_\perp$ baseline with DIM ablated
ax2.set_title('(a)', pad=10)
ax2.plot(x, orthogonal_baseline, color=colors[2], marker='o',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='RDO$_\perp$')
ax2.fill_between(x, orthogonal_baseline - baseline_cosine_sims[vector_names[1]].std(axis=0),
                 orthogonal_baseline + baseline_cosine_sims[vector_names[1]].std(axis=0),
                 color=colors[2], alpha=0.15)
ax2.plot(x, orthogonal_dim_ablated, color=colors[2], marker='s',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle='--',
         label='RDO$_\perp$ with DIM abl.')
ax2.fill_between(x, orthogonal_dim_ablated - cosine_sim_stds[vector_names[0]][vector_names[1]],
                 orthogonal_dim_ablated + cosine_sim_stds[vector_names[0]][vector_names[1]],
                 color=colors[2], alpha=0.25)  # Lighter red for better contrast

# Ax3: DIM baseline with RDO$_\perp$ ablated
ax3.set_title('(b)', pad=10)
ax3.plot(x, dim_baseline, color=colors[0], marker='o',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='DIM')
ax3.fill_between(x, dim_baseline - baseline_cosine_sims[vector_names[0]].std(axis=0),
                 dim_baseline + baseline_cosine_sims[vector_names[0]].std(axis=0),
                 color=colors[0], alpha=0.15)
ax3.plot(x, dim_orthogonal_ablated, color=colors[0], marker='s',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle='--',
         label='DIM with\nRDO$_\perp$ abl.')
ax3.fill_between(x, dim_orthogonal_ablated - cosine_sim_stds[vector_names[1]][vector_names[0]],
                 dim_orthogonal_ablated + cosine_sim_stds[vector_names[1]][vector_names[0]],
                 color=colors[0], alpha=0.25)  # Lighter blue for better contrast

# Ax4: RepInd baseline with DIM ablated
ax4.set_title('(c)', pad=10)
ax4.plot(x, repind_baseline, color=colors[1], marker='o',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='RepInd')
ax4.fill_between(x, repind_baseline - baseline_cosine_sims[vector_names[2]].std(axis=0),
                 repind_baseline + baseline_cosine_sims[vector_names[2]].std(axis=0),
                 color=colors[1], alpha=0.15)
ax4.plot(x, repind_dim_ablated, color=colors[1], marker='s',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle='--',
         label='RepInd with DIM abl.')
ax4.fill_between(x, repind_dim_ablated - cosine_sim_stds[vector_names[0]][vector_names[2]],
                 repind_dim_ablated + cosine_sim_stds[vector_names[0]][vector_names[2]],
                 color=colors[1], alpha=0.25)  # Lighter orange for better contrast

# Ax5: DIM baseline with RepInd ablated
ax5.set_title('(d)', pad=10)
ax5.plot(x, dim_baseline, color=colors[0], marker='o',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='DIM')
ax5.fill_between(x, dim_baseline - baseline_cosine_sims[vector_names[0]].std(axis=0),
                 dim_baseline + baseline_cosine_sims[vector_names[0]].std(axis=0),
                 color=colors[0], alpha=0.15)
ax5.plot(x, dim_repind_ablated, color=colors[0], marker='s',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle='--',
         label='DIM with\nRepInd abl.')
ax5.fill_between(x, dim_repind_ablated - cosine_sim_stds[vector_names[2]][vector_names[0]],
                 dim_repind_ablated + cosine_sim_stds[vector_names[2]][vector_names[0]],
                 color=colors[0], alpha=0.25)  # Lighter blue for better contrast

# Configure all axes with enhanced styling
for ax in [ax2, ax3, ax4, ax5]:
    # Enhanced grid
    ax.grid(True, linestyle='--')
    
    # Better legend with larger font
    ax.legend(frameon=True, fancybox=False, framealpha=0.95,
              shadow=False,
              loc='upper left', bbox_to_anchor=(0.01, 1), ncol=1, handlelength=1.5)
    
    # Labels with larger fonts
    if ax in [ax4, ax5]:
        ax.set_xlabel('Layer')
    if ax in [ax2, ax4]:
        ax.set_ylabel('Cosine similarity')
    
    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tick parameters with larger font
    ax.tick_params(width=0.5)
    ax.set_xticks(range(0, len(x), 5))  # x ticks every 5 layers
    ax.set_ylim(y_min, y_max + 0.1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))  # y ticks every 0.1

# Adjust layout with increased vertical spacing between rows
plt.tight_layout(rect=[0, 0, 1, 0.85], h_pad=1.0, w_pad=0.3)

# Save second version
# plt.show()
os.makedirs('results/plots/crossovereffects', exist_ok=True)
# plt.savefig('results/plots/crossovereffects/orthogonal_four_plots_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Compute similarities after intervention
all_dot_products = {}
dot_product_changes = {}
dot_product_stds = {}

for intervention_vector, intervention_name in zip(intervention_vectors, vector_names):
    all_dot_products[intervention_name] = {}
    dot_product_changes[intervention_name] = {}
    dot_product_stds[intervention_name] = {}
    for measure_vector, measure_name in zip(intervention_vectors, vector_names):
        cosine_sims, dot_products = new_get_cosine_similarities(model, harmful_instructions, measure_vector=measure_vector, intervention_vector=intervention_vector, batch_size=batch_size)
        all_dot_products[intervention_name][measure_name] = dot_products.mean(axis=0)
        
        # Calculate change from baseline and std
        dot_product_changes[intervention_name][measure_name] = dot_products.mean(axis=0) - baseline_dot_products[measure_name].mean(axis=0)
        dot_product_stds[intervention_name][measure_name] = dot_products.std(axis=0)


# %%
# Create figure with more spacing between subplots
# Get min/max across all relevant data for consistent y-axis
dim_baseline = baseline_dot_products[vector_names[0]].mean(axis=0)
orthogonal_baseline = baseline_dot_products[vector_names[1]].mean(axis=0)
repind_baseline = baseline_dot_products[vector_names[2]].mean(axis=0)
orthogonal_dim_ablated = all_dot_products[vector_names[0]][vector_names[1]]  # DiM ablated by orthogonal
dim_orthogonal_ablated = all_dot_products[vector_names[1]][vector_names[0]]  # Orthogonal ablated by DiM
print(dim_baseline - dim_orthogonal_ablated)
repind_dim_ablated = all_dot_products[vector_names[0]][vector_names[2]]  # RepInd ablated by DiM
dim_repind_ablated = all_dot_products[vector_names[2]][vector_names[0]]  # DiM ablated by RepInd

# max between dim_baseline and dim_repind_ablated
diffs = dim_baseline - dim_repind_ablated
print(max(diffs), np.argmax(diffs))

y_min = min(min(dim_baseline), min(orthogonal_baseline), min(repind_baseline),
            min(orthogonal_dim_ablated), min(dim_orthogonal_ablated),
            min(repind_dim_ablated), min(dim_repind_ablated))
y_max = max(max(dim_baseline), max(orthogonal_baseline), max(repind_baseline),
            max(orthogonal_dim_ablated), max(dim_orthogonal_ablated),
            max(repind_dim_ablated), max(dim_repind_ablated))
y_padding = (y_max - y_min) * 0.04
y_min -= y_padding
y_max += y_padding

x = range(model.config.num_hidden_layers)

# %%
# Styling parameters
LINE_WIDTH = 2
MARKER_SIZE = 0  # Increased marker size
ALPHA = 0.8
GRID_ALPHA = 0.6
FONT_SIZE = 10  # Base font size
TITLE_SIZE = 10
LABEL_SIZE = 10
TICK_SIZE = 10
LEGEND_SIZE = 10

# Create second version without first subplot
# figsize was 6,6 for paper
fig3, ((ax2, ax3), (ax4, ax5)) = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=False)

# Ax2: RDO$_\perp$ baseline with DIM ablated
ax2.plot(x, orthogonal_baseline, color='#E74C3C', linewidth=LINE_WIDTH, marker='o',
         markersize=MARKER_SIZE, alpha=ALPHA, label='RDO$_\perp$')
ax2.fill_between(x, orthogonal_baseline - baseline_dot_products[vector_names[1]].std(axis=0),
                 orthogonal_baseline + baseline_dot_products[vector_names[1]].std(axis=0),
                 color='#E74C3C', alpha=0.15)
ax2.plot(x, orthogonal_dim_ablated, color='#E74C3C', linewidth=LINE_WIDTH, marker='s',
         markersize=MARKER_SIZE, alpha=0.5, linestyle='--',
         label='with DIM ab.')
ax2.fill_between(x, orthogonal_dim_ablated - dot_product_stds[vector_names[0]][vector_names[1]],
                 orthogonal_dim_ablated + dot_product_stds[vector_names[0]][vector_names[1]],
                 color='#F5B7B1', alpha=0.25)  # Lighter red for better contrast

# Ax3: DIM baseline with RDO$_\perp$ ablated
ax3.plot(x, dim_baseline, color='#2E86C1', linewidth=LINE_WIDTH, marker='o',
         markersize=MARKER_SIZE, alpha=ALPHA, label='DIM')
ax3.fill_between(x, dim_baseline - baseline_dot_products[vector_names[0]].std(axis=0),
                 dim_baseline + baseline_dot_products[vector_names[0]].std(axis=0),
                 color='#2E86C1', alpha=0.15)
ax3.plot(x, dim_orthogonal_ablated, color='#2E86C1', linewidth=LINE_WIDTH, marker='s',
         markersize=MARKER_SIZE, alpha=0.5, linestyle='--',
         label='with RDO$_\perp$\nablated')
ax3.fill_between(x, dim_orthogonal_ablated - dot_product_stds[vector_names[1]][vector_names[0]],
                 dim_orthogonal_ablated + dot_product_stds[vector_names[1]][vector_names[0]],
                 color='#AED6F1', alpha=0.25)  # Lighter blue for better contrast
# Ax4: RepInd baseline with DIM ablated
ax4.plot(x, repind_baseline, color='#E67E22', linewidth=LINE_WIDTH, marker='o',
         markersize=MARKER_SIZE, alpha=ALPHA, label='RepInd')
ax4.fill_between(x, repind_baseline - baseline_dot_products[vector_names[2]].std(axis=0),
                 repind_baseline + baseline_dot_products[vector_names[2]].std(axis=0),
                 color='#E67E22', alpha=0.15)
ax4.plot(x, repind_dim_ablated, color='#E67E22', linewidth=LINE_WIDTH, marker='s',
         markersize=MARKER_SIZE, alpha=0.5, linestyle='--',
         label='with DIM ablated')
ax4.fill_between(x, repind_dim_ablated - dot_product_stds[vector_names[0]][vector_names[2]],
                 repind_dim_ablated + dot_product_stds[vector_names[0]][vector_names[2]],
                 color='#FAD7A0', alpha=0.25)  # Lighter orange for better contrast
# Ax5: DIM baseline with RepInd ablated
ax5.plot(x, dim_baseline, color='#2E86C1', linewidth=LINE_WIDTH, marker='o',
         markersize=MARKER_SIZE, alpha=ALPHA, label='DIM')
ax5.fill_between(x, dim_baseline - baseline_dot_products[vector_names[0]].std(axis=0),
                 dim_baseline + baseline_dot_products[vector_names[0]].std(axis=0),
                 color='#2E86C1', alpha=0.15)
ax5.plot(x, dim_repind_ablated, color='#2E86C1', linewidth=LINE_WIDTH, marker='s',
         markersize=MARKER_SIZE, alpha=0.5, linestyle='--',
         label='with RepInd\nablated')
ax5.fill_between(x, dim_repind_ablated - dot_product_stds[vector_names[2]][vector_names[0]],
                 dim_repind_ablated + dot_product_stds[vector_names[2]][vector_names[0]],
                 color='#AED6F1', alpha=0.25)  # Lighter blue for better contrast

# Configure all axes with enhanced styling
for ax in [ax2, ax3, ax4, ax5]:
    # Enhanced grid
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    
    # Better legend with larger font
    ax.legend(frameon=True, fancybox=False, framealpha=0.95,
              shadow=False, fontsize=LEGEND_SIZE,
              loc='upper left', bbox_to_anchor=(0.01, 1), ncol=1)
    
    # Labels with larger fonts
    if ax in [ax4, ax5]:
        ax.set_xlabel('Layer', fontsize=LABEL_SIZE, fontweight='bold')
    if ax in [ax2, ax4]:
        ax.set_ylabel('Dot product', fontsize=LABEL_SIZE, fontweight='bold')
    
    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tick parameters with larger font
    ax.tick_params(labelsize=TICK_SIZE, width=0.5)
    ax.set_xticks(range(0, len(x), 5))  # x ticks every 5 layers
    # ax.set_ylim(y_min, y_max + 0.1)
    # ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))  # y ticks every 0.1

# Adjust layout with reduced spacing between columns
plt.tight_layout(rect=[0, 0, 1, 0.85], h_pad=0.3, w_pad=0.3)

# Save second version
os.makedirs('results/plots/crossovereffects', exist_ok=True)
plt.savefig('results/plots/crossovereffects/dot_product_four_plots_fixed.png', dpi=300, bbox_inches='tight')
plt.close()
# %%

import wandb
vector_configs = [
    # {"run_id": "mkmbl6ae", "run_name": "Rep. Ind. Cone"},
    # {"run_id": "kjsv6kv1", "run_name": "Rep. Ind. Cone 2"},
    # {"run_id": "5r398njv", "run_name": "Rep. Ind. Cone 4"},
    # {"run_id": "a6k7w9hg", "run_name": "Rep. Ind. Cone 3"},
    # {"run_id": "f47wr2cf", "run_name": "Rep. Ind. Cone 2"},
    # {"run_id": "6yoyqywn", "run_name": "Rep. Ind. Cone 3"},
    # {"run_id": "xg10zwjk", "run_name": "Rep. Ind. Cone Dim 2"},
    # {"run_id": "4vqlri8m", "run_name": "Rep. Ind. Cone Dim 3"},
    # {"run_id": "tmq6j26o", "run_name": "Rep. Ind. Cone Dim 4"},
    # {"run_id": "5vft9jmn", "run_name": "Rep. Ind. Cone Dim 5"},
    # {"run_id": "ll9feu6z", "run_name": "Rep. Ind. Cone Dim 2 lw 400"},
    # {"run_id": "nsmas9aj", "run_name": "Rep. Ind. Cone Dim 3 lw 400"},
    {"run_id": "3im6w1bn", "run_name": "Rep. Ind. Cone Dim 4"},
    # {"run_id": "zwkjj22m", "run_name": "Rep. Ind. Cone Dim 2 lw 200"},
    # {"run_id": "finrojwc", "run_name": "Rep. Ind. Cone Dim 3 lw 200"},

]
api = wandb.Api()
for vector_config in vector_configs:
    run_id = vector_config["run_id"]
    run = api.run(f"refusal-representations/robust_refusal_subspace/{run_id}")
    artifact = api.artifact(f'refusal-representations/robust_refusal_subspace/trained_vectors_run_{run_id}:v0', type='vector')
    artifact_dir = artifact.download()
    print(f"Artifact dir: {artifact_dir}")
    print(f"Vector config: {vector_config}")
    direction = torch.load(os.path.join(artifact_dir, "lowest_loss_vector.pt"))
    vectors = list(direction.to(model.dtype))


# %%
vector_names = [f"Basis Vector {i+1}" for i in range(len(vectors))]
intervention_vectors = vectors

# intervention_vectors.append((vectors[0] + vectors[1]) / 2)
# intervention_vectors.append((vectors[0] + vectors[2]) / 2)
# intervention_vectors.append((vectors[1] + vectors[2]) / 2)
# vector_names.append("Average of Basis Vectors 1 and 2")
# vector_names.append("Average of Basis Vectors 1 and 3")
# vector_names.append("Average of Basis Vectors 2 and 3")

# %%
# import wandb
# vector_configs = [
#     {"run_id": "y8sfmb74", "run_name": "Rep. Ind."},
# ]
# api = wandb.Api()
# vectors = []
# for vector_config in vector_configs:
#     run_id = vector_config["run_id"]
#     run = api.run(f"refusal-representations/robust_refusal_vector/{run_id}")
#     candidate_idx = run.summary.get("val/candidate_idx")
#     artifact = api.artifact(f'refusal-representations/robust_refusal_vector/trained_vectors_run_{run_id}:v0', type='vector')
#     artifact_dir = artifact.download()
#     print(f"Artifact dir: {artifact_dir}")
#     print(f"Vector config: {vector_config}")
#     if 'Ind' in vector_config["run_name"]:
#         print("Loading lowest loss vector")
#         direction = torch.load(os.path.join(artifact_dir, "lowest_loss_vector.pt"))
#     else:
#         direction = torch.load(os.path.join(artifact_dir, "vector.pt"))[-20+candidate_idx]
#     vectors.append(direction.to(model.dtype))

# vector_names = ["DIM"] + [c["run_name"] for c in vector_configs]
# intervention_vectors = [best_refusal_direction] + vectors

# %%
batch_size = 32

baseline_cosine_sims = {}
baseline_cosine_sims_harmless = {}

for vector_name, vector in zip(vector_names, intervention_vectors):
    # Get similarities for harmful instructions
    cosine_sims, _ = new_get_cosine_similarities(model, harmful_instructions, vector, batch_size=batch_size)
    baseline_cosine_sims[vector_name] = cosine_sims
    
    # # Get similarities for harmless instructions
    cosine_sims, _ = new_get_cosine_similarities(model, harmless_instructions, vector, batch_size=batch_size)
    baseline_cosine_sims_harmless[vector_name] = cosine_sims

# %%
harmful_means = {}
harmful_stds = {}
harmless_means = {}
harmless_stds = {}

for vector_name in vector_names:
    harmful_means[vector_name] = baseline_cosine_sims[vector_name].mean(axis=0)
    harmful_stds[vector_name] = baseline_cosine_sims[vector_name].std(axis=0)
    harmless_means[vector_name] = baseline_cosine_sims_harmless[vector_name].mean(axis=0)
    harmless_stds[vector_name] = baseline_cosine_sims_harmless[vector_name].std(axis=0)

# Create plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle('Baseline Cosine Similarities Between Directions and Mean Activations Across Model Layers')

y_min = min(
    min([min(harmful_means[v] - harmful_stds[v]) for v in vector_names]),
    min([min(harmless_means[v] - harmless_stds[v]) for v in vector_names])
)
y_max = max(
    max([max(harmful_means[v] + harmful_stds[v]) for v in vector_names]), 
    max([max(harmless_means[v] + harmless_stds[v]) for v in vector_names])
)

x = range(model.config.num_hidden_layers)
for i, vector_name in enumerate(vector_names):
    print(vector_name, harmful_means[vector_name])
    color = f'C{i}'
    
    ax1.plot(x, harmful_means[vector_name], label=vector_name, color=color, linestyle='-', marker='o', markersize=4)
    ax1.fill_between(x, harmful_means[vector_name] - harmful_stds[vector_name], 
                     harmful_means[vector_name] + harmful_stds[vector_name], color=color, alpha=0.2)
    
    ax2.plot(x, harmless_means[vector_name], label=vector_name, color=color, linestyle='-', marker='o', markersize=4)
    ax2.fill_between(x, harmless_means[vector_name] - harmless_stds[vector_name],
                     harmless_means[vector_name] + harmless_stds[vector_name], color=color, alpha=0.2)

for ax in [ax1, ax2]:
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)

ax1.set_title('Harmful Instructions')
ax2.set_title('Harmless Instructions')

plt.tight_layout()
plt.show()

# %%
# Compute similarities after intervention
all_cosine_sims = {}
cosine_sim_changes = {}
cosine_sim_stds = {}

for intervention_vector, intervention_name in zip(intervention_vectors, vector_names):
    all_cosine_sims[intervention_name] = {}
    cosine_sim_changes[intervention_name] = {}
    cosine_sim_stds[intervention_name] = {}
    for measure_vector, measure_name in zip(intervention_vectors, vector_names):
        cosine_sims, _ = new_get_cosine_similarities(
            model, harmful_instructions,
            measure_vector=measure_vector,
            intervention_vector=intervention_vector,
            batch_size=batch_size
        )
        all_cosine_sims[intervention_name][measure_name] = cosine_sims.mean(axis=0)
        cosine_sim_changes[intervention_name][measure_name] = (
            cosine_sims.mean(axis=0) - baseline_cosine_sims[measure_name].mean(axis=0)
        )
        cosine_sim_stds[intervention_name][measure_name] = cosine_sims.std(axis=0)

# %%
# Plot cosine similarity changes for all vector pairs
x = range(model.config.num_hidden_layers)
n_vectors = len(vector_names)

# Determine global y-axis limits across all changes and baselines
y_min = min(
    min(cosine_sim_changes[int_name][mea_name].min()
        for int_name in vector_names for mea_name in vector_names),
    min(baseline_cosine_sims[mea_name].mean(axis=0).min()
        for mea_name in vector_names)
)
y_max = max(
    max(cosine_sim_changes[int_name][mea_name].max()
        for int_name in vector_names for mea_name in vector_names),
    max(baseline_cosine_sims[mea_name].mean(axis=0).max()
        for mea_name in vector_names)
)
y_padding = (y_max - y_min) * 0.04
y_min -= y_padding
y_max += y_padding

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=False)
axes = axes.flatten()  # Flatten to make indexing easier

# If we have fewer than 4 vectors, hide the extra subplots
for i in range(n_vectors, 4):
    axes[i].set_visible(False)

for j, measure_name in enumerate(vector_names[:4]):  # Limit to first 4 vectors
    ax = axes[j]
    # Plot baseline for the measured vector
    baseline_mean = baseline_cosine_sims[measure_name].mean(axis=0)
    baseline_std = baseline_cosine_sims[measure_name].std(axis=0)
    ax.plot(x, baseline_mean, color='black', linestyle='--', label='Baseline', linewidth=2, zorder=10)
    # ax.fill_between(x, baseline_mean - baseline_std, baseline_mean + baseline_std, color='gray', alpha=0.3, zorder=9)
    
    # Plot post-intervention curves for all intervention vectors as multiple lines
    for i, intervention_name in enumerate(vector_names):
        color = f'C{i}'
        changes = cosine_sim_changes[intervention_name][measure_name]
        stds = cosine_sim_stds[intervention_name][measure_name]
        post_intervention = baseline_mean + changes
        ax.plot(x, post_intervention, color=color, linestyle='-', marker='o', markersize=4, label=f'Intervention Vector: {intervention_name}')
        # ax.fill_between(x, post_intervention - stds, post_intervention + stds, color=color, alpha=0.2)
    
    # ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, model.config.num_hidden_layers, 5))
    ax.set_title(f'Measure Vector: {measure_name}')
    
    # Add y-label only to the leftmost plots
    if j % 2 == 0:
        ax.set_ylabel('Cosine Similarity')
    
    # Add x-label only to the bottom plots
    if j >= 2:
        ax.set_xlabel('Layer')

# Add a single legend above the plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=1, frameon=False, bbox_to_anchor=(0.5, 1.3))

plt.tight_layout()
plt.savefig('results/plots/crossovereffects/repind_interpolations_new.png', dpi=300, bbox_inches='tight')
plt.show()
# plt.close()

# %%
import wandb
vector_configs = [
    {"run_id": "dlnphp08", "run_name": "RDO$_\perp$"},
    # {"run_id": "3aiv5tbr", "run_name": "RDO$_\perp$"}, # llama3
    # {"run_id": "onxv1cwu", "run_name": "RDO$_\perp$"}, # qwen
]
api = wandb.Api()
vectors = []
for vector_config in vector_configs:
    run_id = vector_config["run_id"]
    run = api.run(f"refusal-representations/robust_refusal_vector/{run_id}")
    candidate_idx = run.summary.get("val/candidate_idx")
    artifact = api.artifact(f'refusal-representations/robust_refusal_vector/trained_vectors_run_{run_id}:v0', type='vector')
    artifact_dir = artifact.download()
    print(f"Artifact dir: {artifact_dir}")
    print(f"Vector config: {vector_config}")
    direction = torch.load(os.path.join(artifact_dir, "vector.pt"))[-20+candidate_idx]
    vectors.append(direction.to(model.dtype))

# %%
vector_names = ["DIM"] + [c["run_name"] for c in vector_configs]
intervention_vectors = [best_refusal_direction] + vectors

# %%
batch_size = 16

baseline_cosine_sims = {}
baseline_cosine_sims_harmless = {}
baseline_dot_products = {}
baseline_dot_products_harmless = {}

for vector_name, vector in zip(vector_names, intervention_vectors):
    # Get similarities for harmful instructions
    cosine_sims, dot_products = new_get_cosine_similarities(model, harmful_instructions, vector, batch_size=batch_size)
    baseline_cosine_sims[vector_name] = cosine_sims
    baseline_dot_products[vector_name] = dot_products
    
    # # Get similarities for harmless instructions
    cosine_sims, dot_products = new_get_cosine_similarities(model, harmless_instructions, vector, batch_size=batch_size)
    baseline_cosine_sims_harmless[vector_name] = cosine_sims
    baseline_dot_products_harmless[vector_name] = dot_products

# %%
# Compute similarities after intervention
all_cosine_sims = {}
cosine_sim_changes = {}
cosine_sim_stds = {}

for intervention_vector, intervention_name in zip(intervention_vectors, vector_names):
    all_cosine_sims[intervention_name] = {}
    cosine_sim_changes[intervention_name] = {}
    cosine_sim_stds[intervention_name] = {}
    for measure_vector, measure_name in zip(intervention_vectors, vector_names):
        cosine_sims, dot_products = new_get_cosine_similarities(model, harmful_instructions, measure_vector=measure_vector, intervention_vector=intervention_vector, batch_size=batch_size)
        all_cosine_sims[intervention_name][measure_name] = cosine_sims.mean(axis=0)
        
        # Calculate change from baseline and std
        cosine_sim_changes[intervention_name][measure_name] = cosine_sims.mean(axis=0) - baseline_cosine_sims[measure_name].mean(axis=0)
        cosine_sim_stds[intervention_name][measure_name] = cosine_sims.std(axis=0)

# %%
# Create figure with more spacing between subplots
# Get min/max across all relevant data for consistent y-axis
dim_baseline = baseline_cosine_sims[vector_names[0]].mean(axis=0)
rdo_baseline = baseline_cosine_sims[vector_names[1]].mean(axis=0)

rdo_dim_ablated = all_cosine_sims[vector_names[0]][vector_names[1]]  # RDO ablated by DIM
dim_rdo_ablated = all_cosine_sims[vector_names[1]][vector_names[0]]  # DIM ablated by RDO

print(dim_baseline - dim_rdo_ablated)

y_min = min(
    min(dim_baseline), min(rdo_baseline),
    min(rdo_dim_ablated), min(dim_rdo_ablated)
)
y_max = max(
    max(dim_baseline), max(rdo_baseline),
    max(rdo_dim_ablated), max(dim_rdo_ablated)
)
y_padding = (y_max - y_min) * 0.04
y_min -= y_padding
y_max += y_padding

x = range(model.config.num_hidden_layers)

# %%
# Styling parameters
MARKER_SIZE = 0  # Increased marker size
LINE_WIDTH = 2

# Create figure with 1 row and 2 columns
fig3, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

# Left: RDO_⊥ baseline with DIM ablated
axes[0].plot(x, rdo_baseline, color=colors[1], marker='o',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='RDO$_\perp$')
axes[0].fill_between(x, rdo_baseline - baseline_cosine_sims[vector_names[1]].std(axis=0),
                 rdo_baseline + baseline_cosine_sims[vector_names[1]].std(axis=0),
                 color=colors[1], alpha=0.15)
axes[0].plot(x, rdo_dim_ablated, color=colors[1], marker='s',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle='--',
         label='RDO$_\perp$ with DIM ablation')
axes[0].fill_between(x, rdo_dim_ablated - cosine_sim_stds[vector_names[0]][vector_names[1]],
                 rdo_dim_ablated + cosine_sim_stds[vector_names[0]][vector_names[1]],
                 color=colors[1], alpha=0.25)

# Right: DIM baseline with RDO_⊥ ablated
axes[1].plot(x, dim_baseline, color=colors[0], marker='o',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='DIM')
axes[1].fill_between(x, dim_baseline - baseline_cosine_sims[vector_names[0]].std(axis=0),
                 dim_baseline + baseline_cosine_sims[vector_names[0]].std(axis=0),
                 color=colors[0], alpha=0.15)
axes[1].plot(x, dim_rdo_ablated, color=colors[0], marker='s',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle='--',
         label='DIM with RDO$_\perp$ ablation')
axes[1].fill_between(x, dim_rdo_ablated - cosine_sim_stds[vector_names[1]][vector_names[0]],
                 dim_rdo_ablated + cosine_sim_stds[vector_names[1]][vector_names[0]],
                 color=colors[0], alpha=0.25)

# Configure all axes with enhanced styling
for ax in axes:
    # Enhanced grid
    ax.grid(True, linestyle='--')
    
    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tick parameters with larger font
    ax.tick_params(width=0.5)
    ax.set_xticks(range(0, len(x), 5))  # x ticks every 5 layers
    ax.set_ylim(y_min, y_max + 0.1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))  # y ticks every 0.1

# Add labels to the leftmost and bottom plots
axes[0].set_ylabel('Cosine similarity')
for ax in axes:
    ax.set_xlabel('Layer')

# Create a single legend for the entire figure
handles = [
    plt.Line2D([0], [0], color=colors[1], marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='RDO$_\perp$ baseline'),
    plt.Line2D([0], [0], color=colors[1], marker='s', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle='--', label='RDO$_\perp$ with DIM ablated'),
    plt.Line2D([0], [0], color=colors[0], marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='DIM baseline'),
    plt.Line2D([0], [0], color=colors[0], marker='s', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle='--', label='DIM with RDO$_\perp$ ablated')
]
# Add text above the figure instead of a title
# plt.figtext(0.5, 0.99, "Gemma 2 2B", ha='center', va='top', fontsize=13)
plt.figtext(0.5, 0.99, "Gemma 2 2B", ha='center', va='top', fontsize=13)
fig3.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=2, frameon=True, fancybox=False, framealpha=0.95)

# Adjust layout with increased vertical spacing between rows
plt.tight_layout(rect=[0, 0, 1, 0.94])

os.makedirs('results/plots/crossovereffects', exist_ok=True)
plt.savefig('results/plots/crossovereffects/orthogonal_ablation_gemma_2_2b.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
