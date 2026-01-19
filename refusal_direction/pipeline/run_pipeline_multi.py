import torch
import random
import json
import os
import argparse

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_train = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)[:len(harmful_train)]
    harmful_val = load_dataset_split(harmtype='harmful', split='val', instructions_only=True)
    harmless_val = load_dataset_split(harmtype='harmless', split='val', instructions_only=True)[:len(harmful_val)]
    return harmful_train, harmless_train, harmful_val, harmless_val

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        print("Filtering train dataset")
        print(f"Number of harmful examples: {len(harmful_train)}")
        print(f"Number of harmless examples: {len(harmless_train)}")
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)

        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        print(len([score for score in harmful_train_scores.tolist() if score > 0]))
        print(len([score for score in harmless_train_scores.tolist() if score < 0]))
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)[:len(harmful_train)]
        print(f"Filtered {len(harmful_train)} harmful examples and {len(harmless_train)} harmless examples")

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
    
    return harmful_train, harmless_train, harmful_val, harmless_val

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

    return mean_diffs

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction")
    )

    with open(f'{cfg.artifact_path()}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction.pt')

    return pos, layer, direction

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens, batch_size=cfg.completions_batch_size)
    
    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)

def run_pipeline(model_path):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)

    # 2. Select the most effective refusal direction
    candidate_directions = torch.load(f"../results/refusal_dir/{model_alias}/generate_directions/mean_diffs.pt")
    direction_evaluations = json.load(open(f"../results/refusal_dir/{model_alias}/select_direction/direction_evaluations_filtered.json"))
    for n in range(1, min(5, len(direction_evaluations))):
        directions = []
        n_directions = n
        for i in range(n_directions):
            best_config = direction_evaluations[i]
            best_layer = best_config["layer"]
            best_token = best_config["position"]
            direction = candidate_directions[best_token, best_layer]
            directions.append(direction)
    
    import wandb
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
    directions_list = []
    for run_name in run_names:
        for run in matching_runs:
            if run_name == run.name:
                file_path = ""
                file_path = "direction.pt"
                download_dir = os.path.join("wandb_downloads", run.group, run.name)
                os.makedirs(download_dir, exist_ok=True)
                run.file(file_path).download(root=download_dir, exist_ok=True)
                file_path = os.path.join(download_dir, file_path)
                direction = torch.load(file_path).squeeze()
                directions_list.append(direction)
    best_refusal_direction = torch.load("../results/refusal_dir/gemma-2-2b-it/direction.pt").squeeze()
    # directions_list.insert(0, best_refusal_direction)
    print(len(directions_list))

    max_n = len(directions_list)
    for n in range(1, max_n+1):
        # directions = torch.load(f"../vectors/repind_4.pt")
        # print(directions.shape)
        # Reorder directions to move the middle vector to the end
        # directions_list = list(directions)
        # if len(directions_list) >= 3:
        #     # Store the middle vector
        #     middle_vector = directions_list[1]
        #     # Remove the middle vector
        #     directions_list.pop(1)
        #     # Add the middle vector to the end
        #     directions_list.append(middle_vector)
        # Take only the first n vectors
        directions = directions_list[:n]
        
        # Get hooks for all directions
        ablation_fwd_pre_hooks = []
        ablation_fwd_hooks = []
        for direction in directions:
            pre_hooks, hooks = get_all_direction_ablation_hooks(model_base, direction)
            ablation_fwd_pre_hooks.extend(pre_hooks)
            ablation_fwd_hooks.extend(hooks)

        # 3a. Generate and save completions on harmful evaluation datasets

        name_postfix = f"top_{n}"
        for dataset_name in cfg.evaluation_datasets:
            generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, f'ablation{name_postfix}', dataset_name)

    # 3b. Evaluate completions and save results on harmful evaluation datasets
    # for n in range(1, min(6, len(direction_evaluations))):
    for n in range(1, max_n+1):
        name_postfix = f"top_{n}"
        for dataset_name in cfg.evaluation_datasets:
            evaluate_completions_and_save_results_for_dataset(cfg, f'ablation{name_postfix}', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path)
