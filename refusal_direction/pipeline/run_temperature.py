import torch
import random
import json
import os
import argparse
import wandb

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction_directopt, select_subspace_direction_directopt, select_subspace_direction_directopt2, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--temperature', type=float, required=True, help='Temperature for sampling')
    parser.add_argument('--n_completions', type=int, required=True, help='Number of completions to generate')
    return parser.parse_args()

def generate_and_save_completions_for_dataset(cfg, download_dir, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None, n_completions=5, temperature=0.7):
    """Generate and save completions for a dataset with temperature sampling."""
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    if not os.path.exists(os.path.join(download_dir, 'completions')):
        os.makedirs(os.path.join(download_dir, 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    all_completions = []
    for i in range(n_completions):
        completions = model_base.generate_completions(
            dataset, 
            fwd_pre_hooks=fwd_pre_hooks, 
            fwd_hooks=fwd_hooks,
            max_new_tokens=cfg.max_new_tokens, 
            batch_size=cfg.completions_batch_size,
            temperature=temperature
        )
        all_completions.append(completions)
    
    # Save each completion set separately
    for i, completions in enumerate(all_completions):
        with open(os.path.join(download_dir, 'completions', f'{dataset_name}_{intervention_label}_temp{temperature}_{i+1}_completions.json'), "w") as f:
            json.dump(completions, f, indent=4)

def evaluate_completions_and_save_results_for_dataset(cfg, download_dir, intervention_label, dataset_name, eval_methodologies, n_completions=5, temperature=1):
    """Evaluate completions and save results for a dataset."""
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    if not os.path.exists(os.path.join(download_dir, 'completions')):
        os.makedirs(os.path.join(download_dir, 'completions'))
        
    all_evaluations = []
    
    for i in range(n_completions):
        completion_file = os.path.join(download_dir, 'completions', f'{dataset_name}_{intervention_label}_temp{temperature}_{i+1}_completions.json')
        with open(completion_file, 'r') as f:
            completions = json.load(f)

        evaluation = evaluate_jailbreak(
            completions=completions,
            methodologies=eval_methodologies,
            evaluation_path=os.path.join(download_dir, "completions", f"{dataset_name}_{intervention_label}_temp{temperature}_{i+1}_evaluations.json"),
        )
        all_evaluations.append(evaluation)

        with open(os.path.join(download_dir, "completions", f"{dataset_name}_{intervention_label}_temp{temperature}_{i+1}_evaluations.json"), "w") as f:
            json.dump(evaluation, f, indent=4)

def run_pipeline(temperature, n_completions):
    """Run the full pipeline."""
    model_path = "google/gemma-2-2b-it"
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)

    entity = "refusal-representations"
    project = f"{entity}/robust_refusal_vector"
    group = "sb_data_with_retain_gemma-2-2b-it"

    api = wandb.Api()
    runs = api.runs(project, {"group": group})
    run_name = "run_5"

    for run in runs:
        if run.group == group and run.name == run_name:
            break
    else:
        print(f"No runs found for {group} {run_name} {project}")
        import sys
        sys.exit(1)
    # Download file from the run's files
    print(f"Found run {run.id} with group {run.group} and name {run.name}")
    file_path = "direction.pt"
    download_dir = os.path.join("wandb_downloads", run.group, run.name)
    os.makedirs(download_dir, exist_ok=True)
    run.file(file_path).download(root=download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, file_path)
    direction = torch.load(file_path).cuda()
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)

    results_dir = f"results/temperature_completions/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    generate_and_save_completions_for_dataset(
        cfg, results_dir, model_base, ablation_fwd_pre_hooks, 
        ablation_fwd_hooks, "ablation", "jailbreakbench", 
        n_completions=n_completions, temperature=temperature
    )

    evaluate_completions_and_save_results_for_dataset(
        cfg, results_dir, "ablation", "jailbreakbench", 
        eval_methodologies=cfg.jailbreak_eval_methodologies,
        n_completions=n_completions, temperature=temperature
    )

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args.temperature, args.n_completions)

