
import torch
import os
import wandb

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.select_direction import select_direction_directopt, select_subspace_direction_directopt, select_subspace_direction_directopt2, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

def run_pipeline():
    """Run the full pipeline."""
    model_path = "google/gemma-2-2b-it"
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)

    # entity = "refusal-representations"
    # project = f"{entity}/robust_refusal_vector"
    # group = "sb_data_with_retain_gemma-2-2b-it"

    # api = wandb.Api()
    # runs = api.runs(project, {"group": group})
    # run_name = "run_5"

    # for run in runs:
    #     if run.group == group and run.name == run_name:
    #         break
    # else:
    #     print(f"No runs found for {group} {run_name} {project}")
    #     import sys
    #     sys.exit(1)
    # # Download file from the run's files
    # print(f"Found run {run.id} with group {run.group} and name {run.name}")
    # file_path = "direction.pt"
    # download_dir = os.path.join("wandb_downloads", run.group, run.name)
    # os.environ["WANDB_DIR"] = "/nfs/homedirs/elsj/adversarial/"
    # api = wandb.Api()

    file_path = "samples.pt"
    base_dir = '/ceph/hdd/students/elsj/paper_results'
    download_dir = os.path.join(base_dir, "subspace_samples_eval", "join_subspace_gemma-2-2b-it", "dim_4")

    # os.makedirs(download_dir, exist_ok=True)
    # run.file(file_path).download(root=download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, file_path)
    samples = torch.load(file_path)
    
    results_dir = f"../results/refusal_scores/gemma-2-2b-it/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    from pipeline.submodules.select_direction import get_refusal_scores

    refusal_scores = []
    jailbreakbench_dataset = load_dataset("jailbreakbench", instructions_only=True)
    from tqdm import tqdm
    for direction in tqdm(samples[:64]):
        ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)

        scores = get_refusal_scores(model_base.model, jailbreakbench_dataset, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=ablation_fwd_pre_hooks, fwd_hooks=ablation_fwd_hooks, batch_size=32)
        refusal_scores.append(scores)
    save_path = os.path.join(results_dir, "refusal_scores.pt")
    torch.save(torch.stack(refusal_scores), save_path)
    print(f"Saved refusal scores to {save_path}")
    print(torch.stack(refusal_scores).shape)


if __name__ == "__main__":
    run_pipeline()

