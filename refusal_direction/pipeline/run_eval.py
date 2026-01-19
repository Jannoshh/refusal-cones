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
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--wandb_project', type=str, default="robust_refusal_vector", help='wandb project')
    parser.add_argument('--wandb_group', type=str, required=True, help='wandb group')
    parser.add_argument('--wandb_run', type=str, required=True, help='wandb run')
    parser.add_argument('--lowest_loss_vector', action='store_true', help='Use the lowest loss vector')
    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    if cfg.sample:
        harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
        harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
        harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
        harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    else:
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

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, add_layer):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(wandb.run.dir, 'select_direction')):
        os.makedirs(os.path.join(wandb.run.dir, 'select_direction'))

    is_subspace = len(candidate_directions.shape) > 2
    if is_subspace:
        candidate_idx, direction = select_subspace_direction_directopt2(
            model_base,
            harmful_val,
            harmless_val,
            candidate_directions,
            artifact_dir=os.path.join(wandb.run.dir, "select_direction"),
            add_layer=add_layer,
            n_samples=cfg.subspace_n_samples
        )
    else:
        kl_threshold = 0.1 if wandb.config.get("use_retain_loss", False) and not "kl_ablation" in wandb.run.group else 9999
        if candidate_directions.shape[0] > 1:
            candidate_idx, direction = select_direction_directopt(
                model_base,
                harmful_val,
                harmless_val,
                candidate_directions,
                artifact_dir=os.path.join(wandb.run.dir, "select_direction"),
                add_layer=add_layer,
                kl_threshold=kl_threshold
            )
        else:
            candidate_idx = 0
            direction = candidate_directions[0]

    with open(f'{wandb.run.dir}/direction_metadata.json', "w") as f:
        json.dump({"candidate_idx": candidate_idx}, f, indent=4)

    torch.save(direction, f'{wandb.run.dir}/direction.pt')
    wandb.save(f'direction.pt')
    wandb.save(f'direction_metadata.json')
    return candidate_idx, direction

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(wandb.run.dir, 'completions')):
        os.makedirs(os.path.join(wandb.run.dir, 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens, batch_size=cfg.completions_batch_size)
    
    with open(f'{wandb.run.dir}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)
    wandb.save(f'{wandb.run.dir}/completions/{dataset_name}_{intervention_label}_completions.json')

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(wandb.run.dir, f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(wandb.run.dir, "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{wandb.run.dir}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)
    wandb.save(f'{wandb.run.dir}/completions/{dataset_name}_{intervention_label}_evaluations.json')

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(wandb.run.dir, 'loss_evals')):
        os.makedirs(os.path.join(wandb.run.dir, 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{wandb.run.dir}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)
    wandb.save(f'{wandb.run.dir}/loss_evals/{intervention_label}_loss_eval.json')

def run_pipeline(model_path, wandb_project, wandb_group, wandb_run, lowest_loss_vector):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    # Resume the existing wandb run
    entity = "refusal-representations"
    project_name = wandb_project
    group_name = wandb_group
    run_name = wandb_run
    os.environ["WANDB_DIR"] = "/nfs/homedirs/elsj/adversarial/"
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project_name}", filters={"group": group_name})
    matching_runs = []
    for run in runs:
        if run.group == group_name and run.name == run_name:
            matching_runs.append(run)
    
    if not matching_runs:
        raise ValueError(f"No run found with group '{group_name}' and name '{run_name}' in project '{project_name}'.")
    
    # Sort by created_at timestamp and take the most recent
    newest_run = max(matching_runs, key=lambda x: x.created_at)
    print(f"Resuming newest run {newest_run.id} (created at {newest_run.created_at})")
    run = wandb.init(project=project_name, id=newest_run.id, resume="allow")

    model_base = construct_model_base(cfg.model_path)

    # if "subspace" in project_name:
    #     is_subspace = True
    #     artifact_name = f"trained_subspaces_run_{wandb.run.id}:v0"
    #     filename = "subspaces.pt"
    #     if lowest_loss_vector:
    #         filename = "lowest_loss_subspace.pt"
    # else:
    is_subspace = "subspace" in project_name
    artifact_name = f"trained_vectors_run_{wandb.run.id}:v0"
    filename = "vectors.pt"
    if lowest_loss_vector:
        filename = "lowest_loss_vector.pt"
    artifact = run.use_artifact(f"{entity}/{project_name}/{artifact_name}")
    artifact_dir = artifact.download()
    print(f"Artifact dir: {artifact_dir}")
    directions = torch.load(os.path.join(artifact_dir, filename))
    directions = torch.stack(directions).cuda()[-20:].squeeze() if not lowest_loss_vector else directions.squeeze().unsqueeze(0).cuda()
    print(directions.shape)

    add_layer = wandb.config["add_layer"]
    alpha = wandb.config["alpha"]
    
    candidate_directions = alpha * directions
    print(candidate_directions.shape)

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)

    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)


    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    if not is_subspace:
        candidate_idx, direction = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, add_layer=add_layer)
        eval_vectors = [{'vector': direction, 'name_postfix': ''}]
    else:
        candidate_idx, direction = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, add_layer=add_layer)
        subspace_dim = len(direction)
        print(f"Subspace dim: {subspace_dim}")
        eval_vectors = []
        for i in range(subspace_dim):
            eval_vectors.append({'vector': direction[i], 'name_postfix': f'_basis_{i+1}'})

    for e in eval_vectors:
        direction = e['vector']
        name_postfix = e['name_postfix']
        ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
        actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[add_layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []

        # 3a. Generate and save completions on harmful evaluation datasets
        for dataset_name in cfg.evaluation_datasets:
            if cfg.evaluate_ablation:
                generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, f'ablation{name_postfix}', dataset_name)
            if cfg.evaluate_actadd:
                generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, f'actadd{name_postfix}', dataset_name)

        # 4a. Generate and save completions on harmless evaluation dataset
        if cfg.evaluate_harmless:
            print("Evaluating on harmless instructions")

            actadd_refusal_pre_hooks, actadd_refusal_hooks = [(model_base.model_block_modules[add_layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))], []
            generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, f'actadd{name_postfix}', 'harmless', dataset=harmless_test)

        if cfg.evaluate_loss:
            # 5. Evaluate loss on harmless datasets
            evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, f'ablation{name_postfix}')
            evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, f'actadd{name_postfix}')

    for e in eval_vectors:
        print("Evaluating completions")
        name_postfix = e['name_postfix']
        model_base.del_model()
        torch.cuda.empty_cache()

        for dataset_name in cfg.evaluation_datasets:
            if cfg.evaluate_ablation:
                evaluate_completions_and_save_results_for_dataset(cfg, f'ablation{name_postfix}', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
            if cfg.evaluate_actadd:
                evaluate_completions_and_save_results_for_dataset(cfg, f'actadd{name_postfix}', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

        if cfg.evaluate_harmless:
            evaluate_completions_and_save_results_for_dataset(cfg, f'actadd{name_postfix}', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, wandb_project=args.wandb_project, wandb_group=args.wandb_group, wandb_run=args.wandb_run, lowest_loss_vector=args.lowest_loss_vector)

