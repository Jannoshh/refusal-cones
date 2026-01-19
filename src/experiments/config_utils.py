import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import yaml


def load_experiment_config(path: str = "experiments/configs/default.yaml") -> Dict[str, Any]:
    """Load experiment config YAML if present, else return empty dict."""
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _get(config: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur = config
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def merge_cli_with_config(args, config: Dict[str, Any]) -> SimpleNamespace:
    """
    Merge a parsed argparse namespace with config dict.

    CLI values take precedence; config fills missing keys only.
    """
    return SimpleNamespace(
        model=args.model,
        seed=args.seed,
        output_dir=args.output_dir,
        use_sparse_gp=args.use_sparse_gp
        if hasattr(args, "use_sparse_gp")
        else _get(config, ["discovery", "use_sparse_gp"], False),
        kernel=args.kernel if hasattr(args, "kernel") else _get(config, ["discovery", "kernel"], "rbf"),
        lengthscale=args.lengthscale
        if hasattr(args, "lengthscale")
        else _get(config, ["discovery", "lengthscale"], 0.3),
        num_inducing=args.num_inducing
        if hasattr(args, "num_inducing")
        else _get(config, ["discovery", "num_inducing"], 64),
        sparse_steps=args.sparse_steps
        if hasattr(args, "sparse_steps")
        else _get(config, ["discovery", "sparse_steps"], 10),
        sparse_lr=args.sparse_lr if hasattr(args, "sparse_lr") else _get(config, ["discovery", "sparse_lr"], 0.05),
    )


def save_run_metadata(output_dir: Path, metadata: Dict[str, Any], filename: str = "metadata.jsonl") -> None:
    """Append a JSONL record with run metadata/results into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    def convert(obj):
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
        except ImportError:
            pass
        if isinstance(obj, Path):
            return str(obj)
        return obj

    record = json.loads(json.dumps(metadata, default=convert))
    with open(path, "a") as f:
        f.write(json.dumps(record) + os.linesep)
