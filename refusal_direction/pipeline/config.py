
import os

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    model_alias: str
    model_path: str
    n_test: int = 128
    splits: str = "cb"
    sample: bool = False
    filter_train: bool = True
    filter_val: bool = True
    # evaluation_datasets: Tuple[str] = ("jailbreakbench", "strongreject", "sorrybench", "xstest")
    evaluation_datasets: Tuple[str] = ("sorrybench", "xstest")
    evaluate_ablation: bool = True
    evaluate_actadd: bool = True
    evaluate_harmless: bool = False
    evaluate_loss: bool = False
    
    subspace_n_samples: int = 64
    max_new_tokens: int = 512
    completions_batch_size: int = 128

    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching", "strongreject")
    refusal_eval_methodologies: Tuple[str] = ("substring_matching",)
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 2048

    def artifact_path(self) -> str:
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../results/refusal_dir", self.model_alias)
        return os.path.join("/ceph/hdd/students/elsj/paper_results/dim_overrefusal", self.model_alias)
    
    def directopt_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../results/directopt", self.model_alias)