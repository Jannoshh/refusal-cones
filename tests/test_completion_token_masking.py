#!/usr/bin/env python3
"""Tests for completion masking with special-token handling."""

from types import SimpleNamespace

import torch

from src.discovery.pareto_boundary_discovery import MultiObjectiveScorer


class DummyBatch(dict):
    """Dict-like batch that supports .to(device)."""

    def to(self, device):
        for key, value in self.items():
            if torch.is_tensor(value):
                self[key] = value.to(device)
        return self


class DummyTokenizer:
    """Minimal tokenizer with optional BOS/EOS insertion by default."""

    def __init__(self, add_special_tokens_default: bool = True):
        self.add_special_tokens_default = add_special_tokens_default
        self.bos_token_id = 101
        self.eos_token_id = 102
        self.pad_token_id = 0

    def _encode(self, text: str, add_special_tokens: bool) -> list[int]:
        token_ids = [10 + (ord(ch) % 50) for ch in text]
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        return token_ids

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens_default

        input_ids = [self._encode(text, add_special_tokens) for text in texts]

        if truncation and max_length is not None:
            input_ids = [ids[:max_length] for ids in input_ids]

        if padding:
            max_len = max((len(ids) for ids in input_ids), default=0)
            padded = []
            attention_mask = []
            for ids in input_ids:
                pad_len = max_len - len(ids)
                padded.append(ids + [self.pad_token_id] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
            input_ids = padded
        else:
            attention_mask = [[1] * len(ids) for ids in input_ids]

        if return_tensors == "pt":
            return DummyBatch(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
            )

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummyModel(torch.nn.Module):
    """Produces zero logits for any input shape."""

    def __init__(self, vocab_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        return SimpleNamespace(logits=logits)


def test_completion_labels_align_with_prompt_tokens():
    tokenizer = DummyTokenizer(add_special_tokens_default=True)
    model = DummyModel()

    harmful_prompts = ["Hi", "Hello"]
    harmful_completions = ["abc", "defg"]

    scorer = MultiObjectiveScorer(
        model=model,
        tokenizer=tokenizer,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmful_prompts,
        refusal_toks=torch.tensor([1], dtype=torch.long),
        device="cpu",
        harmful_completions=harmful_completions,
        harmless_completions=harmful_completions,
        n_score_tokens=2,
        n_kl_tokens=2,
    )

    full_ids = scorer.harmful_full_inputs["input_ids"]
    full_mask = scorer.harmful_full_inputs["attention_mask"]

    for i, prompt in enumerate(harmful_prompts):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][0]
        seq_len = int(full_mask[i].sum().item())
        seq_ids = full_ids[i, :seq_len].tolist()

        # Full text should start with the prompt tokens (no BOS/EOS added).
        assert seq_ids[: len(prompt_ids)] == prompt_ids
        assert tokenizer.bos_token_id not in seq_ids
        assert tokenizer.eos_token_id not in seq_ids

        prompt_len = len(prompt_ids)
        labels = scorer.completion_labels[i]

        # Prompt tokens are masked.
        assert torch.all(labels[:prompt_len] == -100)

        # Only the first n_score_tokens of the completion are scored.
        score_end = min(prompt_len + scorer.n_score_tokens, seq_len)
        assert torch.all(labels[prompt_len:score_end] != -100)
        assert torch.all(labels[score_end:seq_len] == -100)


def test_harmless_completion_mask_respects_prompt_boundary():
    tokenizer = DummyTokenizer(add_special_tokens_default=True)
    model = DummyModel()

    harmless_prompts = ["AB", "WXYZ"]
    harmless_completions = ["klm", "nopqr"]

    scorer = MultiObjectiveScorer(
        model=model,
        tokenizer=tokenizer,
        harmful_prompts=harmless_prompts,
        harmless_prompts=harmless_prompts,
        refusal_toks=torch.tensor([1], dtype=torch.long),
        device="cpu",
        harmful_completions=harmless_completions,
        harmless_completions=harmless_completions,
        n_score_tokens=2,
        n_kl_tokens=2,
    )

    full_ids = scorer.harmless_full_inputs["input_ids"]
    full_mask = scorer.harmless_full_inputs["attention_mask"]

    for i, prompt in enumerate(harmless_prompts):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][0]
        seq_len = int(full_mask[i].sum().item())
        seq_ids = full_ids[i, :seq_len].tolist()

        # Full text should start with the prompt tokens (no BOS/EOS added).
        assert seq_ids[: len(prompt_ids)] == prompt_ids
        assert tokenizer.bos_token_id not in seq_ids
        assert tokenizer.eos_token_id not in seq_ids

        prompt_len = len(prompt_ids)
        mask = scorer.harmless_completion_mask[i]

        # Completion mask should start exactly after the prompt tokens.
        assert not mask[:prompt_len].any()
        expected_end = min(prompt_len + scorer.n_kl_tokens, seq_len)
        assert mask[prompt_len:expected_end].all()
        assert not mask[expected_end:seq_len].any()
