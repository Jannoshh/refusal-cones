#!/usr/bin/env python3
import torch

from src.training.trainers.per_layer_training import compute_ce_loss


def test_compute_ce_loss_handles_mismatched_lengths():
    torch.manual_seed(0)
    batch, seq_len, vocab = 2, 4, 6
    logits = torch.randn(batch, seq_len, vocab, requires_grad=True)

    # Labels shorter than logits sequence length
    labels = torch.tensor([[1, 2], [3, 4]])
    loss = compute_ce_loss(logits, labels)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None


def test_compute_ce_loss_broadcast_single_label_sequence():
    torch.manual_seed(1)
    batch, seq_len, vocab = 3, 5, 7
    logits = torch.randn(batch, seq_len, vocab, requires_grad=True)

    # Single label sequence should broadcast across batch
    labels = torch.tensor([1, 0, 2])
    loss = compute_ce_loss(logits, labels)
    loss.backward()

    assert loss.item() == loss.item()  # finite
    assert logits.grad.shape == logits.shape
