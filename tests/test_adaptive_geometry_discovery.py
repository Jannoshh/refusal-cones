#!/usr/bin/env python3
import torch

from src.discovery.adaptive_geometry_discovery import (
    RefusalGeometryDiscovery,
    GeometryConfig,
)


def test_adaptive_geometry_discovery_sparse_runs():
    torch.manual_seed(0)
    n_layers, hidden_dim = 2, 3
    # Hidden true direction
    v_true = torch.randn(n_layers, hidden_dim)
    v_true = v_true / v_true.norm(dim=1, keepdim=True)

    def measure(v: torch.Tensor) -> float:
        v = v / v.norm(dim=1, keepdim=True)
        return torch.sigmoid((v * v_true).sum()).item()

    config = GeometryConfig(
        n_init_random=4,
        n_iterations=1,
        n_candidates=8,
        acquisition_type="ucb",
        beta=1.5,
        use_sparse_gp=True,
        num_inducing=4,
        sparse_train_steps=2,
        kernel_lengthscale=0.5,
    )

    discovery = RefusalGeometryDiscovery(
        measure_refusal_fn=measure,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
    )

    results = discovery.discover()

    assert "geometry" in results
    assert results["geometry"].get("n_modes", 0) >= 1
    assert results["R_observed"].numel() == len(results["V_observed"])
