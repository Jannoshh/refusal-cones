#!/usr/bin/env python3
import torch

from src.discovery.gradient_discovery import (
    GradientGeometryDiscovery,
    GradientDiscoveryConfig,
)


def test_gradient_discovery_runs_on_mock_measurement():
    torch.manual_seed(0)
    n_layers, hidden_dim = 2, 3

    # True direction to recover
    v_true = torch.randn(n_layers, hidden_dim)
    v_true = v_true / v_true.norm(dim=1, keepdim=True)

    def measure_with_grad(v: torch.Tensor):
        v = v.clone().detach().requires_grad_(True)
        align = (v * v_true).sum()
        R = torch.sigmoid(align)  # scalar score
        R.backward()
        grad = v.grad.clone()
        return R.item(), grad

    v_init = v_true + 0.1 * torch.randn_like(v_true)
    v_init = v_init / v_init.norm(dim=1, keepdim=True)

    config = GradientDiscoveryConfig(
        n_gradient_steps=3,
        n_local_iterations=0,  # skip heavy sampling for test
        enable_global_search=False,
    )

    discovery = GradientGeometryDiscovery(
        measure_refusal_with_grad=measure_with_grad,
        v_init=v_init,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        config=config,
    )

    results = discovery.discover()

    assert len(results["modes"]) >= 1
    assert results["R_observed"].numel() == len(results["V_observed"])
    # Modes should stay normalized
    for mode in results["modes"]:
        norms = mode.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
