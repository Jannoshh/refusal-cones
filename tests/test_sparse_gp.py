#!/usr/bin/env python3
import torch

from src.discovery.adaptive_geometry_discovery import AdaptiveSparseGP


def test_adaptive_sparse_gp_fit_and_predict_shapes():
    torch.manual_seed(0)
    n, d = 20, 6
    # Unit-norm samples to mimic hypersphere directions
    V = torch.randn(n, d)
    V = V / V.norm(dim=1, keepdim=True)
    R = torch.sigmoid(torch.randn(n))  # dummy scores in [0,1]

    gp = AdaptiveSparseGP(
        kernel_type="rbf",
        lengthscale=0.5,
        num_inducing=8,
        train_steps=3,
        max_train_points=32,
    )
    gp.fit(V, R)

    V_test = torch.randn(5, d)
    V_test = V_test / V_test.norm(dim=1, keepdim=True)
    mean, std = gp.predict(V_test)

    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert (std > 0).all()


def test_adaptive_sparse_gp_respects_lengthscale():
    torch.manual_seed(1)
    V = torch.randn(10, 4)
    V = V / V.norm(dim=1, keepdim=True)
    R = torch.linspace(0, 1, 10)

    gp_small = AdaptiveSparseGP(lengthscale=0.1, num_inducing=6, train_steps=2)
    gp_small.fit(V, R)
    _, std_small = gp_small.predict(V[:2])

    gp_large = AdaptiveSparseGP(lengthscale=2.0, num_inducing=6, train_steps=2)
    gp_large.fit(V, R)
    _, std_large = gp_large.predict(V[:2])

    # Larger lengthscale should smooth more and typically reduce predictive variance
    assert torch.all(std_large <= std_small + 1e-6)
