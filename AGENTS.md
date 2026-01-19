# Refusal Cones - Codex Notes

This repo enables:
1. Discover refusal geometry (gradient + GP, optional sparse GP with inducing points)
2. Train refusal vectors/adapters (PEFT-based RDO, per-layer vectors)
3. Run experiments/ablations quickly with configurable kernels and data loaders

## Latest updates
- Discovery supports configurable kernels and optional adaptive sparse GP (`use_sparse_gp`, `kernel_type`, `kernel_lengthscale`, `num_inducing`, `sparse_train_steps`, `sparse_lr`, `sparse_max_train_points`).
- E1 efficiency script exposes flags for the above; run with `python experiments/e1_discovery/run_efficiency.py --help`.
- Added fast CPU-only tests for discovery, sparse GP, and loss alignment; run with `pytest -q` after activating the venv.

## Quick commands
- Activate env: `source .venv/bin/activate`
- Run tests: `pytest -q`
- Discovery demo (small model): `python experiments/e1_discovery/run_efficiency.py --model gemma-2-2b --use_sparse_gp`

## Notes for Codex
- Keep vectors normalized (unit sphere); kernels operate on flattened, normalized vectors.
- When using sparse GP, inducing points are learnable; jitter is auto-boosted for stability.
- Training/evaluation scripts expect GPU by default; for CPU tests use the lightweight unit tests only.
