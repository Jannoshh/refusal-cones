"""
Steering module - Activation steering implementations.

Main components:
- ACE (Affine Concept Editing) from Marshall et al. (2024)
"""

from .ace import (
    ACEConfig,
    ACEVectors,
    ACESteerer,
    ace_transform,
    collect_activations,
    setup_ace_steering,
)

__all__ = [
    'ACEConfig',
    'ACEVectors',
    'ACESteerer',
    'ace_transform',
    'collect_activations',
    'setup_ace_steering',
]
