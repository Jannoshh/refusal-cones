"""Adapter implementations for refusal vector training.

All adapter functionality is in unified_rdo_adapter.py which implements
ACE (Affine Concept Editing) from "Refusal in LLMs is an Affine Function".
"""

from .unified_rdo_adapter import (
    # Base config
    ProjectionConfig,
    # Unified RDO
    UnifiedRDOConfig,
    UnifiedRDOLayer,
    RankKUnifiedLayer,
    UnifiedRDOModel,
    get_unified_rdo_model,
)

__all__ = [
    'ProjectionConfig',
    'UnifiedRDOConfig',
    'UnifiedRDOLayer',
    'RankKUnifiedLayer',
    'UnifiedRDOModel',
    'get_unified_rdo_model',
]
