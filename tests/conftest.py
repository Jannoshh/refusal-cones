"""Pytest configuration for refusal-cones tests."""

import sys
from pathlib import Path

# Add project root to path so tests can import from src.*
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
