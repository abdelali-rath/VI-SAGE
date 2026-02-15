"""
Pytest configuration and shared fixtures for VI-SAGE tests.

Ensures the project root is on sys.path so that `src` and `app` can be imported
when running tests from the repo root (e.g. `pytest tests/`).
"""

import os
import sys
from pathlib import Path

import pytest

# Project root (parent of tests/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def project_root():
    """Return the project root directory as a Path."""
    return ROOT


@pytest.fixture
def sample_image_tensor():
    """Batch of 2 fake RGB images [B, 3, 224, 224] for model forward tests."""
    import torch

    return torch.rand(2, 3, 224, 224)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """A temporary directory suitable for saving a small checkpoint."""
    return tmp_path
