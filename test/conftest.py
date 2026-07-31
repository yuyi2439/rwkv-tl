"""Shared pytest fixtures for rwkv-tl tests.

RWKV_CHECKPOINT_PATH must be set to a valid RWKV7 checkpoint path for tests that require a model checkpoint.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is importable when running `pytest` from repo root without install.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import pytest
import torch


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def ckpt_path() -> str:
    ckpt_path = os.environ.get("RWKV_CHECKPOINT_PATH")
    if not ckpt_path:
        raise RuntimeError("RWKV_CHECKPOINT_PATH must be set for tests that need a checkpoint")
    return ckpt_path


@pytest.fixture(scope="session")
def vocab_path(repo_root: Path) -> str:
    return str(repo_root / "asset" / "rwkv_vocab_v20230424.txt")


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
