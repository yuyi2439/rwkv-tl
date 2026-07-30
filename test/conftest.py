"""Shared pytest fixtures for rwkv-tl tests."""
from __future__ import annotations

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
    return "/home/yuyi2439/rwkv/rwkv7-g1d-0.1b-20260129-ctx8192.pth"


@pytest.fixture(scope="session")
def vocab_path(repo_root: Path) -> str:
    return str(repo_root / "asset" / "rwkv_vocab_v20230424.txt")


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
