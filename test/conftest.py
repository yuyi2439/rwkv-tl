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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "compile: slow torch.compile-of-decode integration test (run with -m compile)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # Only run when explicitly selected via -m compile: compiling decode on
    # the 0.1B model takes ~1 min on a small GPU, so keep it out of the
    # default suite (fast correctness tests run eager).
    if not config.getoption("markexpr"):
        skip = pytest.mark.skip(
            reason="slow; run with `-m compile` to exercise torch.compile"
        )
        for item in items:
            if "compile" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def ckpt_path() -> str:
    ckpt_path = os.environ.get("RWKV_CHECKPOINT_PATH")
    if not ckpt_path:
        raise RuntimeError(
            "RWKV_CHECKPOINT_PATH must be set for tests that need a checkpoint"
        )
    return ckpt_path


@pytest.fixture(scope="session")
def vocab_path(repo_root: Path) -> str:
    return str(repo_root / "asset" / "rwkv_vocab_v20230424.txt")


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
