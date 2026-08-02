"""Run the rwkv-tl pytest suite with CUDA visible.

Usage:
    python script/run_tests.py              # run all tests
    python script/run_tests.py -k kernels   # run only kernel tests
    python script/run_tests.py -v           # verbose

Extra argv after the script name is forwarded to pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    import pytest

    # Default args: run tests/ from repo root, show short test summary.
    argv = [str(REPO / "test"), "-ra", "-q"]
    # Forward any user-supplied args (e.g. -k, -v, --tb=short).
    argv.extend(sys.argv[1:])
    return pytest.main(argv)


if __name__ == "__main__":
    sys.exit(main())
