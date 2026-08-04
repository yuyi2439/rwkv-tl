"""torch.compile integration: generate must not recompile per token.

Passing a Python int token into the compiled ``decode`` makes Dynamo
specialize on the int *value*, so every distinct token id triggers a
recompile; with ``fullgraph=True`` it hard-fails after ``recompile_limit``
(8) distinct values (``FailOnRecompileLimitHit``). ``generate`` must pass a
CUDA 0-dim tensor instead so the runtime token value stays out of the
specialization keys.

The old eager-only correctness tests (test_forward/test_graph) can't catch
this, so this test drives a compiled model through >8 distinct greedy
tokens. Slow (~1 min first compile on small GPUs): run with ``-m compile``.
"""

from __future__ import annotations

import pytest
import torch

from rwkv_tl import RWKV7
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7Weight

N_TOKENS = 32
TOKENS = [(i * 1103515245 + 12345) % 65536 for i in range(N_TOKENS)]
MIN_DISTINCT = 12  # must exceed dynamo's recompile_limit (8) to catch the bug


@pytest.mark.compile
def test_compile_generate_no_recompile(ckpt_path: str) -> None:
    with torch.device("cuda"):
        model = RWKV7(RWKV7Weight(ckpt_path), is_torch_compile=True)
        S = State(
            model.w.N_LAYER,
            model.w.N_EMBD,
            64,
            device=model.emb.device,
        )
        # Greedy degenerates into a repeated-token loop on this model
        # (only ~5 distinct ids), so sample with a fixed seed to get a
        # deterministic continuation with many distinct token ids. With the
        # old int-token bug this raises FailOnRecompileLimitHit on the ~9th
        # distinct token instead of completing.
        torch.manual_seed(0)
        toks, _ = model.generate(TOKENS, S, max_tokens=48, temperature=0.8, top_p=0.9)
    distinct = len(set(toks))
    assert distinct > MIN_DISTINCT, (
        f"need >{MIN_DISTINCT} distinct greedy tokens to exercise the "
        f"recompile-limit path, got {distinct}"
    )
