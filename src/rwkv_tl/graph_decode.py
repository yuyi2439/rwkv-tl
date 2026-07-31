"""CUDA Graph-accelerated single-token decoder for RWKV7.

Captures one ``run_one`` step into a ``torch.cuda.CUDAGraph`` and replays it
per token, eliminating Python dispatch and kernel-launch overhead. On small
models where launch overhead dominates GPU compute (profiler showed ~61% idle
time on MX450), this gives the largest single speedup.

Requires in-place state updates (see ``RWKV7.reset_state``) so that state
tensor addresses stay fixed across replays.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class GraphDecoder:
    """Single-token decoder backed by a captured CUDA Graph.

    Usage::

        model = RWKV7(ckpt, vocab)  # weights on cuda
        dec = GraphDecoder(model)
        dec.reset()
        for tok in prompt:
            logits = dec.step(tok)
        # logits buffer is reused; clone if you need to keep it.

    Args:
        model: An ``RWKV7`` whose weights and state reside on CUDA.
        warmup: Number of warmup iterations on the side stream before
            capture. These runs initialise cuBLAS handles and force lazy
            allocations so the captured graph is self-contained.
    """

    def __init__(self, model, warmup: int = 3) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("GraphDecoder requires CUDA")

        self.model = model
        self.token_buf = torch.zeros(1, dtype=torch.long, device="cuda")
        self.state = self._cuda_state()

        # Output logits buffer — populated by the captured forward.
        self.logits: Tensor | None = None

        self._capture(warmup)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Zero the RNN state in-place (keeps tensor addresses fixed)."""
        self.model.reset_state(self.state)

    def step(self, token_id: int) -> Tensor:
        """Advance one token via graph replay.

        Args:
            token_id: Input token id.

        Returns:
            Logits tensor. The underlying buffer is reused on the next
            call; ``clone()`` it if you need to persist the values.
        """
        self.token_buf[0] = token_id
        self.graph.replay()
        return self.logits

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _cuda_state(self):
        """Create a fresh zero state with all tensors on CUDA."""
        with torch.device("cuda"):
            return self.model.zero_state()

    def _run_step(self) -> Tensor:
        """Execute one forward step reading from ``token_buf``.

        Uses ``F.embedding`` instead of ``self.model.EMB(token_buf[0])``
        because GPU-tensor indexing (``emb[cuda_0d_tensor]``) invalidates
        CUDA Graph capture (``cudaErrorStreamCaptureInvalidated``).
        ``F.embedding`` is a graph-safe gather that reads the token
        dynamically from ``token_buf`` on each replay.
        """
        X = F.embedding(self.token_buf, self.model.emb).squeeze(0)
        v_first: Tensor | None = None
        for (TM, CM), s in zip(self.model.layers, self.state):
            X, v_first, s[0] = TM(X, v_first, s[0])
            X, s[1] = CM(X, s[1])
        return self.model.HEAD(self.model.NORM(X))

    def _capture(self, warmup: int) -> None:
        """Warmup on a side stream then capture the graph."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self.reset()
                self.logits = self._run_step()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            self.logits = self._run_step()
