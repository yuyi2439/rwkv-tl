"""Low-level stateless RWKV7 model interface.

``RWKV7Model`` is the token-only contract shared by every backend (tilelang
fused kernels, pure torch): ``decode`` / ``prefill`` / ``forward`` take an
explicit ``State`` and return it (updated in place), so models never own
runtime state. The base class deliberately has no tokenizer and no state
creation helpers; text-level and state-level conveniences live in higher-level
wrappers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor

from .state import State
from .weight import RWKV7Weight


class RWKV7Model(ABC):
    """Common stateless inference interface for all RWKV7 implementations."""

    def __init__(self, w: RWKV7Weight) -> None:
        self.w = w
        self.L = w.L
        self.C = w.C
        self.N = 64  # head dimension (fixed at 64 for RWKV7)
        self.H = self.C // self.N  # head count (C / N)

    # ------------------------------------------------------------------
    # low-level contract (stateless: state is passed in and returned)
    # ------------------------------------------------------------------

    @abstractmethod
    def decode(self, token: Tensor, S: State) -> tuple[Tensor, State]:
        """Advance one token; returns ``(logits, state)``."""

    @abstractmethod
    def prefill(self, tokens: Tensor, S: State) -> State:
        """Batch-fill a 1-D token sequence, updating ``S`` in place; returns ``S``."""

    def forward(self, tokens: Tensor, S: State) -> tuple[Tensor, State]:
        """Run inference over a token sequence; returns ``(logits, state)``."""
        if tokens.ndim == 0 and tokens.numel() == 1:
            return self.decode(tokens, S)
        if tokens.ndim != 1:
            raise ValueError(f"Expected 1D token sequence, got {tokens.shape}")
        if tokens.numel() == 1:
            return self.decode(tokens[0], S)

        S = self.prefill(tokens[:-1], S)
        token = tokens[-1]
        return self.decode(token, S)
