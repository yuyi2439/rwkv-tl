"""Abstract RWKV7 model interface shared by all implementations.

Every model (tilelang fp16/bf16, MX450, pure torch) conforms to this small
contract -- ``decode`` / ``prefill`` / ``forward`` / ``generate`` -- so
application scripts can build a model via ``make_rwkv7(..., backend="auto")``
and operate on it without knowing which kernel strategy is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor

from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7Weight


class RWKV7Model(ABC):
    """Common inference interface for all RWKV7 implementations."""

    def __init__(self, w: RWKV7Weight, *args, **kwargs) -> None:
        """Initialize common RWKV7 model fields shared by all backends."""
        self.w = w

        self.L = w.L
        self.C = w.C
        self.N = 64  # head dimension (fixed at 64 for RWKV7)
        self.H = self.C // self.N  # head count (C / N)

    @abstractmethod
    def decode(self, token: Tensor, S: State) -> tuple[Tensor, State]:
        """Advance one token; returns ``(logits, state)``."""

    @abstractmethod
    def prefill(self, tokens: Tensor, S: State) -> State:
        """Batch-fill a token sequence, updating ``S`` in place; returns ``S``."""

    @abstractmethod
    def forward(
        self,
        tokens: list[int] | Tensor,
        S: State,
    ) -> tuple[Tensor, State]:
        """Run inference over a token sequence; returns ``(logits, state)``."""

    @abstractmethod
    def generate(
        self,
        tokens: list[int] | Tensor,
        S: State,
        max_tokens: int = 32,
        *,
        temperature: float | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop: list[list[int]] | None = None,
    ) -> tuple[list[int], State]:
        """Generate autoregressively from a prompt; returns ``(tokens, state)``."""
