"""Abstract RWKV7 model interface shared by all implementations.

Every backend (tilelang fused kernels, pure torch) conforms to one stateless
contract: ``decode`` / ``prefill`` / ``forward`` take an explicit ``State``
and return it (updated in place), so models never own runtime state. On top
of that low-level contract the base class adds the user-facing text API:
``generate("prompt")``, ``logits(...)`` for raw distributions, and
``tune_state("prompt")`` for prompt -> state (state tune) with
``State.save`` / ``State.load``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor

from .sampling import sample_logits
from .state import State
from .tokenizer import Tokenizer
from .weight import RWKV7Weight


def _to_ids(input: str | int | Sequence[int] | Tensor, device: torch.device) -> Tensor:
    """Coerce a user input (text, token id, token list, or 1-D tensor) to a
    1-D int64 token tensor on ``device``."""
    if isinstance(input, str):
        raise TypeError("_to_ids does not tokenize text; call model.encode() first")
    if isinstance(input, Tensor):
        ids = input.reshape(-1).tolist()
    elif isinstance(input, int):
        ids = [input]
    else:
        ids = list(input)
    return torch.tensor(ids, dtype=torch.long, device=device)


class RWKV7Model(ABC):
    """Common stateless inference interface for all RWKV7 implementations."""

    def __init__(self, w: RWKV7Weight) -> None:
        self.w = w
        self.L = w.L
        self.C = w.C
        self.N = 64  # head dimension (fixed at 64 for RWKV7)
        self.H = self.C // self.N  # head count (C / N)
        self.tokenizer = Tokenizer()

    # ------------------------------------------------------------------
    # low-level contract (stateless: state is passed in and returned)
    # ------------------------------------------------------------------

    @abstractmethod
    def decode(self, token: Tensor, S: State) -> tuple[Tensor, State]:
        """Advance one token; returns ``(logits, state)``."""

    @abstractmethod
    def prefill(self, tokens: Tensor, S: State) -> State:
        """Batch-fill a 1-D token sequence, updating ``S`` in place; returns ``S``."""

    def new_state(self) -> State:
        """A fresh zero state on the model's device/dtype."""
        return State(self.L, self.C, self.N, device=self.w.device, dtype=self.w.dtype)

    # ------------------------------------------------------------------
    # text-level API (still stateless: pass/return an explicit State)
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Tokenize text into token ids."""
        return self.tokenizer.encode(text)

    def detokenize(self, ids: Sequence[int]) -> str:
        """Decode token ids back into text."""
        return self.tokenizer.decode(list(ids))

    def tune_state(
        self, prompt: str | Sequence[int] | Tensor, state: State | None = None
    ) -> State:
        """State tune: process a full prompt into state.

        ``S = model.tune_state("You are a helpful assistant.")`` returns a
        state positioned after the whole prompt; save/load it with
        ``S.save(path)`` / ``State.load(path)`` and attach it to
        ``generate(..., state=S)``.
        """
        S = state if state is not None else self.new_state()
        if isinstance(prompt, str):
            tokens = _to_ids(self.encode(prompt), self.w.device)
        else:
            tokens = _to_ids(prompt, self.w.device)
        if tokens.numel() == 0:
            return S
        return self.prefill(tokens, S)

    def logits(
        self, input: str | int | Sequence[int] | Tensor, state: State | None = None
    ) -> tuple[Tensor, State]:
        """Low-level: process ``input`` and return ``(logits, state)``.

        ``logits`` is the raw next-token distribution after the whole input
        has been consumed (the same distribution ``generate`` samples from).
        Accepts text, a token id, a token sequence, or a 1-D token tensor.
        """
        S = state if state is not None else self.new_state()
        if isinstance(input, str):
            tokens = _to_ids(self.encode(input), self.w.device)
        else:
            tokens = _to_ids(input, self.w.device)
        if tokens.numel() == 0:
            raise ValueError("logits() requires a non-empty input")
        return self.forward(tokens, S)

    def generate(
        self,
        input: str | int | Sequence[int] | Tensor,
        state: State | None = None,
        *,
        max_tokens: int = 32,
        temperature: float | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> str | list[int]:
        """Generate text (str input -> str output) or token ids from ``input``.

        The input is always processed first (prefill + final-token decode), so
        attaching a tuned state with ``generate("...", state=S)`` continues
        from that state. A fresh zero state is created when ``state`` is None;
        pass one explicitly to keep it (it is updated in place).
        """
        text_mode = isinstance(input, str)
        if text_mode:
            tokens = _to_ids(self.encode(input), self.w.device)
        else:
            tokens = _to_ids(input, self.w.device)
        if tokens.numel() == 0:
            raise ValueError(
                "generate() requires a non-empty input; to continue from a "
                "tuned state, pass the next text/token as input"
            )

        S = state if state is not None else self.new_state()
        out = torch.tensor([], dtype=torch.long, device=self.w.device)

        if tokens.numel() == 1:
            logits, S = self.decode(tokens, S)
        else:
            S = self.prefill(tokens[:-1], S)
            logits, S = self.decode(tokens[-1:], S)

        for _ in range(max_tokens):
            token = sample_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seen=out,
            )
            out = torch.cat([out, token.unsqueeze(0)])
            logits, S = self.decode(token, S)

        return self.detokenize(out.tolist()) if text_mode else out.tolist()

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
