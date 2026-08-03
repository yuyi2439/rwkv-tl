from __future__ import annotations

from typing import Self

import torch
from torch import Tensor


class State(tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]):
    """Runtime state for RWKV7 inference.

    The state is stored as a 2-tuple:
    - element 0: TMIX states, one per layer
    - element 1: CMIX states, one per layer

    Each per-layer state is a dictionary of tensors that persist across token
    steps and are updated in-place during decoding and prefill.
    """

    def __new__(
        cls,
        n_layer: int,
        n_embd: int,
        head_dim: int,
        *,
        device: torch.device | str | None = None,
    ) -> Self:
        head_count = n_embd // head_dim

        tmix_layers: list[dict[str, Tensor]] = []
        cmix_layers: list[dict[str, Tensor]] = []
        for _ in range(n_layer):
            tmix_layers.append(
                {
                    "x": torch.zeros(n_embd, dtype=torch.bfloat16, device=device),
                    "rnn": torch.zeros(
                        (head_count, head_dim, head_dim),
                        dtype=torch.float32,
                        device=device,
                    ),
                }
            )
            cmix_layers.append(
                {"x": torch.zeros(n_embd, dtype=torch.bfloat16, device=device)}
            )

        obj = super().__new__(cls, (tmix_layers, cmix_layers))
        return obj

    @property
    def tmix(self) -> list[dict[str, Tensor]]:
        return self[0]

    @property
    def cmix(self) -> list[dict[str, Tensor]]:
        return self[1]

    def reset(self) -> None:
        """Zero all state tensors in place."""
        for s in self.tmix + self.cmix:
            for tensor in s.values():
                tensor.zero_()
