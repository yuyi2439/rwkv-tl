from __future__ import annotations

from pathlib import Path
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
        dtype: torch.dtype = torch.float16,
    ) -> Self:
        head_count = n_embd // head_dim

        tmix_layers: list[dict[str, Tensor]] = []
        cmix_layers: list[dict[str, Tensor]] = []
        for _ in range(n_layer):
            tmix_layers.append(
                {
                    "x": torch.zeros(n_embd, dtype=dtype, device=device),
                    "rnn": torch.zeros(
                        (head_count, head_dim, head_dim),
                        dtype=torch.float32,
                        device=device,
                    ),
                }
            )
            cmix_layers.append({"x": torch.zeros(n_embd, dtype=dtype, device=device)})

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

    def save(self, path: str | Path) -> None:
        """Save the state to a file (``State.load`` reads it back)."""
        torch.save({"tmix": self.tmix, "cmix": self.cmix}, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Load a state saved by ``State.save``.

        Args:
            path: File written by ``save``.
            device: Move the loaded tensors to this device (None keeps the
                saved device).
            dtype: Convert loaded tensors to this dtype (None keeps the saved
                dtype). The RNN state stays fp32 as usual.
        """
        data = torch.load(path, map_location=device, weights_only=True)
        try:
            tmix = data["tmix"]
            cmix = data["cmix"]
            assert isinstance(tmix, list) and isinstance(cmix, list)
        except (KeyError, TypeError, AssertionError) as e:
            raise ValueError(f"{path} is not a saved rwkv_tl State file") from e

        if dtype is not None:
            tmix = [
                {k: (t.to(dtype) if k != "rnn" else t) for k, t in layer.items()}
                for layer in tmix
            ]
            cmix = [{k: t.to(dtype) for k, t in layer.items()} for layer in cmix]
        return tuple.__new__(cls, (tmix, cmix))
