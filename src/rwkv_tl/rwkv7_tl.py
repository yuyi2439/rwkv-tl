"""Fused RWKV7 model (tilelang kernels), stateless.

Weights are bound to kernel wrappers at construction (see
``rwkv_tl.kernel``): decode/prefill only pass activations and per-layer
state. ``decode`` uses the coarse fused decode kernels; ``prefill`` uses the
fused batched kernels. The instance holds no runtime state -- ``State`` is
passed in and returned, so the model is CUDA-Graph capturable.

Kernel firmware is selected STATICALLY per weight object by the module-level
ops classes (``rwkv_tl.kernel.ops``): a ``QTensor`` weight routes to the
``w8a16`` firmware (the head already uses a real W8A16 GEMV), a plain fp16
weight to the plain firmware. No heuristics, no per-call branch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from rwkv_tl._compat import maybe_torch_compile
from rwkv_tl.core.model import RWKV7Model
from rwkv_tl.core.state import State
from rwkv_tl.core.weight import RWKV7Weight
from rwkv_tl.kernel import ln_kernel
from rwkv_tl.kernel.ops import CmixOps, HeadOps, TmixOps


def _dtype_s(dtype: torch.dtype) -> str:
    return "bfloat16" if dtype == torch.bfloat16 else "float16"


class RWKV7TL(RWKV7Model):
    """Fused tilelang RWKV7 inference model.

    Takes an already-loaded ``RWKV7Weight`` (same declaration as
    ``RWKV7Model``); kernels are compiled lazily on first use and weights are
    bound at construction.
    """

    def __init__(
        self,
        W: RWKV7Weight,
        *,
        is_torch_compile: bool = False,
        cmix_len_block: int = 32,
    ) -> None:
        super().__init__(W)
        self._is_torch_compile = is_torch_compile
        self._cmix_len_block = cmix_len_block
        self._DTYPE = _dtype_s(W.dtype)

        # Each op binds one weight object and statically routes by weight type
        # (QTensor -> w8a16, tensor -> fp16). Prefill kernels are cached inside
        # the ops, keyed by sequence length.
        self._head_ops = HeadOps(W.head, self._DTYPE)
        self._ln_out = ln_kernel(self.C, self._DTYPE, W.ln_out.w, W.ln_out.b)
        self._att = [TmixOps(b.att, self.C, self.H, self._DTYPE) for b in W.blocks]
        self._ffn = [CmixOps(b.ffn, self.C, self._DTYPE, self._cmix_len_block) for b in W.blocks]

        if is_torch_compile and self.w.device.type == "cuda":
            # torch.compile traces decode on first call; compile the bound
            # kernels first so the trace sees the compiled fast path.
            _ = self._ln_out.kernel
            for op in self._att:
                _ = op.decode.kernel
            for op in self._ffn:
                _ = op.decode.kernel

    @maybe_torch_compile
    def decode(self, token: Tensor, S: State) -> tuple[Tensor, State]:
        x = F.embedding(token, self.w.emb)
        if x.dim() > 1:
            x = x.squeeze(0)
        v_first: Tensor | None = None

        for i, block in enumerate(self.w.blocks):
            st = S.tmix[i]
            vf = v_first if v_first is not None else torch.zeros_like(x)
            x = self._att[i](x, st["x"], st["rnn"], vf, 1 if v_first is None else 0)
            v_first = vf
            x = self._ffn[i](x, S.cmix[i]["x"])

        x_ln = self._ln_out(x)
        return self._head_ops(x_ln), S

    def prefill(self, tokens: Tensor, S: State) -> State:
        if tokens.ndim != 1:
            raise ValueError(f"Expected 1D token sequence, got {tokens.shape}")
        if tokens.numel() == 0:
            raise RuntimeError("prefill received an empty token sequence")

        x = self.w.emb[tokens]
        T_len = x.shape[0]
        v_first: Tensor | None = None
        for i, block in enumerate(self.w.blocks):
            st = S.tmix[i]
            vf = (
                v_first
                if v_first is not None
                else torch.zeros(T_len, self.C, device=x.device, dtype=x.dtype)
            )
            x = self._att[i].prefill(T_len)(x, st["x"], st["rnn"], vf, 1 if v_first is None else 0)
            v_first = vf

            pad = (-T_len) % self._cmix_len_block
            x0 = x
            if pad:
                x0 = F.pad(x0, (0, 0, 0, pad))
                # cmix_prefill writes prev_x = LN_pre(x0[-1]); keep the last
                # pad row equal to the real last token so prev_x stays correct.
                x0[-1] = x0[T_len - 1]
            x = self._ffn[i].prefill(T_len)(x0, S.cmix[i]["x"])[:T_len]
        return S