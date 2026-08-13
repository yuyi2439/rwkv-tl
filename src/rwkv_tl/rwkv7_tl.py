"""Fused RWKV7 model (tilelang kernels), stateless.

Weights are bound to kernel wrappers at construction (see
``rwkv_tl.kernel``): decode/prefill only pass activations and per-layer
state. ``decode`` uses the coarse fused decode kernels; ``prefill`` uses the
fused batched kernels. The instance holds no runtime state -- ``State`` is
passed in and returned, so the model is CUDA-Graph capturable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from rwkv_tl._compat import maybe_torch_compile
from rwkv_tl.kernel import (
    cmix_decode_kernel,
    cmix_prefill_kernel,
    ln_kernel,
    tmix_decode_kernel,
    tmix_prefill_kernel,
)
from rwkv_tl.model import RWKV7Model
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7ATTWeight, RWKV7FFNWeight, RWKV7Weight


def _dtype_s(dtype: torch.dtype) -> str:
    return "bfloat16" if dtype == torch.bfloat16 else "float16"


class RWKV7TL(RWKV7Model):
    """Fused tilelang RWKV7 inference model.

    Accepts a checkpoint path or an ``RWKV7Weight``. Kernels are compiled
    lazily on first use; weights are bound at construction.
    """

    def __init__(
        self,
        path_or_weight: str | RWKV7Weight,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
        is_torch_compile: bool = False,
        cmix_len_block: int = 32,
    ) -> None:
        w = (
            path_or_weight
            if isinstance(path_or_weight, RWKV7Weight)
            else RWKV7Weight(path_or_weight, device=device, dtype=dtype)
        )
        super().__init__(w)
        self._is_torch_compile = is_torch_compile
        self._cmix_len_block = cmix_len_block
        self._DTYPE = _dtype_s(w.dtype)

        self._ln_out = ln_kernel(self.C, self._DTYPE, w.ln_out.w, w.ln_out.b)
        self._att = [self._bind_tmix_decode(b.att) for b in w.blocks]
        self._ffn = [self._bind_cmix_decode(b.ffn) for b in w.blocks]
        # (layer, T_len) -> (tmix prefill, cmix prefill) bound kernels
        self._prefill_cache: dict[tuple[int, int], tuple] = {}

        if is_torch_compile and torch.cuda.is_available():
            # torch.compile traces decode on first call; compile the bound
            # kernels first so the trace sees the compiled fast path.
            _ = self._ln_out.kernel
            for k in self._att:
                _ = k.kernel
            for k in self._ffn:
                _ = k.kernel

    def _bind_tmix_decode(self, b: RWKV7ATTWeight):
        return tmix_decode_kernel(
            self.C,
            self._DTYPE,
            self.H,
            b.v1t.shape[0],
            b.w1t.shape[0],
            b.a1t.shape[0],
            b.g1t.shape[0],
            ln_preW=b.ln_pre.w,
            ln_preB=b.ln_pre.b,
            x_rkvwag=b.x_rkvwag,
            rkvWt=b.rkvWt,
            v1t=b.v1t,
            w1t=b.w1t,
            a1t=b.a1t,
            g1t=b.g1t,
            v2t=b.v2t,
            w2t=b.w2t,
            a2t=b.a2t,
            g2t=b.g2t,
            v0=b.v0,
            w0=b.w0,
            a0=b.a0,
            k_k=b.k_k.reshape(-1),
            k_a=b.k_a.reshape(-1),
            r_k=b.r_k,
            ln_xW=b.ln_x.w,
            ln_xB=b.ln_x.b,
            oWt=b.oWt,
        )

    def _bind_cmix_decode(self, b: RWKV7FFNWeight):
        return cmix_decode_kernel(
            self.C,
            self._DTYPE,
            ln_preW=b.ln_pre.w,
            ln_preB=b.ln_pre.b,
            x_k=b.x_k,
            kWt=b.kWt,
            vWt=b.vWt,
        )

    def _prefill_kernels(self, layer: int, T_len: int, block):
        key = (layer, T_len)
        kernels = self._prefill_cache.get(key)
        if kernels is None:
            b = block.att
            tmix = tmix_prefill_kernel(
                T_len,
                self.C,
                self._DTYPE,
                self.H,
                b.v1t.shape[0],
                b.w1t.shape[0],
                b.a1t.shape[0],
                b.g1t.shape[0],
                ln_preW=b.ln_pre.w,
                ln_preB=b.ln_pre.b,
                x_rkvwag=b.x_rkvwag,
                rkvWt=b.rkvWt,
                v1t=b.v1t,
                w1t=b.w1t,
                a1t=b.a1t,
                g1t=b.g1t,
                v2t=b.v2t,
                w2t=b.w2t,
                a2t=b.a2t,
                g2t=b.g2t,
                v0=b.v0,
                w0=b.w0,
                a0=b.a0,
                k_k=b.k_k.reshape(-1),
                k_a=b.k_a.reshape(-1),
                r_k=b.r_k,
                ln_xW=b.ln_x.w,
                ln_xB=b.ln_x.b,
                oWt=b.oWt,
            )
            cmix = cmix_prefill_kernel(
                self.C,
                self._DTYPE,
                self._cmix_len_block,
                ln_preW=block.ffn.ln_pre.w,
                ln_preB=block.ffn.ln_pre.b,
                x_k=block.ffn.x_k,
                kWt=block.ffn.kWt,
                vWt=block.ffn.vWt,
            )
            kernels = (tmix, cmix)
            self._prefill_cache[key] = kernels
        return kernels

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

        return torch.mv(self.w.head, self._ln_out(x)), S

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
            tmix, cmix = self._prefill_kernels(i, T_len, block)
            x = tmix(x, st["x"], st["rnn"], vf, 1 if v_first is None else 0)
            v_first = vf

            pad = (-T_len) % self._cmix_len_block
            x0 = x
            if pad:
                x0 = F.pad(x0, (0, 0, 0, pad))
                # cmix_prefill writes prev_x = LN_pre(x0[-1]); keep the last
                # pad row equal to the real last token so prev_x stays correct.
                x0[-1] = x0[T_len - 1]
            x = cmix(x0, S.cmix[i]["x"])[:T_len]
        return S
