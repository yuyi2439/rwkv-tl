"""Fused RWKV7 model (tilelang kernels), functional style.

Module-level ``time_mix`` / ``channel_mix`` (single-token decode) and
``time_mix_batch`` / ``channel_mix_batch`` (prefill) functions operate on a
``RWKV7ATTWeight`` / ``RWKV7FFNWeight`` plus a per-layer state dict, mirroring
``demo.rwkv7_torch``'s functional structure but dispatching to fused tilelang
kernels instead of plain torch ops.

Decode uses the fused neo kernels (``kernel.cmix_decode`` / ``kernel.tmix_decode``).
Prefill uses the batched fused kernels (``_tmix_prefill_front``/``_back`` for
TMIX, ``kernel.cmix_prefill`` for CMIX), numerically equivalent to the per-token
decode path (verified ~0.01 max-abs on fp16).

All weights stay at the model's IO dtype (fp16 by default); no fp32 weight
copies (a future quantization path must not multiply weight memory).

State tensors are updated in place (``copy_``), never rebound, so the class is
CUDA-Graph capturable.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import Tensor

from rwkv_tl._compat import maybe_torch_compile
from rwkv_tl.kernel import cmix_decode, cmix_prefill, tmix_decode
from rwkv_tl.kernel.tmix import _tmix_prefill_back, _tmix_prefill_front
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7ATTWeight, RWKV7FFNWeight, RWKV7Weight

from ._rwkv7_abc import RWKV7Model


def _dtype_s(x: Tensor) -> str:
    return "bfloat16" if x.dtype == torch.bfloat16 else "float16"


@functools.cache
def _cmix_decode_kernel(C: int, DTYPE: str):
    return cmix_decode(C, DTYPE)


@functools.cache
def _cmix_prefill_kernel(C: int, DTYPE: str, LEN_block: int):
    return cmix_prefill(C, DTYPE, LEN_block)


@functools.cache
def _tmix_decode_kernel(C: int, DTYPE: str, H: int, Rv: int, Rw: int, Ra: int, Rg: int):
    return tmix_decode(C, DTYPE, H, Rv, Rw, Ra, Rg)


@functools.cache
def _tmix_prefill_kernels(
    LEN: int, C: int, DTYPE: str, H: int, Rv: int, Rw: int, Ra: int, Rg: int
):
    return (
        _tmix_prefill_front(LEN, C, DTYPE, H, Rv, Rw, Ra, Rg),
        _tmix_prefill_back(LEN, C, DTYPE, H, Rv, Rw, Ra, Rg),
    )


def time_mix(
    weight: RWKV7ATTWeight,
    x0: Tensor,
    v_first: Tensor | None,
    state: dict[str, Tensor],
    H: int,
    N: int,
    *,
    first: int = 1,
) -> tuple[Tensor, Tensor]:
    """Fused single-token time-mix (decode): the whole TMIX chain in one call.

    ``state["x"]`` and ``state["rnn"]`` are updated in place; ``v_first`` is
    the value residual from layer 0 (pass ``first=1`` on layer 0 so it stores
    v; later layers gate toward it). Not persisted across tokens.
    """
    C = x0.shape[0]
    DTYPE = _dtype_s(x0)
    b = weight
    k = _tmix_decode_kernel(
        C, DTYPE, H, b.v1t.shape[0], b.w1t.shape[0], b.a1t.shape[0], b.g1t.shape[0]
    )
    x_state = state["x"]
    rnn_state = state["rnn"]
    vf = v_first if v_first is not None else torch.zeros_like(x0)
    out = k(
        x0,
        b.ln_pre.w,
        b.ln_pre.b,
        b.x_rkvwag,
        b.rkvWt,
        b.v1t,
        b.w1t,
        b.a1t,
        b.g1t,
        b.v2.T.contiguous(),
        b.w2.T.contiguous(),
        b.a2.T.contiguous(),
        b.g2.T.contiguous(),
        b.v0.reshape(-1),
        b.w0.reshape(-1),
        b.a0.reshape(-1),
        b.k_k.reshape(-1),
        b.k_a.reshape(-1),
        b.r_k,
        b.ln_x.w,
        b.ln_x.b,
        b.oWt,
        x_state,
        rnn_state,
        vf,
        first,
    )
    return out, vf


def channel_mix(
    weight: RWKV7FFNWeight,
    x0: Tensor,
    state: dict[str, Tensor],
) -> Tensor:
    """Fused single-token channel-mix (decode)."""
    C = x0.shape[0]
    DTYPE = _dtype_s(x0)
    b = weight
    k = _cmix_decode_kernel(C, DTYPE)
    return k(
        x0,
        b.ln_pre.w,
        b.ln_pre.b,
        b.x_k,
        b.kWt,
        b.vWt,
        state["x"],
    )


def time_mix_batch(
    weight: RWKV7ATTWeight,
    x0: Tensor,
    v_first: Tensor | None,
    state: dict[str, Tensor],
    H: int,
    N: int,
    *,
    first: int = 1,
) -> tuple[Tensor, Tensor]:
    """Batched time-mix (prefill) via the fused ``_tmix_prefill_front``/``_back``.

    ``v_first`` is the value residual from layer 0: layer 0 (``first != 0``)
    stores its whole ``[T, C]`` v; later layers gate toward it. ``state["x"]`` /
    ``state["rnn"]`` are updated in place.
    """
    T_len = x0.shape[0]
    b = weight
    DTYPE = _dtype_s(x0)
    front, back = _tmix_prefill_kernels(
        T_len,
        b.x_rkvwag.shape[1],
        DTYPE,
        H,
        b.v1t.shape[0],
        b.w1t.shape[0],
        b.a1t.shape[0],
        b.g1t.shape[0],
    )
    x_state = state["x"]
    rnn_state = state["rnn"]
    vf = (
        v_first
        if v_first is not None
        else torch.zeros(T_len, x0.shape[1], device=x0.device, dtype=x0.dtype)
    )
    first_v = 1 if first != 0 else 0
    rkv, w, a, kk_norm, B, g = front(
        x0,
        b.ln_pre.w,
        b.ln_pre.b,
        b.x_rkvwag,
        b.rkvWt,
        b.v1t,
        b.w1t,
        b.a1t,
        b.g1t,
        b.v2.T.contiguous(),
        b.w2.T.contiguous(),
        b.a2.T.contiguous(),
        b.g2.T.contiguous(),
        b.v0.reshape(-1),
        b.w0.reshape(-1),
        b.a0.reshape(-1),
        b.k_k.reshape(-1),
        b.k_a.reshape(-1),
        x_state,
        vf,
        first_v,
    )
    out = back(
        rkv,
        w,
        kk_norm,
        B,
        a,
        g,
        b.r_k,
        b.ln_x.w,
        b.ln_x.b,
        b.oWt,
        x0,
        rnn_state,
    )
    return out, vf


def channel_mix_batch(
    weight: RWKV7FFNWeight,
    x0: Tensor,
    state: dict[str, Tensor],
    LEN_block: int = 32,
) -> Tensor:
    """Batched channel-mix (prefill) via the fused ``cmix_prefill``.

    ``cmix_prefill`` requires ``LEN % LEN_block == 0``; the input is padded up
    to a multiple of ``LEN_block`` (zero rows, sliced off after) so any ``T``
    works.
    """
    C = x0.shape[1]
    T_len = x0.shape[0]
    DTYPE = _dtype_s(x0)
    b = weight
    k = _cmix_prefill_kernel(C, DTYPE, LEN_block)
    pad = (-T_len) % LEN_block
    if pad:
        x0 = F.pad(x0, (0, 0, 0, pad))
        # cmix_prefill writes prev_x = LN_pre(x0[-1]); keep the last pad row
        # equal to the real last token so prev_x stays correct.
        x0[-1] = x0[T_len - 1]
    out = k(
        x0,
        b.ln_pre.w,
        b.ln_pre.b,
        b.x_k,
        b.kWt,
        b.vWt,
        state["x"],
    )
    return out[:T_len]


class RWKV7TL(RWKV7Model):
    """Fused tilelang RWKV7 inference model.

    decode uses the fused neo kernels; prefill uses the batched fused kernels.
    State is passed in/out explicitly, so the instance is stateless.
    """

    def __init__(
        self,
        w: RWKV7Weight,
        *,
        is_torch_compile: bool = False,
        cmix_len_block: int = 32,
    ) -> None:
        super().__init__(w)
        self._is_torch_compile = is_torch_compile
        self._cmix_len_block = cmix_len_block

    @maybe_torch_compile
    def decode(self, token: Tensor, S: State) -> tuple[Tensor, State]:
        x = F.embedding(token, self.w.emb)
        if x.dim() > 1:
            x = x.squeeze(0)
        v_first: Tensor | None = None

        for i, block in enumerate(self.w.blocks):
            x, v_first = time_mix(
                block.att,
                x,
                v_first,
                S.tmix[i],
                self.H,
                self.N,
                first=1 if v_first is None else 0,
            )
            x = channel_mix(block.ffn, x, S.cmix[i])

        return self.w.head @ self.w.ln_out(x), S

    def prefill(self, tokens: Tensor, S: State) -> State:
        if tokens.ndim != 1:
            raise ValueError(f"Expected 1D token sequence, got {tokens.shape}")
        if tokens.numel() == 0:
            raise RuntimeError("prefill received an empty token sequence")

        x = self.w.emb[tokens]
        v_first: Tensor | None = None
        for i, block in enumerate(self.w.blocks):
            x, v_first = time_mix_batch(
                block.att,
                x,
                v_first,
                S.tmix[i],
                self.H,
                self.N,
                first=1 if v_first is None else 0,
            )
            x = channel_mix_batch(block.ffn, x, S.cmix[i], self._cmix_len_block)
        return S
