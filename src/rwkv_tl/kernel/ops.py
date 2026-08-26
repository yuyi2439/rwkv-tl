"""Module-level ops: bind weights and statically select kernel firmware.

Unlike the fine-grained ``*_kernel`` factories in ``gemv``/``ln``/``tmix``/
``cmix``, these classes each own ONE weight object (the head projection, one
attention block, or one ffn block) and route by the **weight type** at
construction -- not by any heuristic:

- A ``QTensor`` big projection (rkvWt/oWt/kWt/vWt/head) selects the ``w8a16``
  path and the kernel firmware that consumes it. The head uses the real W8A16
  GEMV; cmix decode keeps ``kWt/vWt`` int8-resident via a fused W8A16 kernel.
  Paths that still bind fp16 (tmix decode, all prefill) materialize the
  dequantized fp16 view on demand.
- A plain fp16/bfloat16 ``Tensor`` selects the plain fp16 firmware directly.

Low-rank/gate params (v1t/w1t/a1t/g1t/...) are never quantized and are shared
unchanged by both paths. Routing is fixed once at construction; there is no
``isinstance`` at call time.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rwkv_tl.kernel import (
    cmix_decode_kernel,
    cmix_decode_q8_kernel,
    cmix_prefill_kernel,
    gemv_q8_kernel,
    tmix_decode_kernel,
    tmix_prefill_kernel,
)
from rwkv_tl.quant import QTensor

# Big projection weights that may be quantized, per module.
_TMIX_PROJ = ("rkvWt", "oWt")
_CMIX_PROJ = ("kWt", "vWt")


def _is_q8(*w: Any) -> bool:
    """True if any weight is a QTensor (static type route selector)."""
    return any(isinstance(x, QTensor) for x in w)


class HeadOps:
    """Head projection. ``QTensor`` -> real W8A16 GEMV; fp16 -> ``torch.mv``."""

    def __init__(self, head: Tensor | QTensor, DTYPE: str) -> None:
        self._mode = "w8a16" if isinstance(head, QTensor) else "fp16"
        if self._mode == "w8a16":
            self._gemv = gemv_q8_kernel(
                head.shape[1], head.shape[0], head.group, DTYPE, head.q, head.s
            )
            self._head = None
        else:
            self._gemv = None
            self._head = head

    def __call__(self, x: Tensor) -> Tensor:
        return self._gemv(x) if self._gemv is not None else torch.mv(self._head, x)

    def __repr__(self) -> str:
        return f"HeadOps(mode={self._mode!r})"


class TmixOps:
    """Binds one attention block's decode + prefill kernels.

    Static route: ``rkvWt`` or ``oWt`` being a QTensor selects the ``w8a16``
    path. The tmix decode/prefill elements are still fp16 firmware, so a
    QTensor big projection binds via its dequantized fp16 view (materialized
    once on the source weight, int8 payload released) -- a fused W8A16 tmix
    kernel is the follow-up. Prefill kernels are cached per ``T_len``.
    """

    def __init__(self, W, C: int, H: int, DTYPE: str) -> None:
        proj = _TMIX_PROJ
        self._mode = "w8a16" if _is_q8(*(getattr(W, n) for n in proj)) else "fp16"
        if self._mode == "w8a16":
            # tmix firmware is still fp16: materialize the fp16 view on the
            # source weight and release the int8 payload. (Takes ownership.)
            for n in proj:
                setattr(W, n, getattr(W, n).dequant())

        self._R = (W.v1t.shape[0], W.w1t.shape[0], W.a1t.shape[0], W.g1t.shape[0])
        self._C, self._H = C, H
        self._DTYPE = DTYPE
        self._args = (C, DTYPE, H, *self._R)

        self._w: dict[str, Any] = {
            "ln_preW": W.ln_pre.w,
            "ln_preB": W.ln_pre.b,
            "x_rkvwag": W.x_rkvwag,
            "rkvWt": W.rkvWt,
            "v1t": W.v1t,
            "w1t": W.w1t,
            "a1t": W.a1t,
            "g1t": W.g1t,
            "v2t": W.v2t,
            "w2t": W.w2t,
            "a2t": W.a2t,
            "g2t": W.g2t,
            "v0": W.v0,
            "w0": W.w0,
            "a0": W.a0,
            "k_k": W.k_k.reshape(-1),
            "k_a": W.k_a.reshape(-1),
            "r_k": W.r_k,
            "ln_xW": W.ln_x.w,
            "ln_xB": W.ln_x.b,
            "oWt": W.oWt,
        }

        self.decode = tmix_decode_kernel(*self._args, **self._w)
        self._prefill: dict[int, Any] = {}

    def prefill(self, T_len: int):
        k = self._prefill.get(T_len)
        if k is None:
            k = tmix_prefill_kernel(T_len, *self._args, **self._w)
            self._prefill[T_len] = k
        return k

    def __call__(self, x0, prev_x, rnn, v_first, first):
        return self.decode(x0, prev_x, rnn, v_first, first)

    def __repr__(self) -> str:
        return f"TmixOps(mode={self._mode!r})"


class CmixOps:
    """Binds one ffn block's decode + prefill kernels.

    Static route: ``kWt`` or ``vWt`` being a QTensor selects ``w8a16``. The
    decode kernel is a fused W8A16 variant that keeps ``kWt/vWt`` **int8
    resident** (no permanent fp16 copy). Prefill firmware is fp16 (tensor-core
    GEMM), so it is rebuilt on demand from a transient dequant and dropped
    after the call -- decode stays int8-resident, for a real VRAM saving in
    the steady-state decode loop.
    """

    def __init__(self, W, C: int, DTYPE: str, cmix_len_block: int = 32) -> None:
        self._C, self._DTYPE = C, DTYPE
        self._cmix_len_block = cmix_len_block
        proj = _CMIX_PROJ
        self._mode = "w8a16" if _is_q8(*(getattr(W, n) for n in proj)) else "fp16"

        self._w_fp16: dict[str, Any] = {
            "ln_preW": W.ln_pre.w,
            "ln_preB": W.ln_pre.b,
            "x_k": W.x_k,
        }
        if self._mode == "w8a16":
            # Fused W8A16 decode: kWt/vWt stay int8-resident.
            self._kWt = W.kWt
            self._vWt = W.vWt
            self.decode = cmix_decode_q8_kernel(
                C,
                W.kWt.group,
                DTYPE,
                ln_preW=W.ln_pre.w,
                ln_preB=W.ln_pre.b,
                x_k=W.x_k,
                kWq=W.kWt.q,
                ks=W.kWt.s,
                vWq=W.vWt.q,
                vs=W.vWt.s,
            )
        else:
            self._w_fp16["kWt"] = W.kWt
            self._w_fp16["vWt"] = W.vWt
            self.decode = cmix_decode_kernel(C, DTYPE, **self._w_fp16)
        self._prefill: dict[int, Any] = {}

    def prefill(self, T_len: int):
        if self._mode == "w8a16":
            # fp16 GEMM builds a transient dequant each call and is NOT cached,
            # so its fp16 copy is released and decode stays int8-resident.
            w = dict(self._w_fp16)
            w["kWt"] = self._kWt.dequant()
            w["vWt"] = self._vWt.dequant()
            return cmix_prefill_kernel(self._C, self._DTYPE, self._cmix_len_block, **w)
        k = self._prefill.get(T_len)
        if k is None:
            k = cmix_prefill_kernel(
                self._C, self._DTYPE, self._cmix_len_block, **self._w_fp16
            )
            self._prefill[T_len] = k
        return k

    def __call__(self, x0, prev_x):
        return self.decode(x0, prev_x)

    def __repr__(self) -> str:
        return f"CmixOps(mode={self._mode!r})"


__all__ = ["CmixOps", "HeadOps", "TmixOps"]
