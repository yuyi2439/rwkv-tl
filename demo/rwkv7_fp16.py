"""FP16 RWKV7 model (tilelang kernels bound to ``float16``).

Target: Ampere+ sm_80+ with native fp16 tensor cores. Weights are loaded
bf16->fp16 once at ``RWKV7Weight`` load (10-bit mantissa beats bf16's 7-bit).
"""

from __future__ import annotations

from rwkv_tl.weight import RWKV7Weight

from ._rwkv7_base import RWKV7Base

__all__ = ["RWKV7FP16"]


class RWKV7FP16(RWKV7Base):
    def __init__(
        self,
        w: RWKV7Weight,
        *,
        is_torch_compile: bool = True,
    ) -> None:
        from rwkv_tl.kernel import fp16

        super().__init__(w, fp16.kernels, is_torch_compile=is_torch_compile)
