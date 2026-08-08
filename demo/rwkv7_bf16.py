"""BF16 RWKV7 model (tilelang kernels bound to ``bfloat16``).

Keeps the raw checkpoint dtype (load with ``RWKV7Weight(path,
dtype=torch.bfloat16)``, no conversion). bf16's 7-bit mantissa and the lack
of native bf16 tensor cores on sm_75 make this a reference/experimental
variant; fp16 (``RWKV7FP16``) is the default precision.
"""

from __future__ import annotations

from rwkv_tl.weight import RWKV7Weight

from ._rwkv7_base import RWKV7Base

__all__ = ["RWKV7BF16"]


class RWKV7BF16(RWKV7Base):
    def __init__(
        self,
        w: RWKV7Weight,
        *,
        is_torch_compile: bool = True,
    ) -> None:
        from rwkv_tl.kernel import bf16

        super().__init__(w, bf16.kernels, is_torch_compile=is_torch_compile)
