"""Fused batched GEMM kernels for prefill (dtype-parameterized).

RKV GEMM: three [T, C] @ [C, C] matmuls (r/k/v projections) fused into a single
launch. Dispatch by device and GPU arch:

- CPU: stacked eager matmul (no CUDA dependency).
- CUDA sm_80+ (Ampere/Hopper/Blackwell): tilelang T.gemm kernel on TensorCore.
  The m16n8k16 MMA atom with the TransB layout tilelang infers is supported on
  these archs for both fp16 and bf16.
- CUDA sm_75 (Turing) and any GPU where the tilelang kernel is unavailable:
  torch.bmm -> cuBLAS strided batched GEMM (TensorCore, fp32 accumulate).
  sm_75 lacks the TransB MMA atom tilelang infers, so we fall back.

``build(DTYPE)`` returns a namespace with the kernels bound to one element type.
"""

# tilelang's @T.prim_func DSL uses call expressions (T.Tensor(...)) in type
# positions and tilelang-only intrinsics; pyright cannot type-check those.
# pyright: reportInvalidTypeForm=false, reportCallIssue=false, reportAttributeAccessIssue=false
# NOTE: no `from __future__ import annotations` here -- tilelang's eager builder
# evaluates the annotation expressions, and a stringified annotation would lose
# the closure DTYPE param (NameError).

from types import SimpleNamespace

import tilelang
import tilelang.language as T
import torch
from torch import Tensor


def build(DTYPE: str) -> SimpleNamespace:
    """Build the dtype-bound r/k/v GEMM namespace.

    Args:
        DTYPE: Element type string, ``"float16"`` or ``"bfloat16"``.

    Returns:
        ``SimpleNamespace(fused_rkv_gemm=...)``.
    """

    def _torch_bmm_rkv(xr: Tensor, xk: Tensor, xv: Tensor, Wb: Tensor) -> Tensor:
        """cuBLAS batched GEMM path (CUDA) or eager fallback (CPU)."""
        Xb = torch.stack([xr, xk, xv], dim=0)  # [3, T, C]
        return torch.bmm(Xb, Wb)  # [3, T, C]

    def _gpu_supports_tl_gemm() -> bool:
        """Whether the current CUDA GPU supports the tilelang T.gemm path.

        sm_80+ (Ampere/Hopper/Blackwell) supports the m16n8k16 MMA atom with the
        TransB layout tilelang infers, for both fp16 and bf16. sm_75 (Turing)
        only supports m16n8k8 with TransB=false, which tilelang does not emit,
        so we fall back to cuBLAS bmm there.
        """
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) >= (8, 0)

    @tilelang.jit(out_idx=[4])
    def _try_compile_tl_rkv(C: int):
        """Compile the tilelang T.gemm kernel for the current device and C."""
        T_LEN = T.dynamic("T_LEN")
        BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 32

        @T.prim_func
        def _impl(
            xr: T.Tensor((T_LEN, C), DTYPE),
            xk: T.Tensor((T_LEN, C), DTYPE),
            xv: T.Tensor((T_LEN, C), DTYPE),
            Wb: T.Tensor((3, C, C), DTYPE),
            rkv: T.Tensor((3, T_LEN, C), DTYPE),
        ):
            """Fused r/k/v GEMM: rkv[b] = X[b] @ Wb[b] for b in {0,1,2}."""
            with T.Kernel(
                T.ceildiv(C, BLOCK_N),
                T.ceildiv(T_LEN, BLOCK_M),
                3,
                threads=128,
            ) as (bx, by, bz):
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), DTYPE)
                B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), DTYPE)
                C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
                T.clear(C_local)
                for kk in T.Pipelined(T.ceildiv(C, BLOCK_K), num_stages=3):
                    if bz == 0:
                        T.copy(xr[by * BLOCK_M, kk * BLOCK_K], A_shared)
                    elif bz == 1:
                        T.copy(xk[by * BLOCK_M, kk * BLOCK_K], A_shared)
                    else:
                        T.copy(xv[by * BLOCK_M, kk * BLOCK_K], A_shared)
                    T.copy(Wb[bz, kk * BLOCK_K, bx * BLOCK_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, rkv[bz, by * BLOCK_M, bx * BLOCK_N])

        return _impl

    def fused_rkv_gemm(xr: Tensor, xk: Tensor, xv: Tensor, Wb: Tensor) -> Tensor:
        """Fused r/k/v projection GEMM: rkv[b] = X[b] @ Wb[b].

        Args:
            xr/xk/xv: LERP-interpolated activations, each [T, C], DTYPE.
            Wb: Pre-stacked transposed projection weights, [3, C, C], DTYPE.

        Returns:
            Stacked [3, T, C] tensor (rkv[0]=r, rkv[1]=k, rkv[2]=v), DTYPE.
        """
        if xr.device.type != "cuda":
            return _torch_bmm_rkv(xr, xk, xv, Wb)
        if not _gpu_supports_tl_gemm():
            return _torch_bmm_rkv(xr, xk, xv, Wb)
        try:
            return _try_compile_tl_rkv(xr.shape[1])(xr, xk, xv, Wb)
        except Exception:  # noqa: BLE001  (compile failure -> bmm fallback)
            return _torch_bmm_rkv(xr, xk, xv, Wb)

    return SimpleNamespace(fused_rkv_gemm=fused_rkv_gemm)
