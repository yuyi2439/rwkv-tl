"""Fused batched GEMM kernels for prefill.

RKV GEMM: three [T, C] @ [C, C] matmuls (r/k/v projections) fused into a single
launch. Dispatch by device and GPU arch:

- CPU: stacked eager matmul (no CUDA dependency).
- CUDA sm_80+ (Ampere/Hopper/Blackwell): tilelang T.gemm kernel on TensorCore.
  fp16 MMA (m16n8k16, TransB layout) is supported on these archs, so the three
  matmuls run in a single kernel launch with blockIdx.z as the batch dim.
- CUDA sm_75 (Turing) and any GPU where the tilelang kernel is unavailable:
  torch.bmm -> cuBLAS strided batched GEMM (TensorCore, fp32 accumulate).
  sm_75 lacks the fp16 TransB MMA atom tilelang infers, so we fall back.

The weight tensor is pre-stacked once at model construction ([3, C, C]) and
passed in directly, so neither the tilelang nor the bmm path re-stacks per call.
A single compilation serves any RWKV7 model via T.dynamic shapes (C and T_LEN).
"""

# tilelang's @T.prim_func DSL uses call expressions (T.Tensor(...)) in type
# positions and tilelang-only intrinsics; pyright cannot type-check those.
# pyright: reportInvalidTypeForm=false, reportCallIssue=false, reportAttributeAccessIssue=false
from __future__ import annotations

import tilelang
import tilelang.language as T
import torch
from torch import Tensor


def _torch_bmm_rkv(xr: Tensor, xk: Tensor, xv: Tensor, Wb: Tensor) -> Tensor:
    """cuBLAS batched GEMM path (CUDA) or eager fallback (CPU).

    Stacks the three activations into a [3, T, C] batched tensor and calls
    torch.bmm with the pre-stacked [3, C, C] weights. On CUDA this dispatches
    to cuBLAS strided batched GEMM with fp32 accumulation, which is bit-exact
    with three separate mm calls but saves two kernel launches.
    """
    Xb = torch.stack([xr, xk, xv], dim=0)  # [3, T, C]
    return torch.bmm(Xb, Wb)  # [3, T, C]


def _gpu_supports_tl_fp16_gemm() -> bool:
    """Whether the current CUDA GPU supports the tilelang fp16 T.gemm path.

    sm_80+ (Ampere/Hopper/Blackwell) supports the m16n8k16 fp16 MMA atom with
    the TransB layout tilelang infers. sm_75 (Turing) only supports
    m16n8k8 with TransB=false, which tilelang does not emit, so the kernel
    fails to compile there and we must fall back to cuBLAS bmm.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    # sm_80+ required for fp16 m16n8k16 MMA with TransB.
    return (major, minor) >= (8, 0)


@tilelang.jit(out_idx=[4])
def _try_compile_tl_rkv(C: int):
    """Compile the tilelang T.gemm kernel for the current CUDA device and C.

    C is a model constant baked at compile time; T_LEN stays dynamic. Compiled
    lazily on first call (per-C, cached by tilelang's JIT). Only reached on
    sm_80+ where the fp16 TransB MMA is available; target arch is auto-detected
    from the current CUDA device (WGMMA on Hopper, TCGEN5MMA on Blackwell,
    Ampere MMA otherwise).
    """
    T_LEN = T.dynamic("T_LEN")
    # Block tile sizes. block_M=16 matches the MMA M atom on sm_80+; block_N=64
    # and block_K=32 keep shared memory under the per-block limit and give the
    # pipelined k-loop enough stages to hide global latency.
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 32

    @T.prim_func
    def _impl(
        xr: T.Tensor((T_LEN, C), "float16"),
        xk: T.Tensor((T_LEN, C), "float16"),
        xv: T.Tensor((T_LEN, C), "float16"),
        Wb: T.Tensor((3, C, C), "float16"),
        rkv: T.Tensor((3, T_LEN, C), "float16"),
    ):
        """Fused r/k/v GEMM: rkv[b] = X[b] @ Wb[b] for b in {0,1,2}.

        Grid: (ceildiv(C, BLOCK_N), ceildiv(T_LEN, BLOCK_M), 3).
        blockIdx.z selects the batch (0=r, 1=k, 2=v); the matching input and
        weight slice is chosen via a runtime branch inside the k-loop.
        """
        with T.Kernel(
            T.ceildiv(C, BLOCK_N),
            T.ceildiv(T_LEN, BLOCK_M),
            3,
            threads=128,
        ) as (bx, by, bz):
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(C, BLOCK_K), num_stages=3):
                if bz == 0:
                    T.copy(xr[by * BLOCK_M, k * BLOCK_K], A_shared)
                elif bz == 1:
                    T.copy(xk[by * BLOCK_M, k * BLOCK_K], A_shared)
                else:
                    T.copy(xv[by * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(Wb[bz, k * BLOCK_K, bx * BLOCK_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, rkv[bz, by * BLOCK_M, bx * BLOCK_N])

    return _impl


def fused_rkv_gemm(xr: Tensor, xk: Tensor, xv: Tensor, Wb: Tensor) -> Tensor:
    """Fused r/k/v projection GEMM for prefill.

    Computes r = xr @ Wb[0], k = xk @ Wb[1], v = xv @ Wb[2]. Returns a
    [3, T, C] tensor; slice [0]/[1]/[2] for r/k/v.

    Dispatch:
    - CPU: stacked eager matmul (no CUDA dependency).
    - CUDA sm_80+: tilelang T.gemm kernel on TensorCore (single launch for all
      three matmuls), compiled for the device's native arch.
    - CUDA sm_75 (and any GPU where the tilelang kernel is unavailable):
      torch.bmm -> cuBLAS strided batched GEMM (TensorCore, fp32 accumulate).

    Args:
        xr/xk/xv: LERP-interpolated activations, each [T, C], fp16.
        Wb: Pre-stacked transposed projection weights, [3, C, C], fp16.
            Wb[0]=rWt, Wb[1]=kWt, Wb[2]=vWt (each [C, C]).

    Returns:
        Stacked [3, T, C] tensor (rkv[0]=r, rkv[1]=k, rkv[2]=v), fp16.
    """
    if xr.device.type != "cuda":
        return _torch_bmm_rkv(xr, xk, xv, Wb)
    if not _gpu_supports_tl_fp16_gemm():
        return _torch_bmm_rkv(xr, xk, xv, Wb)
    try:
        return _try_compile_tl_rkv(xr.shape[1])(xr, xk, xv, Wb)
    except Exception:  # noqa: BLE001  (compile failure -> bmm fallback)
        return _torch_bmm_rkv(xr, xk, xv, Wb)
