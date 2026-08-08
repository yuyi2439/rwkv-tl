"""Fused batched GEMM kernels for prefill (dtype-parameterized).

RKV GEMM: three [T, C] @ [C, C] matmuls (r/k/v projections) fused into a single
launch. Dispatch by device and GPU arch:

- CPU: stacked eager matmul (no CUDA dependency).
- CUDA sm_80+ (Ampere/Hopper/Blackwell): tilelang T.gemm kernel on TensorCore.
  The m16n8k16 MMA atom with the TransB layout tilelang infers is supported on
  these archs for both fp16 and bf16.
- CUDA sm_75 (Turing): tilelang T.gemm for fp16 using the native m16n8k8 MMA
  atom (tuned config 16x32x32 avoids the pathological cuBLAS fp16 kernels,
  which are ~4-8x slower than fp32 on Turing). bf16 has no tensor-core MMA on
  sm_75, so bf16 falls back to torch.bmm -> cuBLAS (fast fp32 emulation).
- Any compile failure: torch.bmm -> cuBLAS strided batched GEMM.

FFN / output-projection matmuls (``ffn_h``/``ffn_v``/``out_mm``) have the same
sm_75 story: cuBLAS fp16 there is pathological (``volta_s884gemm_fp16_*``), so
fp16 routes them through the general tilelang m16n8k8 kernel with
shape-tuned block configs (autotuned on MX450); bf16 keeps plain matmul
(cuBLAS fp32 emulation) and Ampere+ keeps cuBLAS fp16.

``build(DTYPE)`` returns a namespace with the kernels bound to one element type.
"""

# tilelang's @T.prim_func DSL uses call expressions (T.Tensor(...)) in type
# positions and tilelang-only intrinsics; pyright cannot type-check those.
# pyright: reportInvalidTypeForm=false, reportCallIssue=false, reportAttributeAccessIssue=false
# NOTE: no `from __future__ import annotations` here -- tilelang's eager builder
# evaluates the annotation expressions, and a stringified annotation would lose
# the closure DTYPE param (NameError).

import warnings
from types import SimpleNamespace

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
from torch import Tensor

# Allowed T-specialized kernel lengths for the Turing sm_75 fp16 GEMM: exact
# 1..16 plus powers of two through 1,048,576. tilelang compiles each (C, T_len)
# lazily and caches it. Measured on MX450, the smallest covering kernel >= the
# actual T is always the fastest (kernel time scales ~linearly with T_len, so a
# larger kernel only adds pad waste), so inputs are padded up to the smallest
# allowed length >= T and the extra rows sliced off.
ALLOWED_T_LEN: tuple[int, ...] = (*range(1, 17), *(2**k for k in range(5, 20)))


def _smallest_covering_t_len(T: int) -> int:
    """Smallest allowed kernel length >= T (lower-bound binary search)."""
    lo, hi = 0, len(ALLOWED_T_LEN)
    while lo < hi:
        mid = (lo + hi) // 2
        if ALLOWED_T_LEN[mid] < T:
            lo = mid + 1
        else:
            hi = mid
    return ALLOWED_T_LEN[lo]


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

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[DTYPE]

    def _gpu_supports_tl_gemm(DTYPE: str) -> bool:
        """Whether the current CUDA GPU supports the tilelang T.gemm path.

        sm_80+ (Ampere/Hopper/Blackwell) supports the m16n8k16 MMA atom with the
        TransB layout tilelang infers, for both fp16 and bf16. sm_75 (Turing)
        supports fp16 via the native m16n8k8 MMA atom only; bf16 has no MMA on
        sm_75, so bf16 stays on the cuBLAS bmm path (fast fp32 emulation there).
        """
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) >= (8, 0):
            return True
        return DTYPE == "float16"

    @tilelang.jit(out_idx=[4])
    def _try_compile_tl_rkv(C: int):
        """Compile the dynamic-T tilelang T.gemm kernel (sm_80+ Ampere path)."""
        T_LEN = T.dynamic("T_LEN")
        BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES = 16, 64, 32, 3

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
                for kk in T.Pipelined(T.ceildiv(C, BLOCK_K), num_stages=NUM_STAGES):
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

    @tilelang.jit(out_idx=[4])
    def _try_compile_tl_rkv_T(C: int, T_len: int):
        """Compile a T-specialized tilelang fp16 GEMM for Turing sm_75.

        m16n8k8 is the native Turing fp16 MMA atom. Specializing T_len (instead
        of a dynamic dimension) lets tilelang hoist loop bounds and pipeline
        aggressively -- the 16x32x32/3-stage config autotuned on MX450 runs
        ~4-7x faster than the pathological cuBLAS fp16 kernels and beats fp32
        at small T. Compiled once per (C, T) and cached by tilelang.
        """
        BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES = 16, 32, 32, 3

        @T.prim_func
        def _impl(
            xr: T.Tensor((T_len, C), DTYPE),
            xk: T.Tensor((T_len, C), DTYPE),
            xv: T.Tensor((T_len, C), DTYPE),
            Wb: T.Tensor((3, C, C), DTYPE),
            rkv: T.Tensor((3, T_len, C), DTYPE),
        ):
            """Fused r/k/v GEMM: rkv[b] = X[b] @ Wb[b] for b in {0,1,2}."""
            with T.Kernel(
                T.ceildiv(C, BLOCK_N),
                T.ceildiv(T_len, BLOCK_M),
                3,
                threads=128,
            ) as (bx, by, bz):
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), DTYPE)
                B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), DTYPE)
                C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
                T.clear(C_local)
                for kk in T.Pipelined(T.ceildiv(C, BLOCK_K), num_stages=NUM_STAGES):
                    if bz == 0:
                        T.copy(xr[by * BLOCK_M, kk * BLOCK_K], A_shared)
                    elif bz == 1:
                        T.copy(xk[by * BLOCK_M, kk * BLOCK_K], A_shared)
                    else:  # bz == 2
                        T.copy(xv[by * BLOCK_M, kk * BLOCK_K], A_shared)

                    T.copy(Wb[bz, kk * BLOCK_K, bx * BLOCK_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, rkv[bz, by * BLOCK_M, bx * BLOCK_N])

        return _impl

    @tilelang.jit(out_idx=[2])
    def _try_compile_tl_mm(K: int, N: int, block_n: int, block_k: int, num_stages: int):
        """General fp16 ``[M, K] @ [K, N]`` tilelang GEMM for Turing sm_75.

        Routes the FFN / output-projection matmuls away from the pathological
        cuBLAS fp16 kernels on Turing (``volta_s884gemm_fp16_*``), which for the
        small prefill shapes are ~4-8x slower than fp32 / bf16. The
        ``(block_n, block_k, num_stages)`` triplets were autotuned on MX450 for
        the ``[T, C] @ [C, 4C]`` (ffn_h: 64, 32, 3) and ``[T, 4C] @ [4C, C]``
        (ffn_v: 128, 64, 3) shapes; a naive 16x32x32 config is ~2-4x slower.
        Compiled once per (K, N) and cached by tilelang.
        """
        M_T = T.dynamic("M_T")

        @T.prim_func
        def _impl(
            A: T.Tensor((M_T, K), DTYPE),
            B: T.Tensor((K, N), DTYPE),
            C_out: T.Tensor((M_T, N), DTYPE),
        ):
            """A @ B -> C_out (both operands in ``[K, N]`` B-layout)."""
            with T.Kernel(
                T.ceildiv(N, block_n),
                T.ceildiv(M_T, 16),
                threads=128,
            ) as (bx, by):
                A_shared = T.alloc_shared((16, block_k), DTYPE)
                B_shared = T.alloc_shared((block_k, block_n), DTYPE)
                C_local = T.alloc_fragment((16, block_n), "float32")
                T.clear(C_local)
                for kk in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                    T.copy(A[by * 16, kk * block_k], A_shared)
                    T.copy(B[kk * block_k, bx * block_n], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C_out[by * 16, bx * block_n])

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
        # tilelang kernels are bound to one element type; the bmm path also
        # serves other input dtypes (e.g. RWKV7MX450 passes fp32 activations
        # for the fast fp32 cuBLAS bmm on Turing).
        if not _gpu_supports_tl_gemm(DTYPE) or xr.dtype != torch_dtype:
            return _torch_bmm_rkv(xr, xk, xv, Wb)
        try:
            major, minor = torch.cuda.get_device_capability()
            if (major, minor) < (8, 0):
                # Turing: T-specialized m16n8k8 kernel (compiled lazily and
                # cached per (C, T_len) by tilelang). A dynamic-T version cannot
                # reach the tuned config's speed on sm_75, so we pay one compile
                # per distinct length. Inputs are padded up to the smallest
                # allowed length >= T (binary search) and sliced back -- measured
                # fastest on MX450.
                T_len = xr.shape[0]
                if T_len > ALLOWED_T_LEN[-1]:
                    warnings.warn(
                        f"T_len={T_len} exceeds max specialized kernel length "
                        f"{ALLOWED_T_LEN[-1]}; falling back to cuBLAS bmm on sm_75"
                        "The length is longer than the ctxlen of rwkv7 :(",
                        stacklevel=2,
                    )
                    return _torch_bmm_rkv(xr, xk, xv, Wb)
                k_t = _smallest_covering_t_len(T_len)
                kernel = _try_compile_tl_rkv_T(xr.shape[1], k_t)
                if k_t == T_len:
                    return kernel(xr, xk, xv, Wb)
                pad = k_t - T_len
                xr = F.pad(xr, (0, 0, 0, pad))
                xk = F.pad(xk, (0, 0, 0, pad))
                xv = F.pad(xv, (0, 0, 0, pad))
                out = kernel(xr, xk, xv, Wb)
                return out[:, :T_len, :]
            return _try_compile_tl_rkv(xr.shape[1])(xr, xk, xv, Wb)
        except Exception:  # noqa: BLE001  (compile failure -> bmm fallback)
            return _torch_bmm_rkv(xr, xk, xv, Wb)

    def _use_tl_mm() -> bool:
        """Use the tilelang fp16 GEMM for FFN / output-projection matmuls.

        Only on Turing sm_75 + fp16: cuBLAS fp16 is pathologically slow there
        (``volta_s884gemm_fp16_*``). Ampere+ cuBLAS fp16 is fine, and bf16
        cuBLAS already uses the fast fp32 emulation.
        """
        if DTYPE != "float16" or not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) < (8, 0)

    def _tl_mm(a: Tensor, b: Tensor, block_n: int, block_k: int, num_stages: int) -> Tensor:
        """Run ``a @ b`` through the tilelang fp16 kernel (Turing only)."""
        kernel = _try_compile_tl_mm(b.shape[0], b.shape[1], block_n, block_k, num_stages)
        return kernel(a, b)

    def _tl_mm_or_none(
        a: Tensor, b: Tensor, block_n: int, block_k: int, num_stages: int
    ) -> Tensor | None:
        """Turing fp16 tilelang matmul, or ``None`` when unavailable/failed."""
        if not _use_tl_mm():
            return None
        try:
            return _tl_mm(a, b, block_n, block_k, num_stages)
        except Exception:  # noqa: BLE001 - compile failure -> fall back to cuBLAS
            return None

    def ffn_h(x: Tensor, kWt: Tensor) -> Tensor:
        """FFN first projection: ``h = x @ kWt``, ``[T, C] @ [C, 4C] -> [T, 4C]``."""
        out = _tl_mm_or_none(x, kWt, 64, 32, 3)
        return x @ kWt if out is None else out

    def ffn_v(h: Tensor, vWt: Tensor) -> Tensor:
        """FFN second projection: ``out = h @ vWt``, ``[T, 4C] @ [4C, C] -> [T, C]``."""
        out = _tl_mm_or_none(h, vWt, 128, 64, 3)
        return h @ vWt if out is None else out

    def out_mm(y: Tensor, oWt: Tensor) -> Tensor:
        """Output projection: ``y @ oWt``, ``[T, C] @ [C, C] -> [T, C]``."""
        out = _tl_mm_or_none(y, oWt, 64, 32, 3)
        return y @ oWt if out is None else out

    return SimpleNamespace(
        fused_rkv_gemm=fused_rkv_gemm,
        ffn_h=ffn_h,
        ffn_v=ffn_v,
        out_mm=out_mm,
    )
