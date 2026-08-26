# pyright: reportInvalidTypeForm=false

import tilelang
import tilelang.language as T

from ._op import KernelOp
from ._common import WARP


def gemv_main_macro(M: int, K: int, DTYPE: str, THREADS: int = WARP):
    """GEMV compute: ``acc[i] = sum_k x[k] * W[k, bx*THREADS + i]``.

    Lane-per-output reduction into a fp32 fragment accumulator; returns the
    ``acc`` fragment so the caller owns the store. Callers with a custom
    epilogue (e.g. ``relusq`` or a residual add) post-process ``acc`` in place
    before storing; callers without one should use ``gemv_macro``.

    Args:
        M: Output length (columns of ``W``). Must be a multiple of THREADS.
        K: Reduction length (rows of ``W``, size of ``x``).
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: Threads per block, each owning one output.
    """
    assert M % THREADS == 0

    @T.macro
    def _impl(
        bx,
        x: T.Tensor((K,), DTYPE),
        W: T.Tensor((K, M), DTYPE),
    ):
        """Args:
        bx: Output-block index; this block computes
            ``acc[i] = sum_k x[k] * W[k, bx*THREADS + i]``.
        x: Input vector ``[K]``.
        W: Weight, ``[K, M]`` row-major = the project's ``*t`` ``[in, out]``
            layout (``weight.py``), consumed directly as ``x @ W`` with no
            transpose. Each lane owns one output column and reads the
            ``W[k, :]`` rows coalesced, so the ``[K, M]`` orientation needs no
            extra transposed copy.
        """
        acc = T.alloc_fragment((THREADS,), "float32")
        T.clear(acc)
        for k in T.serial(K):
            for i in T.Parallel(THREADS):
                acc[i] += T.cast(x[k], "float32") * T.cast(
                    W[k, bx * THREADS + i], "float32"
                )

        return acc

    return _impl


def gemv_macro(M: int, K: int, DTYPE: str, THREADS: int = WARP):
    """GEMV ``out[m] = sum_k x[k] * W[k, m]`` (i.e. ``x @ W``), lane-per-output.

    Plain-store wrapper over ``gemv_main_macro``: computes ``acc`` and casts it
    straight to ``out``. Callers that need to fuse an epilogue into the store
    should call ``gemv_main_macro`` instead and post-process ``acc``.

    Args:
        M: Output length (columns of ``W``). Must be a multiple of THREADS.
        K: Reduction length (rows of ``W``, size of ``x``).
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: Threads per block, each owning one output.
    """
    assert M % THREADS == 0

    main = gemv_main_macro(M, K, DTYPE, THREADS)

    @T.macro
    def _impl(
        bx,
        x: T.Tensor((K,), DTYPE),
        W: T.Tensor((K, M), DTYPE),
        *,
        out: T.Tensor((M,), DTYPE),
    ):
        """Args:
        bx: Output-block index; this block computes
            ``out[bx*THREADS : (bx+1)*THREADS]``.
        x: Input vector ``[K]``.
        W: Weight, ``[K, M]`` row-major = the project's ``*t`` ``[in, out]``
            layout (``weight.py``), consumed directly as ``x @ W`` with no
            transpose. Each lane owns one output column and reads the
            ``W[k, :]`` rows coalesced, so the ``[K, M]`` orientation needs no
            extra transposed copy.
        """

        acc = main(bx, x, W)

        for i in T.Parallel(THREADS):
            idx = bx * THREADS + i
            out[idx] = T.cast(acc[i], DTYPE)

    return _impl


def gemv_batch_macro(M: int, K: int, B: int, DTYPE: str, THREADS: int = WARP):
    """Batched GEMV ``out[b, m] = sum_k x[b, k] * W[b, k, m]`` (``x[b] @ W[b]``).

    Fuses ``B`` independent GEMVs into a single launch using a 2D grid
    ``(M // THREADS, B)``: block ``(bx, bz)`` computes
    ``out[bz, bx*THREADS : (bx+1)*THREADS]`` from the ``bz``-th stacked
    member. Used to share one launch across e.g. the r/k/v projections.

    Args:
        M: Output length (columns of each ``W[b]``). Multiple of THREADS.
        K: Reduction length (rows of each ``W[b]``, size of each ``x[b]``).
        B: Stacked batch count.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: Threads per block, each owning one output column.
    """
    assert M % THREADS == 0

    @T.macro
    def _impl(
        bx,
        bz,
        x: T.Tensor((B, K), DTYPE),
        W: T.Tensor((B, K, M), DTYPE),
        *,
        out: T.Tensor((B, M), DTYPE),
    ):
        """Args:
        bx: Output-block index; this block computes
            ``out[bz, bx*THREADS : (bx+1)*THREADS]``.
        bz: Stacked-batch index (blockIdx.y); selects the ``bz``-th GEMV.
        x: Stacked inputs ``[B, K]``.
        W: Stacked weights ``[B, K, M]`` row-major (the ``*t`` layout).
        out: Stacked outputs ``[B, M]``.
        """
        acc = T.alloc_fragment((THREADS,), "float32")
        T.clear(acc)
        for k in T.serial(K):
            for i in T.Parallel(THREADS):
                acc[i] += T.cast(x[bz, k], "float32") * T.cast(
                    W[bz, k, bx * THREADS + i], "float32"
                )
        for i in T.Parallel(THREADS):
            out[bz, bx * THREADS + i] = T.cast(acc[i], DTYPE)

    return _impl


def gemv_batch_T_macro(
    T_len: int, M: int, K: int, B: int, DTYPE: str, THREADS: int = WARP
):
    """Batched GEMV ``out[t, b, m] = sum_k x[t, b, k] * W[b, k, m]``.

    Fuses ``B`` independent GEMVs per row into a single launch using a 3D grid
    ``(M // THREADS, B, T_len)``: block ``(bx, bz, bt)`` computes
    ``out[bt, bz, bx*THREADS : (bx+1)*THREADS]``. Uses the same fp32-accumulating
    lane-per-output reduction as ``gemv_macro`` (not tensor-core MMA), so
    decode and prefill share one r/k/v projection path.

    Args:
        T_len: Row count (sequence length for prefill).
        M: Output length (columns of each ``W[b]``). Multiple of THREADS.
        K: Reduction length (rows of each ``W[b]``, size of each ``x[t, b]``).
        B: Stacked batch count.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: Threads per block, each owning one output column.
    """
    assert M % THREADS == 0

    @T.macro
    def _impl(
        bx,
        bz,
        bt,
        x: T.Tensor((T_len, B, K), DTYPE),
        W: T.Tensor((B, K, M), DTYPE),
        *,
        out: T.Tensor((T_len, B, M), DTYPE),
    ):
        """Args:
        bx: Output-block index; this block computes
            ``out[bt, bz, bx*THREADS : (bx+1)*THREADS]``.
        bz: Stacked-batch index (blockIdx.y); selects the ``bz``-th GEMV.
        bt: Row index (blockIdx.z); selects the ``bt``-th row's GEMVs.
        x: Stacked inputs ``[T_len, B, K]``.
        W: Stacked weights ``[B, K, M]`` row-major (the ``*t`` layout).
        out: Stacked outputs ``[T_len, B, M]``.
        """
        acc = T.alloc_fragment((THREADS,), "float32")
        T.clear(acc)
        for k in T.serial(K):
            for i in T.Parallel(THREADS):
                acc[i] += T.cast(x[bt, bz, k], "float32") * T.cast(
                    W[bz, k, bx * THREADS + i], "float32"
                )
        for i in T.Parallel(THREADS):
            out[bt, bz, bx * THREADS + i] = T.cast(acc[i], DTYPE)

    return _impl


@tilelang.jit(out_idx=[2])
def gemv_jit(M: int, K: int, DTYPE: str, THREADS: int = WARP):
    """GEMV kernel: ``out = x @ W`` (``x [K]``, ``W [K, M]``).

    Args:
        M: Output length (columns of ``W``). Must be a multiple of THREADS.
        K: Reduction length (rows of ``W``, size of ``x``).
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: Threads per block.
    """
    main = gemv_main_macro(M, K, DTYPE, THREADS)

    @T.prim_func
    def _impl(
        x: T.Tensor((K,), DTYPE),
        W: T.Tensor((K, M), DTYPE),
        out: T.Tensor((M,), DTYPE),
    ):
        with T.Kernel(M // THREADS, threads=THREADS) as bx:
            acc = main(bx, x, W)
            for i in T.Parallel(THREADS):
                out[bx * THREADS + i] = T.cast(acc[i], DTYPE)

    return _impl


@tilelang.jit(out_idx=[2])
def gemv_batch_jit(M: int, K: int, B: int, DTYPE: str, THREADS: int = WARP):
    """Batched GEMV kernel: ``out[b] = x[b] @ W[b]``.

    Args:
        M: Output length (columns of each ``W[b]``). Multiple of THREADS.
        K: Reduction length (rows of each ``W[b]``, size of each ``x[b]``).
        B: Stacked batch count.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: Threads per block.
    """
    main = gemv_batch_macro(M, K, B, DTYPE, THREADS)

    @T.prim_func
    def _impl(
        x: T.Tensor((B, K), DTYPE),
        W: T.Tensor((B, K, M), DTYPE),
        out: T.Tensor((B, M), DTYPE),
    ):
        with T.Kernel(M // THREADS, B, threads=THREADS) as (bx, bz):
            main(bx, bz, x, W, out=out)

    return _impl


def gemv_kernel(M: int, K: int, DTYPE: str, W):
    """Bound GEMV: ``g = gemv_kernel(M, K, DTYPE, W)`` then ``y = g(x)``.

    ``W`` is the project's row-major ``[K, M]`` layout (``x @ W``); it is
    captured at construction.
    """
    return KernelOp(
        gemv_jit,
        (M, K, DTYPE),
        bind={"W": W},
        call=("x",),
        name="gemv_kernel",
    )


def gemv_batch_kernel(M: int, K: int, B: int, DTYPE: str, W):
    """Bound batched GEMV (see ``gemv_kernel``); ``W`` is ``[B, K, M]``."""
    return KernelOp(
        gemv_batch_jit,
        (M, K, B, DTYPE),
        bind={"W": W},
        call=("x",),
        name="gemv_batch_kernel",
    )


@tilelang.jit(out_idx=[3])
def gemv_q8_jit(M: int, K: int, G: int, DTYPE: str, THREADS: int = 64):
    """W8A16 GEMV: ``out = x @ dequant(Wq, s)``.

    Each lane owns 4 consecutive int8 columns, so a warp reads a full
    128-byte sector per reduction step (vs 32B with one column per lane).

    Args:
        M: Output length (columns of ``Wq``); multiple of ``THREADS * 4``.
        K: Reduction length; multiple of ``G``.
        G: Quantization group size along K.
        DTYPE: Activation/output element type.
        THREADS: Threads per block.
    """
    VEC = 4

    @T.prim_func
    def _impl(
        x: T.Tensor((K,), DTYPE),
        Wq: T.Tensor((K, M), "int8"),
        s: T.Tensor((K // G, M), "float16"),
        out: T.Tensor((M,), DTYPE),
    ):
        with T.Kernel(M // (THREADS * VEC), threads=THREADS) as bx:
            acc = T.alloc_fragment((THREADS * VEC,), "float32")
            T.clear(acc)
            for k in T.serial(K):
                for i in T.Parallel(THREADS * VEC):
                    col = bx * THREADS * VEC + i
                    acc[i] += T.cast(x[k], "float32") * (
                        T.cast(Wq[k, col], "float32")
                        * T.cast(s[k // G, col], "float32")
                    )
            for i in T.Parallel(THREADS * VEC):
                out[bx * THREADS * VEC + i] = T.cast(acc[i], DTYPE)

    return _impl


def gemv_q8_kernel(M: int, K: int, G: int, DTYPE: str, Wq, s):
    """Bound W8A16 GEMV (see ``gemv_q8_jit``); weights captured at construction.

    ``Wq`` is the int8 payload ``[K, M]`` and ``s`` the fp16 scales
    ``[K // G, M]`` (per-group, per-column) of a ``QTensor``.
    """
    return KernelOp(
        gemv_q8_jit,
        (M, K, G, DTYPE),
        bind={"Wq": Wq, "s": s},
        call=("x",),
        name="gemv_q8_kernel",
    )


# --- W8A16 (int8 weight + fp16 scale) GEMV macros ---------------------------
# Lane-per-output copies of the fp16 macros above: each reads the int8 payload
# ``Wq`` and dequantizes against the per-group-per-column fp16 scale ``s``.
# These exist so a fused decode kernel can keep its layer weight int8-resident
# (VRAM) while still dequantizing per access; they mirror the fp16 grid layout
# exactly (no speed change expected -- the decode GEMVs are occupancy-bound).


def gemv_main_q8_macro(M: int, K: int, G: int, DTYPE: str, THREADS: int = WARP):
    """W8A16 ``acc[i] = sum_k x[k] * (Wq[k,col] * s[k//G,col])``; returns acc.

    fp32-accumulating lane-per-output reduction mirroring ``gemv_main_macro``,
    consumed by callers that fuse an epilogue (e.g. cmix relusq).
    """
    assert M % THREADS == 0 and K % G == 0

    @T.macro
    def _impl(
        bx,
        x: T.Tensor((K,), DTYPE),
        Wq: T.Tensor((K, M), "int8"),
        s: T.Tensor((K // G, M), "float16"),
    ):
        acc = T.alloc_fragment((THREADS,), "float32")
        T.clear(acc)
        for k in T.serial(K):
            for i in T.Parallel(THREADS):
                col = bx * THREADS + i
                acc[i] += T.cast(x[k], "float32") * (
                    T.cast(Wq[k, col], "float32") * T.cast(s[k // G, col], "float32")
                )
        return acc

    return _impl


def gemv_q8_macro(M: int, K: int, G: int, DTYPE: str, THREADS: int = WARP):
    """W8A16 GEMV store wrapper over ``gemv_main_q8_macro``."""

    main = gemv_main_q8_macro(M, K, G, DTYPE, THREADS)

    @T.macro
    def _impl(
        bx,
        x: T.Tensor((K,), DTYPE),
        Wq: T.Tensor((K, M), "int8"),
        s: T.Tensor((K // G, M), "float16"),
        *,
        out: T.Tensor((M,), DTYPE),
    ):
        acc = main(bx, x, Wq, s)
        for i in T.Parallel(THREADS):
            idx = bx * THREADS + i
            out[idx] = T.cast(acc[i], DTYPE)

    return _impl


def gemv_batch_q8_macro(M: int, K: int, B: int, G: int, DTYPE: str, THREADS: int = WARP):
    """W8A16 batched GEMV ``out[b,m] = sum_k x[b,k] * (Wq[b,k,m] * s[b,k//G,m])``."""

    assert M % THREADS == 0 and K % G == 0

    @T.macro
    def _impl(
        bx,
        bz,
        x: T.Tensor((B, K), DTYPE),
        Wq: T.Tensor((B, K, M), "int8"),
        s: T.Tensor((B, K // G, M), "float16"),
        *,
        out: T.Tensor((B, M), DTYPE),
    ):
        acc = T.alloc_fragment((THREADS,), "float32")
        T.clear(acc)
        for k in T.serial(K):
            for i in T.Parallel(THREADS):
                col = bx * THREADS + i
                acc[i] += T.cast(x[bz, k], "float32") * (
                    T.cast(Wq[bz, k, col], "float32")
                    * T.cast(s[bz, k // G, col], "float32")
                )
        for i in T.Parallel(THREADS):
            out[bz, bx * THREADS + i] = T.cast(acc[i], DTYPE)

    return _impl
