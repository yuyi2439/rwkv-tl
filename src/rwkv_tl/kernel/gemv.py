# pyright: reportInvalidTypeForm=false

import tilelang.language as T

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


def gemv_batch_T_macro(T_len: int, M: int, K: int, B: int, DTYPE: str, THREADS: int = WARP):
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
