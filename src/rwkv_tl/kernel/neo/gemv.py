# pyright: reportInvalidTypeForm=false

import tilelang
import tilelang.language as T

from .._common import WARP


def gemv_macro(M: int, K: int, DTYPE: str, THREADS: int = WARP, epilogue=None):
    """GEMV ``out[m] = sum_k x[k] * W[k, m]`` (i.e. ``x @ W``), lane-per-output.

    Args:
        M: Output length (columns of ``W``). Must be a multiple of THREADS.
        K: Reduction length (rows of ``W``, size of ``x``).
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: Threads per block, each owning one output. Measured on MX450,
            WARP (32) is best or tied for every decode shape; larger blocks cut
            launch overhead but shrink the block count below one per SM on a
            small GPU, and this memory-bound kernel loses more to occupancy than
            it gains. Tune per shape/GPU.
        epilogue: Optional Python callable ``(acc, index) -> PrimExpr``
            applied to each fp32 accumulator before the store, for fusing
            elementwise epilogues (e.g. relusq) or a residual add. ``None``
            stores the plain accumulator. The callable is a plain Python
            closure built by the caller, so it can capture extra tensors
            (e.g. an ``x0`` residual) without gemv knowing about them.
    """
    assert M % THREADS == 0

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
                ``W[k, :]`` rows coalesced (Albatross's decode-GEMV structure),
                so the ``[K, M]`` orientation needs no extra transposed copy.
            out: Output vector ``[M]``.
        """
        acc = T.alloc_fragment((THREADS,), "float32")
        T.clear(acc)
        for k in T.serial(K):
            for i in T.Parallel(THREADS):
                acc[i] += T.cast(x[k], "float32") * T.cast(W[k, bx * THREADS + i], "float32")
        for i in T.Parallel(THREADS):
            idx = bx * THREADS + i
            val = acc[i]
            out_val = epilogue(val, idx) if epilogue is not None else val
            out[idx] = T.cast(out_val, DTYPE)

    return _impl


@tilelang.jit(out_idx=[2])
def gemv(M: int, K: int, DTYPE: str, THREADS: int = WARP):
    gemv_m = gemv_macro(M, K, DTYPE, THREADS)

    @T.prim_func
    def _impl(
        x: T.Tensor((K,), DTYPE),
        W: T.Tensor((K, M), DTYPE),
        out: T.Tensor((M,), DTYPE),
    ):
        with T.Kernel(M // THREADS, threads=THREADS) as bx:
            gemv_m(bx, x, W, out=out)

    return _impl
