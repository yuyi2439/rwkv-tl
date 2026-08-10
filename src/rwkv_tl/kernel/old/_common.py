"""Shared constants for tilelang fused kernels.

RWKV7 models share a fixed head dimension (N=64); the embedding width C and
head count H vary by model size (e.g. 0.1B: C=768,H=12; 0.4B: C=1024,H=16).
H and C are fixed at compile time (captured as closure variables in each
@tilelang.jit kernel); distinct model sizes compile separate kernels, cached
automatically by @tilelang.jit. Only T_LEN / M_T use T.dynamic so one kernel
serves any sequence length.
"""

from __future__ import annotations

HEAD_DIM = 64  # N: per-head dimension, constant across RWKV7
BLOCK = 256  # threads per block for elementwise kernels
# Logical warp width for tilelang's `warp_reduce_sum`. tilelang lowers this to
# 32-lane shuffles on BOTH NVIDIA and AMD (see src/tl_templates/{cuda,hip}/reduce.h;
# the HIP backend intentionally keeps 32-lane logical-warp semantics on CDNA
# wave64). So 32 is correct on every backend, and the [H, N] reduction kernels
# run one logical warp per block with 32 threads. Do NOT set this to the AMD
# hardware wavefront (64): tilelang's reduce would then cover only lanes 0-31
# and silently drop half the reduction. If tilelang ever adds a true wave64
# reduce, bump WARP here and the kernels' `threads=`.
WARP = 32
SERIAL = HEAD_DIM // WARP  # 2: elements processed per thread in [H, N] kernels
