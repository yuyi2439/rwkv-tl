"""Shared constants for tilelang fused kernels.

RWKV7 models share a fixed head dimension (N=64); the embedding width C and
head count H vary by model size (e.g. 0.1B: C=768,H=12; 0.4B: C=1024,H=16).
Reduction kernels use T.dynamic for H and elementwise kernels for C, so a
single compilation serves any RWKV7 model.
"""
from __future__ import annotations

import math

HEAD_DIM = 64                  # N: per-head dimension, constant across RWKV7
BLOCK = 256                    # threads per block for elementwise kernels
WARP = 32                      # warp width (matches warp_reduce_sum)
SERIAL = HEAD_DIM // WARP      # 2: elements processed per thread in [H, N] kernels
_SQRT_E = math.sqrt(math.e)    # exp decay gate constant
