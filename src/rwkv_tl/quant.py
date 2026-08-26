"""W8A16 weight-only quantization: per-group int8 storage + fp16 scales.

A ``QTensor`` replaces a large ``[in, out]`` fp16 weight. The int8 payload
``q`` keeps the SAME layout the kernels already use (``[K, M]``, reduction
dim first), so a W8A16 GEMV reads it exactly like the fp16 weight -- only
with a dequant multiply by the per-group scale. Scales are per group AND per
output column (``[K//G, M]`` fp16): far tighter than a per-group scalar at
negligible memory cost, and still contiguous along ``M`` inside the kernel.

Quantization groups run along ``K`` (the reduction dim), matching the decode
GEMV loop: the scale is loop-invariant inside each group.
"""

from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_GROUP = 128


class QTensor:
    """A quantized weight: ``dequant == q.float() * s`` (grouped along K)."""

    q: Tensor
    """int8 payload, shape ``[..., K, M]`` (same layout as the fp16 weight)."""

    s: Tensor
    """fp16 scales, shape ``[..., K // G, M]``."""

    group: int

    def __init__(self, q: Tensor, s: Tensor, group: int = DEFAULT_GROUP) -> None:
        assert q.shape[:-2] == s.shape[:-2], (q.shape, s.shape)
        assert q.shape[-2] == s.shape[-2] * group, (q.shape, s.shape)
        assert q.shape[-1] == s.shape[-1]
        assert q.dtype == torch.int8 and s.dtype == torch.float16
        self.q = q
        self.s = s
        self.group = group

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.q.shape)

    @property
    def dtype(self) -> torch.dtype:
        """The compute dtype activations are dequantized into."""
        return self.s.dtype

    @property
    def device(self) -> torch.device:
        return self.q.device

    def dequant(self) -> Tensor:
        """Materialize the fp16 weight (prefill / reference paths)."""
        s = self.s.repeat_interleave(self.group, dim=-2)
        return (self.q.float() * s.float()).to(self.s.dtype)

    def to(self, device: str | torch.device) -> QTensor:
        return QTensor(self.q.to(device), self.s.to(device), self.group)

    def __repr__(self) -> str:
        return f"QTensor(shape={tuple(self.q.shape)}, group={self.group})"


def quantize(w: Tensor, group: int = DEFAULT_GROUP) -> QTensor:
    """Quantize a ``[..., K, M]`` weight to int8 with per-group-per-column scales.

    Symmetric round-to-nearest; done in fp32 regardless of the input dtype.
    """
    assert w.dim() >= 2
    K, M = w.shape[-2], w.shape[-1]
    if K % group != 0:
        raise ValueError(f"K={K} not divisible by group={group}")
    wf = w.float()
    wg = wf.view(*w.shape[:-2], K // group, group, M)
    s = wg.abs().amax(dim=-2, keepdim=True) / 127.0
    s = s.clamp(min=1e-12)
    q = (wg / s).round().clamp(-127, 127).to(torch.int8).view(*w.shape)
    return QTensor(q, s.squeeze(-2).half(), group)


def quant_error(w: Tensor, qt: QTensor) -> dict[str, float]:
    """Rel error / cosine between the fp16 weight and its dequantized form."""
    a = w.float().flatten()
    b = qt.dequant().float().flatten()
    rel = (a - b).norm() / a.norm()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0)
    return {"rel": rel.item(), "cos": cos.item()}


def stack_qtensors(qts: list[QTensor]) -> QTensor:
    """Stack same-shaped QTensors along a new leading dim (e.g. rkvWt)."""
    assert len(qts) > 0
    g = qts[0].group
    assert all(q.group == g and q.shape == qts[0].shape for q in qts)
    return QTensor(
        torch.stack([q.q for q in qts]),
        torch.stack([q.s for q in qts]),
        g,
    )
