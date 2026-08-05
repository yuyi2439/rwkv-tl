"""Token sampling helpers for autoregressive generation.

Shared by ``RWKV7Model.generate`` and the pure_torch reference, so both
shells sample identically.
"""

from __future__ import annotations

import torch
from torch import Tensor


def sample_logits(
    logits: Tensor,
    *,
    temperature: float | None = None,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    seen: Tensor | None = None,
) -> Tensor:
    """Sample a token id from logits.

    Args:
        logits: Model logits, 1-D ``[V]`` or 2-D ``[B, V]`` (one row per
            sequence). All filtering happens along the last dimension.
        temperature: Softmax temperature. None or <= 0 means greedy argmax.
        top_k: Restrict sampling to the top-k logits (0 = no restriction).
        top_p: Nucleus sampling threshold (1.0 = no restriction). Only the
            smallest set of tokens whose cumulative softmax probability reaches
            top_p is kept.
        repetition_penalty: Standard repetition penalty. For each token already
            in ``seen``, divide its logit by the penalty when the logit > 0 and
            multiply when < 0 (1.0 = no penalty).
        seen: Previously generated token ids (penalized if repetition_penalty
            is set). Indexes the last (vocab) dimension.

    Returns:
        Sampled token ids as a tensor with ``ndim = logits.ndim - 1``: a 0-dim
        scalar for ``[V]`` logits, a 1-D ``[B]`` tensor for ``[B, V]`` logits.
    """
    l = logits.float()

    if repetition_penalty != 1.0 and seen is not None:
        lv = l[..., seen]
        lv = torch.where(lv > 0, lv / repetition_penalty, lv * repetition_penalty)
        l[..., seen] = lv

    if temperature is None or temperature <= 0:
        return l.argmax(dim=-1)

    l = l / temperature

    if top_k > 0:
        k = min(top_k, l.shape[-1])
        if k < l.shape[-1]:
            cutoff = torch.topk(l, k, dim=-1).values[..., -1:]
            l = torch.where(l >= cutoff, l, torch.full_like(l, float("-inf")))

    if top_p < 1.0:
        probs = torch.softmax(l, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        drop = cum - sorted_probs > top_p
        kept = torch.gather(l, -1, sorted_idx)
        vals = torch.where(drop, torch.full_like(kept, float("-inf")), kept)
        l = l.scatter(-1, sorted_idx, vals)

    probs = torch.softmax(l, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)
