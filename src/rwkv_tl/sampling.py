"""Token sampling helpers for autoregressive generation.

Shared by ``rwkv_tl.RWKV7.generate`` and the pure_torch reference, so both
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
    seen: list[int] | None = None,
) -> int:
    """Sample a token id from logits.

    Args:
        logits: Model logits, [V].
        temperature: Softmax temperature. None or <= 0 means greedy argmax.
        top_k: Restrict sampling to the top-k logits (0 = no restriction).
        top_p: Nucleus sampling threshold (1.0 = no restriction). Only the
            smallest set of tokens whose cumulative softmax probability reaches
            top_p is kept.
        repetition_penalty: Standard repetition penalty. For each token already
            in ``seen``, divide its logit by the penalty when the logit > 0 and
            multiply when < 0 (1.0 = no penalty).
        seen: Previously generated token ids (penalized if repetition_penalty
            is set).

    Returns:
        Sampled token id.
    """
    l = logits.float()

    if repetition_penalty != 1.0 and seen:
        idx = torch.as_tensor(seen, device=l.device)
        lv = l[idx]
        lv = torch.where(lv > 0, lv / repetition_penalty, lv * repetition_penalty)
        l[idx] = lv

    if temperature is None or temperature <= 0:
        return int(l.argmax().item())

    l = l / temperature

    if top_k > 0:
        k = min(top_k, l.numel())
        if k < l.numel():
            cutoff = torch.topk(l, k).values[-1]
            l = torch.where(l >= cutoff, l, torch.full_like(l, float("-inf")))

    if top_p < 1.0:
        probs = torch.softmax(l, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        drop = cum - sorted_probs > top_p
        l[sorted_idx[drop]] = float("-inf")

    probs = torch.softmax(l, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def apply_stop(out: list[int], stop: list[list[int]] | None) -> bool:
    """If the generated tail ends with a stop sequence, truncate and stop.

    Checks the generated tokens (``out``) against every sequence in ``stop``.
    On a match the matched sequence is removed from the end and True is
    returned so the caller halts generation.

    Args:
        out: Generated token ids (mutated in place on a match).
        stop: Stop sequences (list of token-id lists); None/empty disables.

    Returns:
        True if a stop sequence matched and was truncated.
    """
    if not stop:
        return False
    for seq in stop:
        if seq and out[-len(seq) :] == seq:
            del out[-len(seq) :]
            return True
    return False
