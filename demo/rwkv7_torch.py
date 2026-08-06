"""Pure PyTorch RWKV7 reference implementation (no fused custom kernels).

A readable, kernel-free baseline that mirrors the ``decode``/``prefill``/
``forward``/``generate`` API of ``demo.rwkv7_fp16.RWKV7FP16``. Slower than the
tilelang path but serves as the numerical reference for correctness tests
and benchmarking.

State tensors are updated in place (``copy_``), never rebound, so the class is
CUDA-Graph capturable: ``make_rwkv7(..., backend="torch", use_graph=True)``
wraps it like any other CUDA model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from rwkv_tl._compat import maybe_torch_compile
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7ATTWeight, RWKV7FFNWeight, RWKV7Weight

from ._rwkv7_abc import RWKV7Model


def _sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def _relusq(x: Tensor) -> Tensor:
    return F.relu(x) ** 2


def _lerp(x: Tensor, y: Tensor, w: Tensor) -> Tensor:
    return x + w * (y - x)


def _l2_rwkv(x: Tensor) -> Tensor:
    # dim=-1 so this works for both single-token (H,N) and batched (T,H,N).
    den = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True))
    return x / torch.clamp(den, min=1e-12)


def _group_norm(x: Tensor, w: Tensor, b: Tensor, eps: float) -> Tensor:
    h, n = x.shape
    return F.group_norm(x.reshape(1, h * n), h, w, b, eps).reshape(-1)


def _dplr_rwkv(
    S: Tensor,
    R: Tensor,
    W: Tensor,
    K: Tensor,
    V: Tensor,
    A: Tensor,
    B: Tensor,
) -> tuple[Tensor, Tensor]:
    S_new = (
        torch.einsum("hvk,hk->hvk", S, W.float())
        + torch.einsum("hva,ha,hb->hvb", S, A.float(), B.float())
        + torch.einsum("hv,hk->hvk", V.float(), K.float())
    )
    y = torch.einsum("hvk,hk->hv", S_new, R.float())
    return y.to(V.dtype), S_new


def time_mix(
    weight: RWKV7ATTWeight,
    x0: Tensor,
    v_first: Tensor | None,
    state: dict[str, Tensor],
    H: int,
    N: int,
) -> tuple[Tensor, Tensor]:
    x = weight.ln_pre(x0)
    prev = state["x"]

    xr = _lerp(x, prev, weight.x_r)
    xw = _lerp(x, prev, weight.x_w)
    xk = _lerp(x, prev, weight.x_k)
    xv = _lerp(x, prev, weight.x_v)
    xa = _lerp(x, prev, weight.x_a)
    xg = _lerp(x, prev, weight.x_g)
    # copy after reading prev: state["x"] aliases prev, so an
    # earlier in-place copy_ here would corrupt prev before use.
    state["x"].copy_(x)

    r = xr @ weight.rkvWt[0]
    k = xk @ weight.rkvWt[1]
    v = xv @ weight.rkvWt[2]

    if v_first is None:
        v_first = v
    else:
        v = _lerp(
            v,
            v_first,
            _sigmoid(weight.v0.reshape(-1) + xv @ weight.v1 @ weight.v2),
        )

    w = torch.exp(
        -_sigmoid(weight.w0.reshape(-1) + torch.tanh(xw @ weight.w1) @ weight.w2)
        / (torch.e**0.5)
    )
    a = _sigmoid(weight.a0.reshape(-1) + xa @ weight.a1 @ weight.a2)
    kk = k * weight.k_k.reshape(-1)
    k = _lerp(k, k * a, weight.k_a.reshape(-1))

    r, w, k, v, kk, a = [z.reshape(H, N) for z in (r, w, k, v, kk, a)]
    kk = _l2_rwkv(kk)

    y, rnn_new = _dplr_rwkv(state["rnn"], r, w, k, v, kk, -kk * a)
    state["rnn"].copy_(rnn_new)
    y = _group_norm(y, weight.ln_x.w, weight.ln_x.b, 64e-5)
    y += (torch.sum(r * k * weight.r_k, dim=1, keepdim=True) * v).reshape(-1)
    g = torch.mv(
        weight.g2.T.contiguous(),
        torch.sigmoid(torch.mv(weight.g1t, xg)),
    )
    return torch.add(x0, (y * g) @ weight.oWt), v_first


def channel_mix(weight: RWKV7FFNWeight, x0: Tensor, state: dict[str, Tensor]) -> Tensor:
    x_ln = weight.ln_pre(x0)
    prev = state["x"]
    x = _lerp(x_ln, prev, weight.x_k)
    state["x"].copy_(x_ln)
    return torch.add(x0, _relusq(x @ weight.kWt) @ weight.vWt)


def time_mix_batch(
    weight: RWKV7ATTWeight,
    x0: Tensor,
    v_first: Tensor | None,
    state: dict[str, Tensor],
    H: int,
    N: int,
) -> tuple[Tensor, Tensor]:
    """Batched TMIX for prefill: [T, C] GEMM path instead of per-token GEMV.

    DPLR recurrence stays serial over T (state-dependent).
    """
    T_len = x0.shape[0]
    x = weight.ln_pre(x0)
    prev = torch.cat([state["x"].unsqueeze(0), x[:-1]], dim=0)
    xr = _lerp(x, prev, weight.x_r)
    xw = _lerp(x, prev, weight.x_w)
    xk = _lerp(x, prev, weight.x_k)
    xv = _lerp(x, prev, weight.x_v)
    xa = _lerp(x, prev, weight.x_a)
    xg = _lerp(x, prev, weight.x_g)
    state["x"].copy_(x[-1])

    r, k, v = torch.bmm(torch.stack([xr, xk, xv], 0), weight.rkvWt).unbind(0)

    if v_first is None:
        v_first = v
    else:
        v = _lerp(
            v,
            v_first,
            _sigmoid(weight.v0.reshape(-1) + xv @ weight.v1 @ weight.v2),
        )

    w = torch.exp(
        -_sigmoid(weight.w0.reshape(-1) + torch.tanh(xw @ weight.w1) @ weight.w2)
        / (torch.e**0.5)
    )
    a = _sigmoid(weight.a0.reshape(-1) + xa @ weight.a1 @ weight.a2)
    kk = k * weight.k_k.reshape(-1)
    k = _lerp(k, k * a, weight.k_a.reshape(-1))

    r, w, k, v, kk, a = [z.view(T_len, H, N) for z in (r, w, k, v, kk, a)]
    kk = _l2_rwkv(kk)
    neg_kk_a = -kk * a

    y = torch.empty(T_len, H, N, dtype=x0.dtype, device=x0.device)
    for t in range(T_len):
        y_t, rnn_new = _dplr_rwkv(
            state["rnn"], r[t], w[t], k[t], v[t], kk[t], neg_kk_a[t]
        )
        state["rnn"].copy_(rnn_new)
        y[t] = y_t

    y = F.group_norm(y.reshape(T_len, H * N), H, weight.ln_x.w, weight.ln_x.b, 64e-5)
    rkrk = torch.sum(r * k * weight.r_k, dim=-1, keepdim=True)
    y = (y.view(T_len, H, N) + rkrk * v).reshape(T_len, H * N)
    g = torch.sigmoid(xg @ weight.g1) @ weight.g2
    return x0 + (y * g) @ weight.oWt, v_first


def channel_mix_batch(
    weight: RWKV7FFNWeight, x0: Tensor, state: dict[str, Tensor]
) -> Tensor:
    """Batched CMIX for prefill: [T, C] GEMM path."""
    x_ln = weight.ln_pre(x0)
    prev = torch.cat([state["x"].unsqueeze(0), x_ln[:-1]], dim=0)
    x = _lerp(x_ln, prev, weight.x_k)
    state["x"].copy_(x_ln[-1])
    return x0 + _relusq(x @ weight.kWt) @ weight.vWt


class RWKV7Torch(RWKV7Model):
    """Pure PyTorch RWKV7 baseline without fused custom kernels.

    State is passed in and out explicitly (``State``), so the instance itself is stateless.
    """

    def __init__(
        self,
        w: RWKV7Weight,
        *,
        is_torch_compile: bool = True,
    ) -> None:
        super().__init__(w)
        self._is_torch_compile = is_torch_compile

    @maybe_torch_compile
    def decode(self, token: Tensor, S: State) -> tuple[Tensor, State]:
        x = self.EMB(token)
        v_first: Tensor | None = None

        for i, block in enumerate(self.w.blocks):
            x, v_first = time_mix(block.att, x, v_first, S.tmix[i], self.H, self.N)
            x = channel_mix(block.ffn, x, S.cmix[i])

        return self.HEAD(self.w.ln_out(x)), S

    def prefill(self, tokens: Tensor, S: State) -> State:
        if tokens.ndim != 1:
            raise ValueError(f"Expected 1D token sequence, got {tokens.shape}")
        if tokens.numel() == 0:
            raise RuntimeError("prefill received an empty token sequence")

        x = self.w.emb[tokens]
        v_first: Tensor | None = None
        for i, block in enumerate(self.w.blocks):
            x, v_first = time_mix_batch(
                block.att, x, v_first, S.tmix[i], self.H, self.N
            )
            x = channel_mix_batch(block.ffn, x, S.cmix[i])
        return S

    def EMB(self, token: Tensor) -> Tensor:
        x = F.embedding(token, self.w.emb)
        return x.squeeze(0) if x.dim() > 1 else x

    def HEAD(self, X: Tensor) -> Tensor:
        return torch.mv(self.w.head, X)
