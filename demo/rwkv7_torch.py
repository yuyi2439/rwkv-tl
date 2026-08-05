"""Pure PyTorch RWKV7 reference implementation (no fused custom kernels).

A readable, kernel-free baseline that mirrors the ``decode``/``prefill``/
``forward``/``generate`` API of ``demo.rwkv7_fp16.RWKV7FP16``. Slower than the
tilelang path but serves as the numerical reference for correctness tests
and benchmarking.
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

    y, state["rnn"] = _dplr_rwkv(state["rnn"], r, w, k, v, kk, -kk * a)
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
        self.layers_batch = [
            (self.make_TMIX_batch(i), self.make_CMIX_batch(i)) for i in range(self.L)
        ]

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

        x = self.EMB(tokens)
        v_first: Tensor | None = None
        for (TM, CM), tmix_state, cmix_state in zip(self.layers_batch, S.tmix, S.cmix):
            x, v_first = TM(x, v_first, tmix_state)
            x = CM(x, cmix_state)
        return S

    def EMB(self, token: Tensor) -> Tensor:
        x = F.embedding(token, self.w.emb)
        return x.squeeze(0) if x.dim() > 1 else x

    def HEAD(self, X: Tensor) -> Tensor:
        return torch.mv(self.w.head, X)

    def make_TMIX_batch(self, i: int):
        # Batched TMIX for prefill: [T, C] GEMM path instead of per-token GEMV.
        # DPLR recurrence stays serial over T (state-dependent).
        att = self.w.blocks[i].att

        def layer(
            x0: Tensor, v_first: Tensor | None, state: dict[str, Tensor]
        ) -> tuple[Tensor, Tensor]:
            T_len = x0.shape[0]
            x = att.ln_pre(x0)
            prev = torch.cat([state["x"].unsqueeze(0), x[:-1]], dim=0)
            xr = _lerp(x, prev, att.x_r)
            xw = _lerp(x, prev, att.x_w)
            xk = _lerp(x, prev, att.x_k)
            xv = _lerp(x, prev, att.x_v)
            xa = _lerp(x, prev, att.x_a)
            xg = _lerp(x, prev, att.x_g)
            state["x"] = x[-1]

            r, k, v = torch.bmm(
                torch.stack([xr, xk, xv], 0), att.rkvWt
            ).unbind(0)

            if v_first is None:
                v_first = v
            else:
                v = _lerp(
                    v,
                    v_first,
                    _sigmoid(att.v0.reshape(-1) + xv @ att.v1 @ att.v2),
                )

            w = torch.exp(
                -_sigmoid(att.w0.reshape(-1) + torch.tanh(xw @ att.w1) @ att.w2)
                / (torch.e**0.5)
            )
            a = _sigmoid(att.a0.reshape(-1) + xa @ att.a1 @ att.a2)
            kk = k * att.k_k.reshape(-1)
            k = _lerp(k, k * a, att.k_a.reshape(-1))

            r, w, k, v, kk, a = [
                z.view(T_len, self.H, self.N) for z in (r, w, k, v, kk, a)
            ]
            kk = _l2_rwkv(kk)
            neg_kk_a = -kk * a

            y = torch.empty(T_len, self.H, self.N, dtype=x0.dtype, device=x0.device)
            for t in range(T_len):
                y_t, state["rnn"] = _dplr_rwkv(
                    state["rnn"], r[t], w[t], k[t], v[t], kk[t], neg_kk_a[t]
                )
                y[t] = y_t

            y = F.group_norm(
                y.reshape(T_len, self.H * self.N), self.H, att.ln_x.w, att.ln_x.b, 64e-5
            )
            rkrk = torch.sum(r * k * att.r_k, dim=-1, keepdim=True)
            y = (y.view(T_len, self.H, self.N) + rkrk * v).reshape(
                T_len, self.H * self.N
            )
            g = torch.sigmoid(xg @ att.g1) @ att.g2
            return x0 + (y * g) @ att.oWt, v_first

        return layer

    def make_CMIX_batch(self, i: int):
        # Batched CMIX for prefill: [T, C] GEMM path.
        ffn = self.w.blocks[i].ffn

        def layer(x0: Tensor, state: dict[str, Tensor]) -> Tensor:
            x_ln = ffn.ln_pre(x0)
            prev = torch.cat([state["x"].unsqueeze(0), x_ln[:-1]], dim=0)
            x = _lerp(x_ln, prev, ffn.x_k)
            state["x"] = x_ln[-1]
            return x0 + _relusq(x @ ffn.kWt) @ ffn.vWt

        return layer
