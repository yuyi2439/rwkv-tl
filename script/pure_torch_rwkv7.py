from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from rwkv_tl._compat import maybe_torch_compile
from rwkv_tl.model import RWKV7Weight
from rwkv_tl.state import State


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


def _layer_norm(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return F.layer_norm(x, (x.shape[-1],), w, b)


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
    return y.bfloat16(), S_new


class RWKV7Torch:
    """Pure PyTorch RWKV7 baseline without fused custom kernels.

    Shares the same ``RWKV7Weight`` as ``rwkv_tl.RWKV7``; state is passed in
    and out explicitly (``State``), so the instance itself is stateless.
    """

    def __init__(
        self,
        w: RWKV7Weight,
        *,
        is_torch_compile: bool = True,
    ) -> None:
        self.w = w
        self._is_torch_compile = is_torch_compile
        self.n_layer = w.N_LAYER
        self.C, self.N = w.N_EMBD, 64
        self.H = self.C // self.N
        self.emb = _layer_norm(w.emb, w.ln_in.w, w.ln_in.b)
        self.ln_outW, self.ln_outB, self.head = (
            w.ln_out.w,
            w.ln_out.b,
            w.head,
        )
        self.layers = [
            (self.make_TMIX(i), self.make_CMIX(i)) for i in range(self.n_layer)
        ]
        self.layers_batch = [
            (self.make_TMIX_batch(i), self.make_CMIX_batch(i))
            for i in range(self.n_layer)
        ]

    def HEAD(self, X: Tensor) -> Tensor:
        return self.head @ X

    @maybe_torch_compile
    def decode(self, token: int, S: State) -> tuple[Tensor, State]:
        X = self.EMB(token)
        v_first: Tensor | None = None

        for (TM, CM), tmix_state, cmix_state in zip(self.layers, S.tmix, S.cmix):
            X, v_first, tmix_state = TM(X, v_first, tmix_state)
            X, cmix_state = CM(X, cmix_state)
        return self.HEAD(self.LN_OUT(X)), S

    def forward(
        self,
        tokens: list[int] | Tensor,
        S: State,
    ) -> tuple[Tensor, State]:
        """Run inference over a token sequence.

        Single-token inputs use the `decode` path. Multi-token inputs are
        routed to `prefill` for a batched-GEMM path instead of a per-token
        Python loop of GEMVs.
        """
        if isinstance(tokens, Tensor):
            tok = tokens.reshape(-1)
        else:
            tok = torch.as_tensor(
                list(tokens), dtype=torch.long, device=self.emb.device
            )
        if tok.numel() == 0:
            raise RuntimeError("forward received an empty token sequence")
        if tok.numel() == 1:
            return self.decode(int(tok.item()), S)
        return self.prefill(tok, S)

    def prefill(
        self,
        tokens: list[int] | Tensor,
        S: State,
    ) -> tuple[Tensor, State]:
        if isinstance(tokens, Tensor):
            tok = tokens.reshape(-1)
        else:
            tok = torch.as_tensor(
                list(tokens), dtype=torch.long, device=self.emb.device
            )
        if tok.numel() == 0:
            raise RuntimeError("prefill received an empty token sequence")
        X = self.emb[tok]
        v_first: Tensor | None = None
        for (TM, CM), tmix_state, cmix_state in zip(self.layers_batch, S.tmix, S.cmix):
            X, v_first, tmix_state = TM(X, v_first, tmix_state)
            X, cmix_state = CM(X, cmix_state)
        return self.HEAD(self.LN_OUT(X[-1])), S

    def generate(
        self,
        tokens: list[int] | Tensor,
        S: State,
        max_tokens: int = 32,
    ) -> tuple[list[int], State]:
        logits, S = self.forward(tokens, S)
        out: list[int] = []
        for _ in range(max_tokens):
            token = int(torch.argmax(logits))
            out.append(token)
            logits, S = self.decode(token, S)
        return out, S

    def EMB(self, token: int) -> Tensor:
        return self.emb[token]

    def LN_OUT(self, X: Tensor) -> Tensor:
        return _layer_norm(X, self.ln_outW, self.ln_outB)

    def make_TMIX(self, i: int):
        b, att, H, N = self.w.blocks[i], self.w.blocks[i].att, self.H, self.N

        def layer(
            x0: Tensor, v_first: Tensor | None, state: dict[str, Tensor]
        ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
            x = _layer_norm(x0, b.ln_pret.w, b.ln_pret.b)
            prev = state["x"]

            xr = _lerp(x, prev, att.x_r.reshape(-1))
            xw = _lerp(x, prev, att.x_w.reshape(-1))
            xk = _lerp(x, prev, att.x_k.reshape(-1))
            xv = _lerp(x, prev, att.x_v.reshape(-1))
            xa = _lerp(x, prev, att.x_a.reshape(-1))
            xg = _lerp(x, prev, att.x_g.reshape(-1))
            # copy after reading prev: state["x"] aliases prev, so an
            # earlier in-place copy_ here would corrupt prev before use.
            state["x"].copy_(x)

            r = torch.mv(att.receptance_weight, xr)
            k = torch.mv(att.key_weight, xk)
            v = torch.mv(att.value_weight, xv)

            if v_first is None:
                v_first = v
            else:
                v = _lerp(
                    v,
                    v_first,
                    _sigmoid(
                        att.v0.reshape(-1)
                        + xv @ att.v1 @ att.v2
                    ),
                )

            w = torch.exp(
                -_sigmoid(
                    att.w0.reshape(-1)
                    + torch.tanh(xw @ att.w1) @ att.w2
                )
                / (torch.e**0.5)
            )
            a = _sigmoid(att.a0.reshape(-1) + xa @ att.a1 @ att.a2)
            kk = k * att.k_k.reshape(-1)
            k = _lerp(k, k * a, att.k_a.reshape(-1))

            r, w, k, v, kk, a = [z.reshape(H, N) for z in (r, w, k, v, kk, a)]
            kk = _l2_rwkv(kk)

            y, state["rnn"] = _dplr_rwkv(state["rnn"], r, w, k, v, kk, -kk * a)
            y = _group_norm(y, att.ln_x.w, att.ln_x.b, 64e-5)
            y += (
                torch.sum(r * k * att.r_k, dim=1, keepdim=True) * v
            ).reshape(-1)
            g = torch.mv(
                att.g2.T.contiguous(),
                torch.sigmoid(torch.mv(att.g1.T.contiguous(), xg)),
            )
            return torch.addmv(x0, att.output_weight, (y * g)), v_first, state

        return layer

    def make_CMIX(self, i: int):
        b, ffn = self.w.blocks[i], self.w.blocks[i].ffn

        def layer(
            x0: Tensor, state: dict[str, Tensor]
        ) -> tuple[Tensor, dict[str, Tensor]]:
            x_ln = _layer_norm(x0, b.ln_prec.w, b.ln_prec.b)
            prev = state["x"]
            x = _lerp(x_ln, prev, ffn.x_k.reshape(-1))
            state["x"].copy_(x_ln)
            return (
                torch.addmv(
                    x0, ffn.value_weight, _relusq(torch.mv(ffn.key_weight, x))
                ),
                state,
            )

        return layer

    def make_TMIX_batch(self, i: int):
        # Batched TMIX for prefill: [T, C] GEMM path instead of per-token GEMV.
        # DPLR recurrence stays serial over T (state-dependent).
        b, att, H, N = self.w.blocks[i], self.w.blocks[i].att, self.H, self.N
        rWt, kWt, vWt, oWt = (
            att.receptance_weight.T.contiguous(),
            att.key_weight.T.contiguous(),
            att.value_weight.T.contiguous(),
            att.output_weight.T.contiguous(),
        )
        w1, w2, a1, a2, v1, v2, g1, g2 = (
            att.w1,
            att.w2,
            att.a1,
            att.a2,
            att.v1,
            att.v2,
            att.g1,
            att.g2,
        )

        def layer(
            x0: Tensor, v_first: Tensor | None, state: dict[str, Tensor]
        ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
            T_len = x0.shape[0]
            x = _layer_norm(x0, b.ln_pret.w, b.ln_pret.b)
            prev = torch.cat([state["x"].unsqueeze(0), x[:-1]], dim=0)
            xr = _lerp(x, prev, att.x_r.reshape(-1))
            xw = _lerp(x, prev, att.x_w.reshape(-1))
            xk = _lerp(x, prev, att.x_k.reshape(-1))
            xv = _lerp(x, prev, att.x_v.reshape(-1))
            xa = _lerp(x, prev, att.x_a.reshape(-1))
            xg = _lerp(x, prev, att.x_g.reshape(-1))
            state["x"] = x[-1]

            r = xr @ rWt
            k = xk @ kWt
            v = xv @ vWt

            if v_first is None:
                v_first = v
            else:
                v = _lerp(
                    v,
                    v_first,
                    _sigmoid(att.v0.reshape(-1) + xv @ v1 @ v2),
                )

            w = torch.exp(
                -_sigmoid(
                    att.w0.reshape(-1)
                    + torch.tanh(xw @ w1) @ w2
                )
                / (torch.e**0.5)
            )
            a = _sigmoid(att.a0.reshape(-1) + xa @ a1 @ a2)
            kk = k * att.k_k.reshape(-1)
            k = _lerp(k, k * a, att.k_a.reshape(-1))

            r, w, k, v, kk, a = [z.view(T_len, H, N) for z in (r, w, k, v, kk, a)]
            kk = _l2_rwkv(kk)
            neg_kk_a = -kk * a

            y = torch.empty(T_len, H, N, dtype=x0.dtype, device=x0.device)
            for t in range(T_len):
                y_t, state["rnn"] = _dplr_rwkv(
                    state["rnn"], r[t], w[t], k[t], v[t], kk[t], neg_kk_a[t]
                )
                y[t] = y_t

            y = F.group_norm(y.reshape(T_len, H * N), H, att.ln_x.w, att.ln_x.b, 64e-5)
            rkrk = torch.sum(r * k * att.r_k, dim=-1, keepdim=True)
            y = (y.view(T_len, H, N) + rkrk * v).reshape(T_len, H * N)
            g = torch.sigmoid(xg @ g1) @ g2
            return x0 + (y * g) @ oWt, v_first, state

        return layer

    def make_CMIX_batch(self, i: int):
        # Batched CMIX for prefill: [T, C] GEMM path.
        b, ffn = self.w.blocks[i], self.w.blocks[i].ffn
        kWt, vWt = ffn.key_weight.T.contiguous(), ffn.value_weight.T.contiguous()

        def layer(
            x0: Tensor, state: dict[str, Tensor]
        ) -> tuple[Tensor, dict[str, Tensor]]:
            x_ln = _layer_norm(x0, b.ln_prec.w, b.ln_prec.b)
            prev = torch.cat([state["x"].unsqueeze(0), x_ln[:-1]], dim=0)
            x = _lerp(x_ln, prev, ffn.x_k.reshape(-1))
            state["x"] = x_ln[-1]
            return x0 + _relusq(x @ kWt) @ vWt, state

        return layer
