from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from rwkv_tl.tokenizer import Tokenizer


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
    return F.layer_norm(x, (x.shape[-1],), w, b, 1e-5)


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
    S = (
        torch.einsum("hvk,hk->hvk", S, W)
        + torch.einsum("hva,ha,hb->hvb", S, A, B)
        + torch.einsum("hv,hk->hvk", V, K)
    )
    return torch.einsum("hvk,hk->hv", S, R), S


class RWKV7Torch:
    """Pure PyTorch RWKV7 baseline without fused custom kernels."""

    def __init__(self, checkpoint_path: str, vocab_path: str) -> None:
        pth: dict[str, Tensor] = torch.load(checkpoint_path)
        self.W = pth

        W = self.W
        self.tokenizer = Tokenizer(vocab_path)
        self.n_layer = 1 + max(
            int(k.split(".")[1]) for k in W if k.startswith("blocks.")
        )
        self.C, self.N = W["emb.weight"].shape[1], 64
        self.H = self.C // self.N
        self.emb = _layer_norm(
            W["emb.weight"], W["blocks.0.ln0.weight"], W["blocks.0.ln0.bias"]
        )
        self.ln_outW, self.ln_outB, self.head = (
            W["ln_out.weight"],
            W["ln_out.bias"],
            W["head.weight"],
        )
        self.layers = [
            (self.make_TMIX(i), self.make_CMIX(i)) for i in range(self.n_layer)
        ]
        self.layers_batch = [
            (self.make_TMIX_batch(i), self.make_CMIX_batch(i))
            for i in range(self.n_layer)
        ]

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def zero_state(self) -> list[list[dict[str, Tensor]]]:
        S: list[list[dict[str, Tensor]]] = []
        for _ in range(self.n_layer):
            S.append(
                [
                    {
                        "x": torch.zeros(self.C, dtype=torch.bfloat16),
                        "rnn": torch.zeros(
                            (self.H, self.N, self.N), dtype=torch.bfloat16
                        ),
                    },
                    {"x": torch.zeros(self.C, dtype=torch.bfloat16)},
                ]
            )
        return S

    def reset_state(
        self, S: list[list[dict[str, Tensor]]]
    ) -> list[list[dict[str, Tensor]]]:
        for layer_state in S:
            for slot in layer_state:
                for v in slot.values():
                    v.zero_()
        return S

    def HEAD(self, X: Tensor) -> Tensor:
        return self.head @ X

    def run_one(
        self, token: int, S: list[list[dict[str, Tensor]]]
    ) -> tuple[Tensor, list[list[dict[str, Tensor]]]]:
        X = self.EMB(token)
        v_first: Tensor | None = None

        for (TM, CM), s in zip(self.layers, S):
            X, v_first, s[0] = TM(X, v_first, s[0])
            X, s[1] = CM(X, s[1])
        return self.HEAD(self.NORM(X)), S

    def forward(
        self,
        tokens: list[int] | Tensor,
        S: list[list[dict[str, Tensor]]] | None = None,
    ) -> tuple[Tensor, list[list[dict[str, Tensor]]]]:
        """Run inference over a token sequence.

        Single-token inputs use the `run_one` decode path. Multi-token inputs
        are routed to `forward_prefill` for a batched-GEMM path instead of a
        per-token Python loop of GEMVs.
        """
        S = self.zero_state() if S is None else S
        if isinstance(tokens, Tensor):
            tok = tokens.reshape(-1)
        else:
            tok = torch.as_tensor(
                list(tokens), dtype=torch.long, device=self.emb.device
            )
        if tok.numel() == 0:
            raise RuntimeError("forward received an empty token sequence")
        if tok.numel() == 1:
            return self.run_one(int(tok.item()), S)
        return self.forward_prefill(tok, S)

    def forward_prefill(
        self,
        tokens: list[int] | Tensor,
        S: list[list[dict[str, Tensor]]] | None = None,
    ) -> tuple[Tensor, list[list[dict[str, Tensor]]]]:
        S = self.zero_state() if S is None else S
        if isinstance(tokens, Tensor):
            tok = tokens.reshape(-1)
        else:
            tok = torch.as_tensor(
                list(tokens), dtype=torch.long, device=self.emb.device
            )
        if tok.numel() == 0:
            raise RuntimeError("forward_prefill received an empty token sequence")
        X = self.emb[tok]
        v_first: Tensor | None = None
        for (TM, CM), s in zip(self.layers_batch, S):
            X, v_first, s[0] = TM(X, v_first, s[0])
            X, s[1] = CM(X, s[1])
        return self.HEAD(self.NORM(X[-1])), S

    def EMB(self, token: int) -> Tensor:
        return self.emb[token]

    def NORM(self, X: Tensor) -> Tensor:
        return _layer_norm(X, self.ln_outW, self.ln_outB)

    def make_TMIX(self, i: int):
        p, W, H, N = f"blocks.{i}.att.", self.W, self.H, self.N
        lnW, lnB = W[f"blocks.{i}.ln1.weight"], W[f"blocks.{i}.ln1.bias"]
        x_r, x_w, x_k, x_v, x_a, x_g = (
            W[p + "x_r"].reshape(-1),
            W[p + "x_w"].reshape(-1),
            W[p + "x_k"].reshape(-1),
            W[p + "x_v"].reshape(-1),
            W[p + "x_a"].reshape(-1),
            W[p + "x_g"].reshape(-1),
        )
        rW, kW, vW, oW = (
            W[p + "receptance.weight"],
            W[p + "key.weight"],
            W[p + "value.weight"],
            W[p + "output.weight"],
        )
        w0, w1, w2, a0, a1, a2, v0, v1, v2 = (
            W[p + "w0"].reshape(-1),
            W[p + "w1"],
            W[p + "w2"],
            W[p + "a0"].reshape(-1),
            W[p + "a1"],
            W[p + "a2"],
            W[p + "v0"].reshape(-1),
            W[p + "v1"],
            W[p + "v2"],
        )
        g1_t, g2_t, k_k, k_a, r_k, ln_xW, ln_xB = (
            W[p + "g1"].T.contiguous(),
            W[p + "g2"].T.contiguous(),
            W[p + "k_k"].reshape(-1),
            W[p + "k_a"].reshape(-1),
            W[p + "r_k"],
            W[p + "ln_x.weight"],
            W[p + "ln_x.bias"],
        )

        def layer(
            x0: Tensor, v_first: Tensor | None, state: dict[str, Tensor]
        ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
            x = _layer_norm(x0, lnW, lnB)
            prev = state["x"]

            xr = _lerp(x, prev, x_r)
            xw = _lerp(x, prev, x_w)
            xk = _lerp(x, prev, x_k)
            xv = _lerp(x, prev, x_v)
            xa = _lerp(x, prev, x_a)
            xg = _lerp(x, prev, x_g)
            # copy after reading prev: state["x"] aliases prev, so an
            # earlier in-place copy_ here would corrupt prev before use.
            state["x"].copy_(x)

            r = torch.mv(rW, xr)
            k = torch.mv(kW, xk)
            v = torch.mv(vW, xv)

            if v_first is None:
                v_first = v
            else:
                v = _lerp(v, v_first, _sigmoid(v0 + xv @ v1 @ v2))

            w = torch.exp(-_sigmoid(w0 + torch.tanh(xw @ w1) @ w2) / (torch.e**0.5))
            a = _sigmoid(a0 + xa @ a1 @ a2)
            kk = k * k_k
            k = _lerp(k, k * a, k_a)

            r, w, k, v, kk, a = [z.reshape(H, N) for z in (r, w, k, v, kk, a)]
            kk = _l2_rwkv(kk)

            y, state["rnn"] = _dplr_rwkv(state["rnn"], r, w, k, v, kk, -kk * a)
            y = _group_norm(y, ln_xW, ln_xB, 64e-5)
            y += (torch.sum(r * k * r_k, dim=1, keepdim=True) * v).reshape(-1)
            g = torch.mv(g2_t, torch.sigmoid(torch.mv(g1_t, xg)))
            return torch.addmv(x0, oW, (y * g)), v_first, state

        return layer

    def make_CMIX(self, i: int):
        p, W = f"blocks.{i}.ffn.", self.W
        lnW, lnB, x_k, kW, vW = (
            W[f"blocks.{i}.ln2.weight"],
            W[f"blocks.{i}.ln2.bias"],
            W[p + "x_k"].reshape(-1),
            W[p + "key.weight"],
            W[p + "value.weight"],
        )

        def layer(
            x0: Tensor, state: dict[str, Tensor]
        ) -> tuple[Tensor, dict[str, Tensor]]:
            x_ln = _layer_norm(x0, lnW, lnB)
            prev = state["x"]
            x = _lerp(x_ln, prev, x_k)
            state["x"].copy_(x_ln)
            return torch.addmv(x0, vW, _relusq(torch.mv(kW, x))), state

        return layer

    def make_TMIX_batch(self, i: int):
        # Batched TMIX for prefill: [T, C] GEMM path instead of per-token GEMV.
        # DPLR recurrence stays serial over T (state-dependent).
        p, W, H, N = f"blocks.{i}.att.", self.W, self.H, self.N
        lnW, lnB = W[f"blocks.{i}.ln1.weight"], W[f"blocks.{i}.ln1.bias"]
        x_r, x_w, x_k, x_v, x_a, x_g = (
            W[p + "x_r"].reshape(-1),
            W[p + "x_w"].reshape(-1),
            W[p + "x_k"].reshape(-1),
            W[p + "x_v"].reshape(-1),
            W[p + "x_a"].reshape(-1),
            W[p + "x_g"].reshape(-1),
        )
        rWt, kWt, vWt, oWt = (
            W[p + "receptance.weight"].T.contiguous(),
            W[p + "key.weight"].T.contiguous(),
            W[p + "value.weight"].T.contiguous(),
            W[p + "output.weight"].T.contiguous(),
        )
        w0, w1, w2, a0, a1, a2, v0, v1, v2 = (
            W[p + "w0"].reshape(-1),
            W[p + "w1"],
            W[p + "w2"],
            W[p + "a0"].reshape(-1),
            W[p + "a1"],
            W[p + "a2"],
            W[p + "v0"].reshape(-1),
            W[p + "v1"],
            W[p + "v2"],
        )
        g1, g2, k_k, k_a, r_k, ln_xW, ln_xB = (
            W[p + "g1"],
            W[p + "g2"],
            W[p + "k_k"].reshape(-1),
            W[p + "k_a"].reshape(-1),
            W[p + "r_k"],
            W[p + "ln_x.weight"],
            W[p + "ln_x.bias"],
        )

        def layer(
            x0: Tensor, v_first: Tensor | None, state: dict[str, Tensor]
        ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
            T_len = x0.shape[0]
            x = _layer_norm(x0, lnW, lnB)
            prev = torch.cat([state["x"].unsqueeze(0), x[:-1]], dim=0)
            xr = _lerp(x, prev, x_r)
            xw = _lerp(x, prev, x_w)
            xk = _lerp(x, prev, x_k)
            xv = _lerp(x, prev, x_v)
            xa = _lerp(x, prev, x_a)
            xg = _lerp(x, prev, x_g)
            state["x"] = x[-1]

            r = xr @ rWt
            k = xk @ kWt
            v = xv @ vWt

            if v_first is None:
                v_first = v
            else:
                v = _lerp(v, v_first, _sigmoid(v0 + xv @ v1 @ v2))

            w = torch.exp(-_sigmoid(w0 + torch.tanh(xw @ w1) @ w2) / (torch.e**0.5))
            a = _sigmoid(a0 + xa @ a1 @ a2)
            kk = k * k_k
            k = _lerp(k, k * a, k_a)

            r, w, k, v, kk, a = [z.view(T_len, H, N) for z in (r, w, k, v, kk, a)]
            kk = _l2_rwkv(kk)
            neg_kk_a = -kk * a

            y = torch.empty(T_len, H, N, dtype=x0.dtype, device=x0.device)
            for t in range(T_len):
                y_t, state["rnn"] = _dplr_rwkv(
                    state["rnn"], r[t], w[t], k[t], v[t], kk[t], neg_kk_a[t]
                )
                y[t] = y_t

            y = F.group_norm(y.reshape(T_len, H * N), H, ln_xW, ln_xB, 64e-5)
            rkrk = torch.sum(r * k * r_k, dim=-1, keepdim=True)
            y = (y.view(T_len, H, N) + rkrk * v).reshape(T_len, H * N)
            g = torch.sigmoid(xg @ g1) @ g2
            return x0 + (y * g) @ oWt, v_first, state

        return layer

    def make_CMIX_batch(self, i: int):
        # Batched CMIX for prefill: [T, C] GEMM path.
        p, W = f"blocks.{i}.ffn.", self.W
        lnW, lnB, x_k, kWt, vWt = (
            W[f"blocks.{i}.ln2.weight"],
            W[f"blocks.{i}.ln2.bias"],
            W[p + "x_k"].reshape(-1),
            W[p + "key.weight"].T.contiguous(),
            W[p + "value.weight"].T.contiguous(),
        )

        def layer(
            x0: Tensor, state: dict[str, Tensor]
        ) -> tuple[Tensor, dict[str, Tensor]]:
            x_ln = _layer_norm(x0, lnW, lnB)
            prev = torch.cat([state["x"].unsqueeze(0), x_ln[:-1]], dim=0)
            x = _lerp(x_ln, prev, x_k)
            state["x"] = x_ln[-1]
            return x0 + _relusq(x @ kWt) @ vWt, state

        return layer
