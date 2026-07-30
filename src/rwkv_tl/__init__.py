from __future__ import annotations

import torch

from .tokenizer import Tokenizer


def SIGMOID(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def RELUSQ(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0) ** 2


def LERP(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x + w * (y - x)


def L2_RWKV(x: torch.Tensor) -> torch.Tensor:
    den = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
    return x / torch.clamp(den, min=1e-12)


def LAYER_NORM(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(axis=-1, keepdims=True)) / (
        x.var(axis=-1, keepdims=True) + 1e-5
    ) ** 0.5 * w + b


def GROUP_NORM(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    y = (x - x.mean(axis=1, keepdims=True)) / (
        x.var(axis=1, keepdims=True) + eps
    ) ** 0.5
    return y.reshape(-1) * w + b


def DPLR_RWKV(
    S: torch.Tensor,
    R: torch.Tensor,
    W: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    S = (
        torch.einsum("hvk,hk->hvk", S, W)
        + torch.einsum("hva,ha,hb->hvb", S, A, B)
        + torch.einsum("hv,hk->hvk", V, K)
    )
    return torch.einsum("hvk,hk->hv", S, R), S


class RWKV7:
    def __init__(self, checkpoint_path: str, vocab_path: str) -> None:
        pth: dict[str, torch.Tensor] = torch.load(checkpoint_path)
        self.W = pth

        W = self.W
        self.tokenizer = Tokenizer(vocab_path)
        self.n_layer = 1 + max(
            int(k.split(".")[1]) for k in W if k.startswith("blocks.")
        )
        self.C, self.N = W["emb.weight"].shape[1], 64
        self.H = self.C // self.N
        self.emb = LAYER_NORM(
            W["emb.weight"], W["blocks.0.ln0.weight"], W["blocks.0.ln0.bias"]
        )
        self.ln_outW, self.ln_outB, self.head = (
            W["ln_out.weight"],
            W["ln_out.bias"],
            W["head.weight"].T,
        )
        self.TM = tuple(self.make_TMIX(i) for i in range(self.n_layer))
        self.CM = tuple(self.make_CMIX(i) for i in range(self.n_layer))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def zero_state(self) -> list[list[dict[str, torch.Tensor]]]:
        S: list[list[dict[str, torch.Tensor]]] = []
        for _ in range(self.n_layer):
            S.append(
                [
                    {
                        "x": torch.zeros(self.C, dtype=torch.bfloat16),
                        "rnn": torch.zeros((self.H, self.N, self.N), dtype=torch.bfloat16),
                    },
                    {"x": torch.zeros(self.C, dtype=torch.bfloat16)},
                ]
            )
        return S

    def HEAD(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.head

    def run_one(self, token: int, S: list[list[dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, list[list[dict[str, torch.Tensor]]]]:
        X = self.EMB(int(token))
        for TM, CM, s in zip(self.TM, self.CM, S):
            X, s[0] = TM(X, s[0])
            X, s[1] = CM(X, s[1])
        return self.HEAD(self.NORM(X)), S

    def forward(
        self,
        tokens: list[int] | torch.Tensor,
        S: list[list[dict[str, torch.Tensor]]] | None = None,
    ) -> tuple[torch.Tensor, list[list[dict[str, torch.Tensor]]]]:
        S = self.zero_state() if S is None else S
        logits: torch.Tensor | None = None
        for token in tokens:
            logits, S = self.run_one(int(token), S)
        if logits is None:
            raise RuntimeError("forward received an empty token sequence")
        return logits, S

    def EMB(self, token: int) -> tuple[torch.Tensor, None]:
        return (self.emb[token], None)

    def NORM(self, X: tuple[torch.Tensor, None]) -> torch.Tensor:
        return LAYER_NORM(X[0], self.ln_outW, self.ln_outB)

    def make_TMIX(self, i: int):
        p, W, H, N = f"blocks.{i}.att.", self.W, self.H, self.N
        lnW, lnB = W[f"blocks.{i}.ln1.weight"], W[f"blocks.{i}.ln1.bias"]
        x_r, x_w, x_k, x_v, x_a, x_g = (
            W[p + "x_r"],
            W[p + "x_w"],
            W[p + "x_k"],
            W[p + "x_v"],
            W[p + "x_a"],
            W[p + "x_g"],
        )
        rW, kW, vW, oW = (
            W[p + "receptance.weight"].T,
            W[p + "key.weight"].T,
            W[p + "value.weight"].T,
            W[p + "output.weight"].T,
        )
        w0, w1, w2, a0, a1, a2, v0, v1, v2 = (
            W[p + "w0"],
            W[p + "w1"],
            W[p + "w2"],
            W[p + "a0"],
            W[p + "a1"],
            W[p + "a2"],
            W[p + "v0"],
            W[p + "v1"],
            W[p + "v2"],
        )
        g1, g2, k_k, k_a, r_k, ln_xW, ln_xB = (
            W[p + "g1"],
            W[p + "g2"],
            W[p + "k_k"],
            W[p + "k_a"],
            W[p + "r_k"],
            W[p + "ln_x.weight"],
            W[p + "ln_x.bias"],
        )

        def layer(X: tuple[torch.Tensor, torch.Tensor | None], state: dict[str, torch.Tensor]) -> tuple[tuple[torch.Tensor, torch.Tensor | None], dict[str, torch.Tensor]]:
            x0, v_first = X
            x = LAYER_NORM(x0, lnW, lnB)
            prev, state["x"] = state["x"], x
            xr, xw, xk = LERP(x, prev, x_r), LERP(x, prev, x_w), LERP(x, prev, x_k)
            xv, xa, xg = LERP(x, prev, x_v), LERP(x, prev, x_a), LERP(x, prev, x_g)

            r, k, v = xr @ rW, xk @ kW, xv @ vW
            if v_first is None:
                v_first = v
            else:
                v = LERP(v, v_first, SIGMOID(v0 + xv @ v1 @ v2))
            w = torch.exp(-SIGMOID(w0 + torch.tanh(xw @ w1) @ w2) / (torch.e ** 0.5))
            a = SIGMOID(a0 + xa @ a1 @ a2)
            kk = k * k_k
            k = LERP(k, k * a, k_a)
            r, w, k, v, kk, a = [z.reshape(H, N) for z in (r, w, k, v, kk, a)]
            kk = L2_RWKV(kk)

            y, state["rnn"] = DPLR_RWKV(state["rnn"], r, w, k, v, kk, -kk * a)
            y = GROUP_NORM(y, ln_xW, ln_xB, 64e-5)
            y += (torch.sum(r * k * r_k, dim=1, keepdim=True) * v).reshape(-1)
            g = SIGMOID(xg @ g1) @ g2
            return (x0 + (y * g) @ oW, v_first), state

        return layer

    def make_CMIX(self, i):
        p, W = f"blocks.{i}.ffn.", self.W
        lnW, lnB, x_k, kW, vW = (
            W[f"blocks.{i}.ln2.weight"],
            W[f"blocks.{i}.ln2.bias"],
            W[p + "x_k"],
            W[p + "key.weight"].T,
            W[p + "value.weight"].T,
        )

        def layer(X: tuple[torch.Tensor, torch.Tensor | None], state: dict[str, torch.Tensor]) -> tuple[tuple[torch.Tensor, torch.Tensor | None], dict[str, torch.Tensor]]:
            x0, v_first = X
            x = LAYER_NORM(x0, lnW, lnB)
            prev, state["x"] = state["x"], x
            x = LERP(x, prev, x_k)
            return (x0 + RELUSQ(x @ kW) @ vW, v_first), state

        return layer
