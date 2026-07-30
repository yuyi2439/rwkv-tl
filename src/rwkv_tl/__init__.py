from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .kernels.tile_kernels import fused_lerp6
from .tokenizer import Tokenizer

# Precomputed constant for the w decay gate: exp(-sigmoid(...) / sqrt(e)).
# `torch.e ** 0.5` is a Python-level recomputation each call; hoisting it
# avoids the per-token overhead and makes intent explicit.
_SQRT_E = torch.e**0.5


def SIGMOID(x: Tensor) -> Tensor:
    """Numerically stable sigmoid via the fused torch primitive."""
    return torch.sigmoid(x)


def RELUSQ(x: Tensor) -> Tensor:
    """Squared ReLU; `F.relu(x) ** 2` fuses to a single kernel."""
    return F.relu(x) ** 2


def LERP(x: Tensor, y: Tensor, w: Tensor) -> Tensor:
    """Linear interpolation `x + w*(y-x)`.

    `torch.lerp` is NOT used: on bf16 it falls back to separate sub/mul/add
    kernels (verified via profiler), offering no fusion benefit over the
    explicit form.
    """
    return x + w * (y - x)


def L2_RWKV(x: Tensor) -> Tensor:
    """L2-normalize rows along dim=1.

    `F.normalize` is NOT used: it dispatches to the same reduce+elementwise
    kernels as the explicit form (verified via profiler), so the explicit
    form is kept for clarity and to avoid the extra `max(||x||, eps)` vs
    `clamp(..., min=eps)` semantic divergence.
    """
    den = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
    return x / torch.clamp(den, min=1e-12)


def LAYER_NORM(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """Apply LayerNorm over the last dimension.

    Args:
        x: Input tensor.
        w: Weight tensor for the last dimension.
        b: Bias tensor for the last dimension.

    Returns:
        The normalized tensor.
    """
    return F.layer_norm(x, (x.shape[-1],), w, b, 1e-5)


def GROUP_NORM(x: Tensor, w: Tensor, b: Tensor, eps: float) -> Tensor:
    """Apply grouped normalization over an `[H, N]` tensor.

    The handwritten implementation normalizes along axis 1 for each head and
    then applies affine parameters over the flattened `H * N` dimension. This
    is equivalent to `group_norm` with `num_groups=H`, treating `[H, N]` as
    `[1, H * N]` with `H` groups.

    Args:
        x: Input tensor of shape `[H, N]`.
        w: Affine weight tensor of shape `[H * N]`.
        b: Affine bias tensor of shape `[H * N]`.
        eps: Normalization epsilon.

    Returns:
        The normalized tensor of shape `[H * N]`.
    """
    h, n = x.shape
    return F.group_norm(x.reshape(1, h * n), h, w, b, eps).reshape(-1)


def DPLR_RWKV(
    S: Tensor,
    R: Tensor,
    W: Tensor,
    K: Tensor,
    V: Tensor,
    A: Tensor,
    B: Tensor,
) -> tuple[Tensor, Tensor]:
    """DPLR state update for RWKV7 time-mixing.

    S = S*W + S@A@Bᵀ + V⊗K;  y = S @ R.

    The einsum form is kept as-is: rewriting the first/third terms to
    broadcast mul (S*W[:,None,:], V[:,:,None]*K[:,None,:]) was numerically
    identical but increased gemv kernel dispatch count (profiler showed
    gemv2T calls 3104 -> 4656), slowing CUDA forward. einsum routes these
    through a fused bmm path instead.

    Args:
        S: State tensor of shape `[H, N, N]`.
        R: Receptance, `[H, N]`.
        W: Decay gate, `[H, N]`.
        K: Key, `[H, N]`.
        V: Value, `[H, N]`.
        A: kk-normalized key, `[H, N]`.
        B: `-kk * a`, `[H, N]`.

    Returns:
        (y, S): output `[H, N]` and updated state `[H, N, N]`.
    """
    S = (
        torch.einsum("hvk,hk->hvk", S, W)
        + torch.einsum("hva,ha,hb->hvb", S, A, B)
        + torch.einsum("hv,hk->hvk", V, K)
    )
    return torch.einsum("hvk,hk->hv", S, R), S


class RWKV7:
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
        self.emb = LAYER_NORM(
            W["emb.weight"], W["blocks.0.ln0.weight"], W["blocks.0.ln0.bias"]
        )
        self.ln_outW, self.ln_outB, self.head = (
            W["ln_out.weight"],
            W["ln_out.bias"],
            W["head.weight"].T,
        )
        self.layers = [
            (self.make_TMIX(i), self.make_CMIX(i)) for i in range(self.n_layer)
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

    def HEAD(self, X: Tensor) -> Tensor:
        return X @ self.head

    def run_one(
        self, token: int, S: list[list[dict[str, Tensor]]]
    ) -> tuple[Tensor, list[list[dict[str, Tensor]]]]:
        """Advance one token and return `(logits, state)`.

        Args:
            token: Input token id.
            S: Current model state.

        Returns:
            Updated logits and state.
        """
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
        S = self.zero_state() if S is None else S
        logits: Tensor | None = None
        for token in tokens:
            logits, S = self.run_one(int(token), S)
        if logits is None:
            raise RuntimeError("forward received an empty token sequence")
        return logits, S

    def EMB(self, token: int) -> Tensor:
        return self.emb[token]

    def NORM(self, X: Tensor) -> Tensor:
        return LAYER_NORM(X, self.ln_outW, self.ln_outB)

    def make_TMIX(self, i: int):
        p, W, H, N = f"blocks.{i}.att.", self.W, self.H, self.N
        lnW, lnB = W[f"blocks.{i}.ln1.weight"], W[f"blocks.{i}.ln1.bias"]
        # Weights stored as (1,1,N_EMBD); flatten to 1D for the fused LERP kernel.
        x_x = tuple(W[p + n].reshape(-1) for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"))
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

        def layer(
            x0: Tensor, v_first: Tensor | None, state: dict[str, Tensor]
        ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
            x = LAYER_NORM(x0, lnW, lnB)
            prev, state["x"] = state["x"], x
            # Fuse 6 token-shift LERPs into a single tilelang kernel.
            xr, xw, xk, xv, xa, xg = fused_lerp6(x, prev, *x_x)

            r, k, v = xr @ rW, xk @ kW, xv @ vW
            if v_first is None:
                v_first = v
            else:
                v = LERP(v, v_first, SIGMOID(v0 + xv @ v1 @ v2))
            w = torch.exp(-SIGMOID(w0 + torch.tanh(xw @ w1) @ w2) / _SQRT_E)
            a = SIGMOID(a0 + xa @ a1 @ a2)
            kk = k * k_k
            k = LERP(k, k * a, k_a)
            r, w, k, v, kk, a = [z.reshape(H, N) for z in (r, w, k, v, kk, a)]
            kk = L2_RWKV(kk)

            y, state["rnn"] = DPLR_RWKV(state["rnn"], r, w, k, v, kk, -kk * a)
            y = GROUP_NORM(y, ln_xW, ln_xB, 64e-5)
            y += (torch.sum(r * k * r_k, dim=1, keepdim=True) * v).reshape(-1)
            g = SIGMOID(xg @ g1) @ g2
            return (x0 + (y * g) @ oW), v_first, state

        return layer

    def make_CMIX(self, i: int):
        p, W = f"blocks.{i}.ffn.", self.W
        lnW, lnB, x_k, kW, vW = (
            W[f"blocks.{i}.ln2.weight"],
            W[f"blocks.{i}.ln2.bias"],
            W[p + "x_k"].reshape(-1),  # Flatten (1,1,N_EMBD) -> 1D to match TMIX fused_lerp6.
            W[p + "key.weight"].T,
            W[p + "value.weight"].T,
        )

        def layer(
            x0: Tensor, state: dict[str, Tensor]
        ) -> tuple[Tensor, dict[str, Tensor]]:
            x = LAYER_NORM(x0, lnW, lnB)
            prev, state["x"] = state["x"], x
            x = LERP(x, prev, x_k)
            return (x0 + RELUSQ(x @ kW) @ vW), state

        return layer
