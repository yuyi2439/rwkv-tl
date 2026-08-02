from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor

from . import operators  # noqa: F401  (registers torch.library custom ops)
from ._compat import maybe_torch_compile, supports_native_bf16
from .kernels import (
    fused_a_kk_k,
    fused_dplr,
    fused_gn_rkrk,
    fused_l2norm_neg_kk_a,
    fused_lerp1_copy,
    fused_lerp6_rkv_copy,
    fused_rkv_gemm,
    fused_v_gate,
    fused_w_gate,
)
from .tokenizer import Tokenizer


class _TmixCache(TypedDict):
    lnW: Tensor
    lnB: Tensor
    x_x: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    oW: Tensor
    rWt_stack: Tensor
    oWt: Tensor
    w0: Tensor
    w1: Tensor
    w2: Tensor
    a0: Tensor
    a1: Tensor
    a2: Tensor
    v0: Tensor
    v1: Tensor
    v2: Tensor
    g1: Tensor
    g2: Tensor
    k_k: Tensor
    k_a: Tensor
    r_k: Tensor
    ln_xW: Tensor
    ln_xB: Tensor


class _CMixCache(TypedDict):
    lnW: Tensor
    lnB: Tensor
    x_k: Tensor
    kW: Tensor
    vW: Tensor
    kWt: Tensor
    vWt: Tensor


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
    return F.layer_norm(x, (x.shape[-1],), w, b)


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
        W: dict[str, Tensor] = torch.load(checkpoint_path)

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
            W["head.weight"],  # [V, C] contiguous; HEAD does head @ x
        )
        self._tmix_cache: dict[int, _TmixCache] = {}
        self._cmix_cache: dict[int, _CMixCache] = {}
        self._init_layer_caches(W)
        # The decode closures dispatch through the registered custom ops
        # (rwkv_tl::*) only when torch.compile is enabled for this device, so
        # dynamo can trace a single graph. On devices where we stay eager
        # (no native bf16), the closures call the raw tilelang kernels to avoid
        # the per-call custom-op dispatch overhead.
        use_custom_ops = supports_native_bf16(self.emb.device.type)
        self._use_custom_ops = use_custom_ops
        self.layers = [
            (self.make_TMIX(i, use_custom_ops), self.make_CMIX(i, use_custom_ops))
            for i in range(self.n_layer)
        ]
        # Prefill path is built eagerly, but both paths reuse shared caches.
        self.layers_batch = [
            (self.make_TMIX_batch(i), self.make_CMIX_batch(i))
            for i in range(self.n_layer)
        ]
        # torch.compile is applied lazily by the @maybe_torch_compile decorator
        # on run_one (native-bf16 devices only, e.g. sm_80+ / AMD MI). The
        # eager method is kept for testing and comparison.
        self._eager_run_one = self.run_one.__wrapped__.__get__(self, type(self))  # type: ignore[attr-defined]

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
        """Zero all state tensors in-place.

        Unlike ``zero_state``, this reuses existing buffers so tensor
        addresses stay fixed -- a requirement for CUDA Graph replay.
        """
        for layer_state in S:
            for slot in layer_state:
                for v in slot.values():
                    v.zero_()
        return S

    def HEAD(self, x: Tensor) -> Tensor:
        return self.head @ x

    @maybe_torch_compile
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
        x = self.EMB(token)
        v_first: Tensor | None = None

        for (TM, CM), s in zip(self.layers, S):
            x, v_first, s[0] = TM(x, v_first, s[0])
            x, s[1] = CM(x, s[1])

        x = LAYER_NORM(x, self.ln_outW, self.ln_outB)
        return self.HEAD(x), S

    def forward(
        self,
        tokens: list[int] | Tensor,
        S: list[list[dict[str, Tensor]]] | None = None,
    ) -> tuple[Tensor, list[list[dict[str, Tensor]]]]:
        """Run inference over a token sequence.

        Single-token inputs use the `run_one` decode path. Multi-token inputs
        are routed to `forward_prefill` for batched GEMM-heavy prefill.
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

    def EMB(self, token: int) -> Tensor:
        return self.emb[token]

    def NORM(self, X: Tensor) -> Tensor:
        return LAYER_NORM(X, self.ln_outW, self.ln_outB)

    def _init_layer_caches(self, W) -> None:
        for i in range(self.n_layer):
            p_att = f"blocks.{i}.att."
            rWt, kWt, vWt, oWt = (
                W[p_att + n + ".weight"].T
                for n in ("receptance", "key", "value", "output")
            )
            self._tmix_cache[i] = {
                "lnW": W[f"blocks.{i}.ln1.weight"],
                "lnB": W[f"blocks.{i}.ln1.bias"],
                "x_x": tuple(
                    W[p_att + n].reshape(-1)
                    for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")
                ),
                "oW": W[p_att + "output.weight"],
                "rWt_stack": torch.stack([rWt, kWt, vWt], dim=0).contiguous(),
                "oWt": oWt,
                "w0": W[p_att + "w0"].reshape(-1),
                "w1": W[p_att + "w1"],
                "w2": W[p_att + "w2"],
                "a0": W[p_att + "a0"].reshape(-1),
                "a1": W[p_att + "a1"],
                "a2": W[p_att + "a2"],
                "v0": W[p_att + "v0"].reshape(-1),
                "v1": W[p_att + "v1"],
                "v2": W[p_att + "v2"],
                "g1": W[p_att + "g1"],
                "g2": W[p_att + "g2"],
                "k_k": W[p_att + "k_k"].reshape(-1),
                "k_a": W[p_att + "k_a"].reshape(-1),
                "r_k": W[p_att + "r_k"],
                "ln_xW": W[p_att + "ln_x.weight"],
                "ln_xB": W[p_att + "ln_x.bias"],
            }

            p_ffn = f"blocks.{i}.ffn."
            self._cmix_cache[i] = {
                "lnW": W[f"blocks.{i}.ln2.weight"],
                "lnB": W[f"blocks.{i}.ln2.bias"],
                "x_k": W[p_ffn + "x_k"].reshape(-1),
                "kW": W[p_ffn + "key.weight"],
                "vW": W[p_ffn + "value.weight"],
                "kWt": W[p_ffn + "key.weight"].T,
                "vWt": W[p_ffn + "value.weight"].T,
            }

    def _get_tmix_cache(self, i: int) -> _TmixCache:
        return self._tmix_cache[i]

    def _get_cmix_cache(self, i: int) -> _CMixCache:
        return self._cmix_cache[i]

    def make_TMIX(self, i: int, use_custom_ops: bool):
        H, N = self.H, self.N
        c = self._get_tmix_cache(i)
        lnW, lnB = c["lnW"], c["lnB"]
        x_x = c["x_x"]
        oW = c["oW"]
        rWt_stack = c["rWt_stack"]
        w0, w1, w2 = c["w0"], c["w1"], c["w2"]
        a0, a1, a2 = c["a0"], c["a1"], c["a2"]
        v0, v1, v2 = c["v0"], c["v1"], c["v2"]
        g1, g2 = c["g1"], c["g2"]
        k_k, k_a, r_k = c["k_k"], c["k_a"], c["r_k"]
        ln_xW, ln_xB = c["ln_xW"], c["ln_xB"]

        # Dispatch through the registered custom ops only when torch.compile is
        # active (dynamo traces them as single-graph nodes). Eager calls go
        # straight to the raw tilelang kernels to avoid the custom-op dispatch
        # overhead per kernel launch.
        _rkv: Callable[
            ..., tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        ]
        _akk: Callable[..., tuple[Tensor, Tensor, Tensor]]
        _l2kk: Callable[..., tuple[Tensor, Tensor]]
        _vgate: Callable[..., Tensor]
        _wgate: Callable[..., Tensor]
        _gn: Callable[..., Tensor]
        if use_custom_ops:
            _rkv = torch.ops.rwkv_tl.fused_lerp6_rkv_copy
            _vgate = torch.ops.rwkv_tl.fused_v_gate
            _wgate = torch.ops.rwkv_tl.fused_w_gate
            _akk = torch.ops.rwkv_tl.fused_a_kk_k
            _l2kk = torch.ops.rwkv_tl.fused_l2norm_neg_kk_a
            _gn = torch.ops.rwkv_tl.fused_gn_rkrk
        else:
            _rkv = fused_lerp6_rkv_copy
            _vgate = fused_v_gate
            _wgate = fused_w_gate
            _akk = fused_a_kk_k
            _l2kk = fused_l2norm_neg_kk_a
            _gn = fused_gn_rkrk

        def _dplr(
            rnn: Tensor,
            r: Tensor,
            w: Tensor,
            k: Tensor,
            v: Tensor,
            kk: Tensor,
            b: Tensor,
        ) -> Tensor:
            if use_custom_ops:
                return torch.ops.rwkv_tl.fused_dplr(rnn, r, w, k, v, kk, b)
            y, _ = fused_dplr(rnn, r, w, k, v, kk, b)
            return y

        def layer(
            x0: Tensor, v_first: Tensor | None, state: dict[str, Tensor]
        ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
            x = LAYER_NORM(x0, lnW, lnB)
            prev = state["x"]
            r, k, v, xv, xw, xa, xg = _rkv(x, prev, *x_x, state["x"], rWt_stack)

            if v_first is None:
                v_first = v
            else:
                v = _vgate(v, v_first, v0, xv @ v1 @ v2)
            w = _wgate(torch.tanh(xw @ w1) @ w2, w0)
            a, kk, k = _akk(a0, xa @ a1 @ a2, k, k_k, k_a)
            r, w, k, v, kk, a = [z.reshape(H, N) for z in (r, w, k, v, kk, a)]
            # Fused L2-normalize + neg*kk*a: B = -kk_norm * a, and returns the
            # normalized kk for the DPLR A term (S@kk_norm@(-kk_norm*a)).
            kk_norm, B = _l2kk(kk, a)
            y = _dplr(state["rnn"], r, w, k, v, kk_norm, B)
            # state["rnn"] is updated in-place by _dplr; no copy needed.
            # Fused GroupNorm + r*k*r_k residual: replaces GROUP_NORM + y+=
            y = _gn(y, r, k, v, r_k, ln_xW, ln_xB)
            g = torch.sigmoid(xg @ g1) @ g2
            # addmv fuses (x0 + oW @ (y*g)) into a single GEMV+bias call
            # v_first is always a Tensor here: the None case is replaced with v
            # above and the else branch narrows it to Tensor.
            return torch.addmv(x0, oW, (y * g)), v_first, state  # type: ignore[return-value]

        return layer

    def make_TMIX_batch(self, i: int):
        # Batched TMIX for prefill: [T, C] GEMM path.
        # GEMV -> GEMM, token-shift via cat-shift, gates via broadcast.
        # DPLR stays per-token (1.3% of time, serially dependent on state).
        H, N = self.H, self.N
        c = self._get_tmix_cache(i)
        lnW, lnB = c["lnW"], c["lnB"]
        x_x = c["x_x"]
        rWt_stack, oWt = c["rWt_stack"], c["oWt"]
        w0, w1, w2 = c["w0"], c["w1"], c["w2"]
        a0, a1, a2 = c["a0"], c["a1"], c["a2"]
        v0, v1, v2 = c["v0"], c["v1"], c["v2"]
        g1, g2 = c["g1"], c["g2"]
        k_k, k_a, r_k = c["k_k"], c["k_a"], c["r_k"]
        ln_xW, ln_xB = c["ln_xW"], c["ln_xB"]

        def layer(
            x0: Tensor, v_first: Tensor | None, state: dict[str, Tensor]
        ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
            T_len = x0.shape[0]
            x = LAYER_NORM(x0, lnW, lnB)
            # token-shift: prev[t] = x[t-1], prev[0] = state["x"]
            prev = torch.cat([state["x"].unsqueeze(0), x[:-1]], dim=0)
            diff = prev - x
            xr = x + x_x[0] * diff
            xw = x + x_x[1] * diff
            xk = x + x_x[2] * diff
            xv = x + x_x[3] * diff
            xa = x + x_x[4] * diff
            xg = x + x_x[5] * diff
            state["x"] = x[-1]

            # Fused r/k/v projection: 3 separate mms -> 1 batched GEMM launch
            # (tilelang T.gemm on sm_80+, cuBLAS bmm on sm_75/CPU).
            rkv = fused_rkv_gemm(xr, xk, xv, rWt_stack)
            r, k, v = rkv[0], rkv[1], rkv[2]

            if v_first is None:
                v_first = v
            else:
                v12 = xv @ v1 @ v2
                v = v + torch.sigmoid(v0 + v12) * (v_first - v)
            # math.sqrt(math.e): w decay gate constant, matches make_TMIX.
            w = torch.exp(
                -torch.sigmoid(w0 + torch.tanh(xw @ w1) @ w2) / 1.6487212707001282
            )
            a = torch.sigmoid(a0 + (xa @ a1 @ a2))
            kk = k * k_k
            k = k + k_a * (k * a - k)

            r, w, k, v, kk, a = [z.view(T_len, H, N) for z in (r, w, k, v, kk, a)]
            den = torch.sqrt((kk * kk).sum(dim=2, keepdim=True))
            kk_norm = kk / torch.clamp(den, min=1e-12)
            B = -kk_norm * a

            # DPLR: serial over T (state-dependent), reuses fused_dplr kernel.
            y = torch.empty(T_len, H, N, dtype=x0.dtype, device=x0.device)
            for t in range(T_len):
                y_t, _ = fused_dplr(
                    state["rnn"], r[t], w[t], k[t], v[t], kk_norm[t], B[t]
                )
                y[t] = y_t

            y_flat = F.group_norm(y.reshape(T_len, H * N), H, ln_xW, ln_xB, 64e-5)
            rkrk = (r * k * r_k).sum(dim=2, keepdim=True)
            y_out = (y_flat.view(T_len, H, N) + rkrk * v).reshape(T_len, H * N)
            g = torch.sigmoid(xg @ g1) @ g2
            return x0 + (y_out * g) @ oWt, v_first, state

        return layer

    def make_CMIX(self, i: int, use_custom_ops: bool):
        c = self._get_cmix_cache(i)
        lnW, lnB = c["lnW"], c["lnB"]
        x_k, kW, vW = c["x_k"], c["kW"], c["vW"]
        _lerp1 = (
            torch.ops.rwkv_tl.fused_lerp1_copy if use_custom_ops else fused_lerp1_copy
        )

        def layer(
            x0: Tensor, state: dict[str, Tensor]
        ) -> tuple[Tensor, dict[str, Tensor]]:
            x_ln = LAYER_NORM(x0, lnW, lnB)
            prev = state["x"]
            # Fused single LERP + copy x_ln to state["x"] in-place.
            x = _lerp1(x_ln, prev, x_k, state["x"])
            return torch.addmv(x0, vW, RELUSQ(torch.mv(kW, x))), state

        return layer

    def make_CMIX_batch(self, i: int):
        # Batched CMIX for prefill: [T, C] GEMM path.
        c = self._get_cmix_cache(i)
        lnW, lnB = c["lnW"], c["lnB"]
        x_k = c["x_k"]
        kWt, vWt = c["kWt"], c["vWt"]

        def layer(
            x0: Tensor, state: dict[str, Tensor]
        ) -> tuple[Tensor, dict[str, Tensor]]:
            x_ln = LAYER_NORM(x0, lnW, lnB)
            prev = torch.cat([state["x"].unsqueeze(0), x_ln[:-1]], dim=0)
            x = x_ln + x_k * (prev - x_ln)
            state["x"] = x_ln[-1]
            return x0 + RELUSQ(x @ kWt) @ vWt, state

        return layer

    def forward_prefill(
        self,
        tokens: list[int] | Tensor,
        S: list[list[dict[str, Tensor]]] | None = None,
    ) -> tuple[Tensor, list[list[dict[str, Tensor]]]]:
        # Batched prefill: [T, C] matmuls instead of per-token GEMV loops.
        # DPLR stays per-token (serial state). Returns last-token logits.
        S = self.zero_state() if S is None else S
        if isinstance(tokens, Tensor):
            tok = tokens.reshape(-1)
        else:
            tok = torch.as_tensor(
                list(tokens), dtype=torch.long, device=self.emb.device
            )
        if tok.numel() == 0:
            raise RuntimeError("forward_prefill received an empty token sequence")
        # GPU tensor indexing keeps the whole prefill path CUDA-side (graph-safe).
        x = self.emb[tok]
        v_first: Tensor | None = None
        for (TM, CM), s in zip(self.layers_batch, S):
            x, v_first, s[0] = TM(x, v_first, s[0])
            x, s[1] = CM(x, s[1])
        return self.HEAD(self.NORM(x[-1])), S
