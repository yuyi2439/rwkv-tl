from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from . import operators  # noqa: F401  (registers torch.library custom ops)
from ._compat import maybe_torch_compile, supports_native_bf16
from .kernels import (
    fused_a_kk_k,
    fused_dplr,
    fused_dplr_T,
    fused_gn_rkrk,
    fused_l2norm_neg_kk_a,
    fused_lerp1_copy,
    fused_lerp6_rkv_copy,
    fused_rkv_gemm,
    fused_v_gate,
    fused_w_gate,
)
from .model import RWKV7Weight
from .state import State


def RELUSQ(x: Tensor) -> Tensor:
    """Squared ReLU; `F.relu(x) ** 2` fuses to a single kernel."""
    return F.relu(x) ** 2


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


class RWKV7:
    def __init__(
        self,
        w: RWKV7Weight,
        *,
        is_torch_compile: bool = True,
    ) -> None:
        self.w = w
        self._is_torch_compile = is_torch_compile
        self.N_LAYER = self.w.N_LAYER
        self.N_EMBD = self.w.N_EMBD
        self.HEAD_DIM = 64
        self.HEAD_CNT = self.N_EMBD // self.HEAD_DIM

        self.emb = LAYER_NORM(self.w.emb, self.w.ln_in.w, self.w.ln_in.b)
        # The decode closures dispatch through the registered custom ops
        # (rwkv_tl::*) only when torch.compile is enabled for this device, so
        # dynamo can trace a single graph. On devices where we stay eager
        # (no native bf16), the closures call the raw tilelang kernels to avoid
        # the per-call custom-op dispatch overhead.
        use_custom_ops = self._is_torch_compile and supports_native_bf16(
            self.emb.device.type
        )
        self._use_custom_ops = use_custom_ops
        self.layers = [
            (self.make_TMIX(i, use_custom_ops), self.make_CMIX(i, use_custom_ops))
            for i in range(self.N_LAYER)
        ]
        # Prefill path is built eagerly, but both paths reuse shared caches.
        # prefill is NOT torch.compile'd: each distinct prompt length would
        # recompile a fresh graph (minutes on 0.4B, GPU idle meanwhile) for a
        # <1.5x win, which is not worth it. Keep it eager.
        self.layers_batch = [
            (self.make_TMIX_batch(i), self.make_CMIX_batch(i))
            for i in range(self.N_LAYER)
        ]

    def HEAD(self, x: Tensor) -> Tensor:
        return self.w.head @ x

    def LN_OUT(self, X: Tensor) -> Tensor:
        return LAYER_NORM(X, self.w.ln_out.w, self.w.ln_out.b)

    @maybe_torch_compile
    def decode(self, token: int, S: State) -> tuple[Tensor, State]:
        """Advance one token and return `(logits, state)`.

        Args:
            token: Input token id.
            S: Current model state.

        Returns:
            Updated logits and state.
        """
        x = self.EMB(token)
        v_first: Tensor | None = None

        for (TM, CM), tmix_state, cmix_state in zip(self.layers, S.tmix, S.cmix):
            x, v_first, tmix_state = TM(x, v_first, tmix_state)
            x, cmix_state = CM(x, cmix_state)

        x = LAYER_NORM(x, self.w.ln_out.w, self.w.ln_out.b)
        return self.HEAD(x), S

    def forward(
        self,
        tokens: list[int] | Tensor,
        S: State,
    ) -> tuple[Tensor, State]:
        """Run inference over a token sequence.

        Single-token inputs use the `decode` path. Multi-token inputs are
        routed to `prefill` for batched GEMM-heavy processing.
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
        # Batched prefill: [T, C] matmuls instead of per-token GEMV loops.
        # DPLR stays per-token (serial state). Returns last-token logits.
        if isinstance(tokens, Tensor):
            tok = tokens.reshape(-1)
        else:
            tok = torch.as_tensor(
                list(tokens), dtype=torch.long, device=self.emb.device
            )
        if tok.numel() == 0:
            raise RuntimeError("prefill received an empty token sequence")
        # GPU tensor indexing keeps the whole prefill path CUDA-side (graph-safe).
        x = self.emb[tok]
        v_first: Tensor | None = None
        for (TM, CM), tmix_state, cmix_state in zip(self.layers_batch, S.tmix, S.cmix):
            x, v_first, tmix_state = TM(x, v_first, tmix_state)
            x, cmix_state = CM(x, cmix_state)
        return self.HEAD(LAYER_NORM(x[-1], self.w.ln_out.w, self.w.ln_out.b)), S

    def generate(
        self,
        tokens: list[int] | Tensor,
        S: State,
        max_tokens: int,
    ) -> tuple[list[int], State]:
        """Generate autoregressively from a prompt.

        Args:
            tokens: Prompt tokens, then greedily continued.
            S: Model state.
            max_tokens: Number of tokens to generate after the prompt.

        Returns:
            (generated tokens, final state).
        """
        logits, S = self.forward(tokens, S)
        out: list[int] = []
        for _ in range(max_tokens):
            token = int(torch.argmax(logits))
            out.append(token)
            logits, S = self.decode(token, S)
        return out, S

    def EMB(self, token: int) -> Tensor:
        return self.emb[token]

    def make_TMIX(self, i: int, use_custom_ops: bool):
        H, N = self.HEAD_CNT, self.HEAD_DIM
        b = self.w.blocks[i]
        att = b.att
        rWt = att.receptance_weight.T
        kWt = att.key_weight.T
        vWt = att.value_weight.T
        lnW, lnB = b.ln1.w, b.ln1.b
        x_x = (
            att.x_r.reshape(-1),
            att.x_w.reshape(-1),
            att.x_k.reshape(-1),
            att.x_v.reshape(-1),
            att.x_a.reshape(-1),
            att.x_g.reshape(-1),
        )
        oW = att.output_weight
        rWt_stack = torch.stack([rWt, kWt, vWt], dim=0).contiguous()
        w0, w1, w2 = att.w0.reshape(-1), att.w1, att.w2
        a0, a1, a2 = att.a0.reshape(-1), att.a1, att.a2
        v0, v1, v2 = att.v0.reshape(-1), att.v1, att.v2
        g1, g2 = att.g1, att.g2
        k_k, k_a, r_k = att.k_k.reshape(-1), att.k_a.reshape(-1), att.r_k
        ln_xW, ln_xB = att.ln_x_weight, att.ln_x_bias

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
        H, N = self.HEAD_CNT, self.HEAD_DIM
        b = self.w.blocks[i]
        att = b.att
        rWt = att.receptance_weight.T
        kWt = att.key_weight.T
        vWt = att.value_weight.T
        lnW, lnB = b.ln1.w, b.ln1.b
        x_x = (
            att.x_r.reshape(-1),
            att.x_w.reshape(-1),
            att.x_k.reshape(-1),
            att.x_v.reshape(-1),
            att.x_a.reshape(-1),
            att.x_g.reshape(-1),
        )
        rWt_stack = torch.stack([rWt, kWt, vWt], dim=0).contiguous()
        oWt = att.output_weight.T
        w0, w1, w2 = att.w0.reshape(-1), att.w1, att.w2
        a0, a1, a2 = att.a0.reshape(-1), att.a1, att.a2
        v0, v1, v2 = att.v0.reshape(-1), att.v1, att.v2
        g1, g2 = att.g1, att.g2
        k_k, k_a, r_k = att.k_k.reshape(-1), att.k_a.reshape(-1), att.r_k
        ln_xW, ln_xB = att.ln_x_weight, att.ln_x_bias

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

            # DPLR: single-shot over the whole sequence (one launch, serial state
            # recurrence inside), replacing the per-token Python loop.
            y, _ = fused_dplr_T(state["rnn"], r, w, k, v, kk_norm, B)

            y_flat = F.group_norm(y.reshape(T_len, H * N), H, ln_xW, ln_xB, 64e-5)
            rkrk = (r * k * r_k).sum(dim=2, keepdim=True)
            y_out = (y_flat.view(T_len, H, N) + rkrk * v).reshape(T_len, H * N)
            g = torch.sigmoid(xg @ g1) @ g2
            return x0 + (y_out * g) @ oWt, v_first, state

        return layer

    def make_CMIX(self, i: int, use_custom_ops: bool):
        b = self.w.blocks[i]
        ffn = b.ffn
        lnW, lnB = b.ln2.w, b.ln2.b
        x_k, kW, vW = ffn.x_k.reshape(-1), ffn.key_weight, ffn.value_weight
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
        b = self.w.blocks[i]
        ffn = b.ffn
        lnW, lnB = b.ln2.w, b.ln2.b
        x_k = ffn.x_k.reshape(-1)
        kWt, vWt = ffn.key_weight.T, ffn.value_weight.T

        def layer(
            x0: Tensor, state: dict[str, Tensor]
        ) -> tuple[Tensor, dict[str, Tensor]]:
            x_ln = LAYER_NORM(x0, lnW, lnB)
            prev = torch.cat([state["x"].unsqueeze(0), x_ln[:-1]], dim=0)
            x = x_ln + x_k * (prev - x_ln)
            state["x"] = x_ln[-1]
            return x0 + RELUSQ(x @ kWt) @ vWt, state

        return layer
