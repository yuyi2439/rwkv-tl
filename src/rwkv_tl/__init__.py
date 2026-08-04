from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from . import operators  # noqa: F401  (registers torch.library custom ops)
from ._compat import maybe_torch_compile
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
from .model import LNWeight, RWKV7Weight
from .sampling import apply_stop, sample_logits
from .state import State


def RELUSQ(x: Tensor) -> Tensor:
    """Squared ReLU; `F.relu(x) ** 2` fuses to a single kernel."""
    return F.relu(x) ** 2


def LAYER_NORM(x: Tensor, ln: LNWeight) -> Tensor:
    """Apply LayerNorm over the last dimension.

    Args:
        x: Input tensor.
        ln: LNWeight with `w`/`b` tensors for the last dimension.

    Returns:
        The normalized tensor.
    """
    return F.layer_norm(x, (x.shape[-1],), ln.w, ln.b)


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

        self.emb = LAYER_NORM(self.w.emb, self.w.ln_in)
        # The decode closures dispatch through the registered custom ops
        # (rwkv_tl::*) only when torch.compile is enabled, so dynamo can trace a
        # single graph. Eager instances (is_torch_compile=False) call the raw
        # tilelang kernels directly to avoid per-call custom-op dispatch
        # overhead.
        use_custom_ops = self._is_torch_compile
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

    def LN_OUT(self, X: Tensor) -> Tensor:
        return LAYER_NORM(X, self.w.ln_out)

    @maybe_torch_compile
    def decode(self, token: int | Tensor, S: State) -> tuple[Tensor, State]:
        """Advance one token and return `(logits, state)`.

        Args:
            token: Input token id.
            S: Current model state.

        Returns:
            Updated logits and state.
        """
        # F.embedding is a graph-safe gather (unlike emb[tensor] advanced
        # indexing, which torch.compile can't specialize); works for both a
        # Python int token and a CUDA 0-dim/[1] tensor.
        idx = (
            token
            if isinstance(token, Tensor)
            else torch.as_tensor(token, device=self.emb.device)
        )
        x = F.embedding(idx, self.emb)
        if x.dim() > 1:
            x = x.squeeze(0)
        v_first: Tensor | None = None

        for (TM, CM), tmix_state, cmix_state in zip(self.layers, S.tmix, S.cmix):
            x, v_first = TM(x, v_first, tmix_state)
            x = CM(x, cmix_state)

        x = LAYER_NORM(x, self.w.ln_out)
        return self.w.head @ x, S

    def prefill(
        self,
        tokens: Tensor,
        S: State,
    ) -> State:
        # Batched prefill: update S in place from the token sequence.
        # [T, C] matmuls instead of per-token GEMV loops; DPLR stays per-token
        # (serial state). The layers write into S.tmix[i] / S.cmix[i] directly.
        if tokens.numel() == 0:
            raise RuntimeError("prefill received an empty token sequence")

        # GPU tensor indexing keeps the whole prefill path CUDA-side (graph-safe).
        x = self.emb[tokens]
        v_first: Tensor | None = None
        for (TM, CM), tmix_state, cmix_state in zip(self.layers_batch, S.tmix, S.cmix):
            x, v_first = TM(x, v_first, tmix_state)
            x = CM(x, cmix_state)
        return S

    def forward(
        self,
        tokens: list[int] | Tensor,
        S: State,
    ) -> tuple[Tensor, State]:
        """Run inference over a token sequence; returns ``(logits, S)``.

        Single-token inputs use `decode`. Multi-token inputs batch-fill all but
        the last token via `prefill`, then single-step the last one for the
        final logits.
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
            return self.decode(tok, S)
        S = self.prefill(tok[:-1], S)
        return self.decode(tok[-1], S)

    def generate(
        self,
        tokens: list[int] | Tensor,
        S: State,
        max_tokens: int = 32,
        *,
        temperature: float | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop: list[list[int]] | None = None,
    ) -> tuple[list[int], State]:
        """Generate autoregressively from a prompt.

        Args:
            tokens: Prompt tokens, then continued.
            S: Model state.
            max_tokens: Number of tokens to generate after the prompt.
            temperature: Softmax temperature; None or <= 0 = greedy.
            top_k: Top-k sampling (0 = off).
            top_p: Nucleus sampling threshold (1.0 = off).
            repetition_penalty: Standard repetition penalty on generated tokens
                (1.0 = off). Helps avoid degenerate repetition loops.
            stop: List of stop-token sequences. As soon as the generated tail
                matches one, the match is truncated off the end and generation
                halts. Pass tokenizer.encode("...") results, e.g.
                ``[tokenizer.encode("\\n\\nUser:")]``.

        Returns:
            (generated tokens, final state).
        """
        logits, S = self.forward(tokens, S)
        out: list[int] = []
        for _ in range(max_tokens):
            token = sample_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seen=out,
            )
            out.append(token)
            if apply_stop(out, stop):
                break
            # Pass a CUDA 0-dim tensor, not a Python int: torch.compile
            # specializes on int values, so each new token would recompile
            # decode (and blow past recompile_limit with fullgraph=True).
            logits, S = self.decode(torch.as_tensor(token, device=self.emb.device), S)
        return out, S

    def make_TMIX(self, i: int, use_custom_ops: bool):
        H, N = self.HEAD_CNT, self.HEAD_DIM
        b = self.w.blocks[i]
        att = b.att
        rWt = att.receptance_weight.T
        kWt = att.key_weight.T
        vWt = att.value_weight.T
        ln_pre = b.ln_pret

        oW = att.output_weight
        rWt_stack = torch.stack([rWt, kWt, vWt], dim=0).contiguous()
        w0, w2 = att.w0.reshape(-1), att.w2
        a0, a2 = att.a0.reshape(-1), att.a2
        v0, v2 = att.v0.reshape(-1), att.v2
        g2 = att.g2
        k_k, k_a, r_k = att.k_k.reshape(-1), att.k_a.reshape(-1), att.r_k
        # Low-rank rank-in weights are [C, R] (R << C); `x @ W` reads them
        # column-major (gemv2T). Pre-transpose to [R, C] and use torch.mv for
        # row-major access (gemv2N) -- measured ~1.65x faster on decode.
        w1t, a1t, v1t, g1t = (
            att.w1.t().contiguous(),
            att.a1.t().contiguous(),
            att.v1.t().contiguous(),
            att.g1.t().contiguous(),
        )

        # Dispatch through the registered custom ops only when torch.compile is
        # active (dynamo traces them as single-graph nodes). Eager calls go
        # straight to the raw tilelang kernels to avoid the custom-op dispatch
        # overhead per kernel launch.
        _fused_lerp6_rkv_copy: Callable[
            ..., tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        ]
        _akk: Callable[..., tuple[Tensor, Tensor, Tensor]]
        _l2kk: Callable[..., tuple[Tensor, Tensor]]
        _vgate: Callable[..., Tensor]
        _wgate: Callable[..., Tensor]
        _gn: Callable[..., Tensor]
        if use_custom_ops:
            _fused_lerp6_rkv_copy = torch.ops.rwkv_tl.fused_lerp6_rkv_copy
            _vgate = torch.ops.rwkv_tl.fused_v_gate
            _wgate = torch.ops.rwkv_tl.fused_w_gate
            _akk = torch.ops.rwkv_tl.fused_a_kk_k
            _l2kk = torch.ops.rwkv_tl.fused_l2norm_neg_kk_a
            _gn = torch.ops.rwkv_tl.fused_gn_rkrk
        else:
            _fused_lerp6_rkv_copy = fused_lerp6_rkv_copy
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
        ) -> tuple[Tensor, Tensor]:
            x = LAYER_NORM(x0, ln_pre)
            prev = state["x"]
            r, k, v, xv, xw, xa, xg = _fused_lerp6_rkv_copy(
                x,
                prev,
                *(
                    att.x_r,
                    att.x_w,
                    att.x_k,
                    att.x_v,
                    att.x_a,
                    att.x_g,
                ),
                state["x"],
                rWt_stack,
            )

            if v_first is None:
                v_first = v
            else:
                v = _vgate(v, v_first, v0, torch.mv(v1t, xv) @ v2)
            w = _wgate(torch.tanh(torch.mv(w1t, xw)) @ w2, w0)
            a, kk, k = _akk(a0, torch.mv(a1t, xa) @ a2, k, k_k, k_a)
            r, w, k, v, kk, a = [z.reshape(H, N) for z in (r, w, k, v, kk, a)]
            # Fused L2-normalize + neg*kk*a: B = -kk_norm * a, and returns the
            # normalized kk for the DPLR A term (S@kk_norm@(-kk_norm*a)).
            kk_norm, B = _l2kk(kk, a)
            y = _dplr(state["rnn"], r, w, k, v, kk_norm, B)
            # state["rnn"] is updated in-place by _dplr; no copy needed.
            # Fused GroupNorm + r*k*r_k residual: replaces GROUP_NORM + y+=
            y = _gn(y, r, k, v, r_k, att.ln_x.w, att.ln_x.b)
            g = torch.sigmoid(torch.mv(g1t, xg)) @ g2
            # addmv fuses (x0 + oW @ (y*g)) into a single GEMV+bias call
            # v_first is always a Tensor here: the None case is replaced with v
            # above and the else branch narrows it to Tensor.
            assert v_first is not None
            return torch.addmv(x0, oW, (y * g)), v_first

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
        ln_pre = b.ln_pret

        rWt_stack = torch.stack([rWt, kWt, vWt], dim=0).contiguous()
        oWt = att.output_weight.T
        w0, w1, w2 = att.w0.reshape(-1), att.w1, att.w2
        a0, a1, a2 = att.a0.reshape(-1), att.a1, att.a2
        v0, v1, v2 = att.v0.reshape(-1), att.v1, att.v2
        g1, g2 = att.g1, att.g2
        k_k, k_a, r_k = att.k_k.reshape(-1), att.k_a.reshape(-1), att.r_k

        def layer(
            x0: Tensor, v_first: Tensor | None, state: dict[str, Tensor]
        ) -> tuple[Tensor, Tensor]:
            T_len = x0.shape[0]
            x = LAYER_NORM(x0, ln_pre)
            # token-shift: prev[t] = x[t-1], prev[0] = state["x"]
            prev = torch.cat([state["x"].unsqueeze(0), x[:-1]], dim=0)
            diff = prev - x
            xr = x + att.x_r * diff
            xw = x + att.x_w * diff
            xk = x + att.x_k * diff
            xv = x + att.x_v * diff
            xa = x + att.x_a * diff
            xg = x + att.x_g * diff
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

            y_flat = F.group_norm(
                y.reshape(T_len, H * N), H, att.ln_x.w, att.ln_x.b, 64e-5
            )
            rkrk = (r * k * r_k).sum(dim=2, keepdim=True)
            y_out = (y_flat.view(T_len, H, N) + rkrk * v).reshape(T_len, H * N)
            g = torch.sigmoid(xg @ g1) @ g2
            return x0 + (y_out * g) @ oWt, v_first

        return layer

    def make_CMIX(self, i: int, use_custom_ops: bool):
        b = self.w.blocks[i]
        ffn = b.ffn
        ln_pre = b.ln_prec
        x_k = ffn.x_k.reshape(-1)
        kW = ffn.key_weight
        vW = ffn.value_weight
        _fused_lerp1_copy = (
            torch.ops.rwkv_tl.fused_lerp1_copy if use_custom_ops else fused_lerp1_copy
        )

        def layer(x0: Tensor, state: dict[str, Tensor]) -> Tensor:
            x_ln = LAYER_NORM(x0, ln_pre)
            prev = state["x"]
            # Fused single LERP + copy x_ln to state["x"] in-place.
            x = _fused_lerp1_copy(x_ln, prev, x_k, state["x"])
            return torch.addmv(x0, vW, RELUSQ(torch.mv(kW, x)))

        return layer

    def make_CMIX_batch(self, i: int):
        # Batched CMIX for prefill: [T, C] GEMM path.
        b = self.w.blocks[i]
        ffn = b.ffn
        ln_pre = b.ln_prec
        x_k = ffn.x_k.reshape(-1)
        kWt, vWt = ffn.key_weight.T, ffn.value_weight.T

        def layer(x0: Tensor, state: dict[str, Tensor]) -> Tensor:
            x_ln = LAYER_NORM(x0, ln_pre)
            prev = torch.cat([state["x"].unsqueeze(0), x_ln[:-1]], dim=0)
            x = x_ln + x_k * (prev - x_ln)
            state["x"] = x_ln[-1]
            return x0 + RELUSQ(x @ kWt) @ vWt

        return layer
