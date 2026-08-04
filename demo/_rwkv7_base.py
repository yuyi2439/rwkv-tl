"""Shared RWKV7 model base, parameterized by a kernel namespace.

Subclasses bind a dtype-specific kernel namespace (``rwkv_tl.kernels.fp16`` /
``rwkv_tl.kernels.bf16``) so the same forward/decode/prefill/generate code runs
with either element type. The DPLR RNN state stays fp32 in both variants.

The decode path dispatches through the registered custom ops
(``rwkv_tl::*``, which route by input dtype) only when torch.compile is
enabled; eager instances call the raw dtype-bound kernels directly.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from rwkv_tl._compat import maybe_torch_compile
from rwkv_tl.kernels import Kernels
from rwkv_tl.sampling import apply_stop, sample_logits
from rwkv_tl.state import State
from rwkv_tl.weight import LNWeight, RWKV7Weight

from ._rwkv7_abc import RWKV7Model


def RELUSQ(x: Tensor) -> Tensor:
    """Squared ReLU; `F.relu(x) ** 2` fuses to a single kernel."""
    return F.relu(x) ** 2


def LAYER_NORM(x: Tensor, ln: LNWeight) -> Tensor:
    """Apply LayerNorm over the last dimension."""
    return F.layer_norm(x, (x.shape[-1],), ln.w, ln.b)


class RWKV7Base(RWKV7Model):
    """Dtype-agnostic RWKV7 inference model (bind via subclass).

    Args:
        w: Loaded weights (``rwkv_tl.weight.RWKV7Weight``) in the model's
            dtype (fp16 or bf16).
        kernels: Dtype-bound kernel namespace (``rwkv_tl.kernels.fp16`` or
            ``rwkv_tl.kernels.bf16``).
        is_torch_compile: Compile ``decode`` via torch.compile + custom ops.
    """

    def __init__(
        self,
        w: RWKV7Weight,
        kernels: Kernels,
        *,
        is_torch_compile: bool = True,
    ) -> None:
        self.w = w
        self._k = kernels
        self.dtype = kernels.torch_dtype
        self._is_torch_compile = is_torch_compile
        self.N_LAYER = self.w.N_LAYER
        self.N_EMBD = self.w.N_EMBD
        self.HEAD_DIM = 64
        self.HEAD_CNT = self.N_EMBD // self.HEAD_DIM

        self.emb = LAYER_NORM(self.w.emb, self.w.ln_in)
        use_custom_ops = self._is_torch_compile
        self._use_custom_ops = use_custom_ops
        self.layers = [
            (self.make_TMIX(i, use_custom_ops), self.make_CMIX(i, use_custom_ops))
            for i in range(self.N_LAYER)
        ]
        # prefill is NOT torch.compile'd (recompiles per distinct prompt
        # length); keep it eager.
        self.layers_batch = [
            (self.make_TMIX_batch(i), self.make_CMIX_batch(i))
            for i in range(self.N_LAYER)
        ]

    def LN_OUT(self, X: Tensor) -> Tensor:
        return LAYER_NORM(X, self.w.ln_out)

    @maybe_torch_compile
    def decode(self, token: int | Tensor, S: State) -> tuple[Tensor, State]:
        """Advance one token and return `(logits, state)`."""
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

    def prefill(self, tokens: Tensor, S: State) -> State:
        """Batched prefill: update S in place from the token sequence."""
        if tokens.numel() == 0:
            raise RuntimeError("prefill received an empty token sequence")
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
        """Run inference over a token sequence; returns ``(logits, S)``."""
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
        """Generate autoregressively from a prompt."""
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
            # CUDA 0-dim tensor (not a Python int) so a compiled decode
            # doesn't specialize/recompile per token value.
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
        # Low-rank rank-in weights are [C, R]; pre-transpose to [R, C] and use
        # torch.mv for row-major access (gemv2N) -- measured ~1.65x faster.
        w1t, a1t, v1t, g1t = (
            att.w1.t().contiguous(),
            att.a1.t().contiguous(),
            att.v1.t().contiguous(),
            att.g1.t().contiguous(),
        )

        _fused_lerp6_rkv_copy: Callable[..., tuple[Tensor, ...]]
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
            ks = self._k
            _fused_lerp6_rkv_copy = ks.fused_lerp6_rkv_copy
            _vgate = ks.fused_v_gate
            _wgate = ks.fused_w_gate
            _akk = ks.fused_a_kk_k
            _l2kk = ks.fused_l2norm_neg_kk_a
            _gn = ks.fused_gn_rkrk

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
            y, _ = self._k.fused_dplr(rnn, r, w, k, v, kk, b)
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
            kk_norm, B = _l2kk(kk, a)
            y = _dplr(state["rnn"], r, w, k, v, kk_norm, B)
            # state["rnn"] is updated in-place by _dplr; no copy needed.
            y = _gn(y, r, k, v, r_k, att.ln_x.w, att.ln_x.b)
            g = torch.sigmoid(torch.mv(g1t, xg)) @ g2
            assert v_first is not None
            return torch.addmv(x0, oW, (y * g)), v_first

        return layer

    def make_TMIX_batch(self, i: int):
        # Batched TMIX for prefill: [T, C] GEMM path. DPLR stays per-token
        # (serial state), single-shot via fused_dplr_T.
        H, N = self.HEAD_CNT, self.HEAD_DIM
        b = self.w.blocks[i]
        att = b.att
        rWt = att.receptance_weight.T
        kWt = att.key_weight.T
        vWt = att.value_weight.T
        ln_pre = b.ln_pret

        rWt_stack = torch.stack([rWt, kWt, vWt], dim=0).contiguous()
        # .contiguous(): a transposed cuBLAS operand is ~2.7x slower (Turing).
        oWt = att.output_weight.T.contiguous()
        w0, w1, w2 = att.w0.reshape(-1), att.w1, att.w2
        a0, a1, a2 = att.a0.reshape(-1), att.a1, att.a2
        v0, v1, v2 = att.v0.reshape(-1), att.v1, att.v2
        g1, g2 = att.g1, att.g2
        k_k, k_a, r_k = att.k_k.reshape(-1), att.k_a.reshape(-1), att.r_k
        ks = self._k

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

            # Fused r/k/v projection (tilelang T.gemm on sm_80+, cuBLAS else).
            rkv = ks.fused_rkv_gemm(xr, xk, xv, rWt_stack)
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

            # DPLR: single-shot over the whole sequence (one launch).
            y, _ = ks.fused_dplr_T(state["rnn"], r, w, k, v, kk_norm, B)

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
        if use_custom_ops:
            _fused_lerp1_copy = torch.ops.rwkv_tl.fused_lerp1_copy
        else:
            _fused_lerp1_copy = self._k.fused_lerp1_copy

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
        kWt, vWt = ffn.key_weight.T.contiguous(), ffn.value_weight.T.contiguous()

        def layer(x0: Tensor, state: dict[str, Tensor]) -> Tensor:
            x_ln = LAYER_NORM(x0, ln_pre)
            prev = torch.cat([state["x"].unsqueeze(0), x_ln[:-1]], dim=0)
            x = x_ln + x_k * (prev - x_ln)
            state["x"] = x_ln[-1]
            return x0 + RELUSQ(x @ kWt) @ vWt

        return layer
