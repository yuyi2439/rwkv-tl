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
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7Weight

from ._rwkv7_abc import RWKV7Model


def RELUSQ(x: Tensor) -> Tensor:
    """Squared ReLU; `F.relu(x) ** 2` fuses to a single kernel."""
    return F.relu(x) ** 2


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
        super().__init__(w)
        self._is_torch_compile = is_torch_compile
        self._k = kernels
        if kernels.torch_dtype != w.dtype:
            raise ValueError(
                f"kernel dtype {kernels.torch_dtype} does not match weight dtype {w.dtype}"
            )

        use_custom_ops = self._is_torch_compile
        self._use_custom_ops = use_custom_ops
        self.layers = [
            (self.make_TMIX(i, use_custom_ops), self.make_CMIX(i, use_custom_ops))
            for i in range(self.L)
        ]
        # prefill is NOT torch.compile'd (recompiles per distinct prompt
        # length); keep it eager.
        self.layers_batch = [
            (self.make_TMIX_batch(i), self.make_CMIX_batch(i)) for i in range(self.L)
        ]

    def LN_OUT(self, X: Tensor) -> Tensor:
        return self.w.ln_out(X)

    @maybe_torch_compile
    def decode(self, token: Tensor, S: State) -> tuple[Tensor, State]:
        """Advance one token and return `(logits, state)`."""
        x = F.embedding(token, self.w.emb)
        if x.dim() > 1:
            x = x.squeeze(0)
        v_first: Tensor | None = None

        for (TM, CM), tmix_state, cmix_state in zip(self.layers, S.tmix, S.cmix):
            x, v_first = TM(x, v_first, tmix_state)
            x = CM(x, cmix_state)

        x = self.w.ln_out(x)
        return self.w.head @ x, S

    def prefill(self, tokens: Tensor, S: State) -> State:
        """Batched prefill: update S in place from the token sequence."""
        if tokens.numel() == 0:
            raise RuntimeError("prefill received an empty token sequence")
        x = self.w.emb[tokens]
        v_first: Tensor | None = None
        for (TM, CM), tmix_state, cmix_state in zip(self.layers_batch, S.tmix, S.cmix):
            x, v_first = TM(x, v_first, tmix_state)
            x = CM(x, cmix_state)
        return S

    def make_TMIX(self, i: int, use_custom_ops: bool):
        H, N = self.H, self.N
        b = self.w.blocks[i]
        att = b.att
        w0, w2 = att.w0.reshape(-1), att.w2
        a0, a2 = att.a0.reshape(-1), att.a2
        v0, v2 = att.v0.reshape(-1), att.v2
        g2 = att.g2
        k_k, k_a, r_k = att.k_k.reshape(-1), att.k_a.reshape(-1), att.r_k

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
            x = att.ln_pre(x0)
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
                att.rkvWt,
            )

            if v_first is None:
                v_first = v
            else:
                v = _vgate(v, v_first, v0, torch.mv(att.v1t, xv) @ v2)
            w = _wgate(torch.tanh(torch.mv(att.w1t, xw)) @ w2, w0)
            a, kk, k = _akk(a0, torch.mv(att.a1t, xa) @ a2, k, k_k, k_a)
            r, w, k, v, kk, a = [z.reshape(H, N) for z in (r, w, k, v, kk, a)]
            kk_norm, B = _l2kk(kk, a)
            y = _dplr(state["rnn"], r, w, k, v, kk_norm, B)
            # state["rnn"] is updated in-place by _dplr; no copy needed.
            y = _gn(y, r, k, v, r_k, att.ln_x.w, att.ln_x.b)
            g = torch.sigmoid(torch.mv(att.g1t, xg)) @ g2
            assert v_first is not None
            return torch.add(x0, (y * g) @ att.oWt), v_first

        return layer

    def make_TMIX_batch(self, i: int):
        # Batched TMIX for prefill: [T, C] GEMM path. DPLR stays per-token
        # (serial state), single-shot via fused_dplr_T.
        H, N = self.H, self.N
        b = self.w.blocks[i]
        att = b.att
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
            x = att.ln_pre(x0)
            # token-shift: prev[t] = x[t-1], prev[0] = state["x"]
            prev = torch.cat([state["x"].unsqueeze(0), x[:-1]], dim=0)
            diff = prev - x
            xr = x + att.x_r * diff
            xw = x + att.x_w * diff
            xk = x + att.x_k * diff
            xv = x + att.x_v * diff
            xa = x + att.x_a * diff
            xg = x + att.x_g * diff
            state["x"].copy_(x[-1])

            # Fused r/k/v projection (tilelang T.gemm on sm_80+, cuBLAS else).
            rkv = ks.fused_rkv_gemm(xr, xk, xv, att.rkvWt)
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
            return x0 + (y_out * g) @ att.oWt, v_first

        return layer

    def make_CMIX(self, i: int, use_custom_ops: bool):
        b = self.w.blocks[i]
        ffn = b.ffn
        x_k = ffn.x_k
        if use_custom_ops:
            _fused_lerp1_copy = torch.ops.rwkv_tl.fused_lerp1_copy
        else:
            _fused_lerp1_copy = self._k.fused_lerp1_copy

        def layer(x0: Tensor, state: dict[str, Tensor]) -> Tensor:
            x_ln = ffn.ln_pre(x0)
            prev = state["x"]
            # Fused single LERP + copy x_ln to state["x"] in-place.
            x = _fused_lerp1_copy(x_ln, prev, x_k, state["x"])
            return torch.add(x0, RELUSQ(x @ ffn.kWt) @ ffn.vWt)

        return layer

    def make_CMIX_batch(self, i: int):
        # Batched CMIX for prefill: [T, C] GEMM path.
        b = self.w.blocks[i]
        ffn = b.ffn
        x_k = ffn.x_k

        def layer(x0: Tensor, state: dict[str, Tensor]) -> Tensor:
            x_ln = ffn.ln_pre(x0)
            prev = torch.cat([state["x"].unsqueeze(0), x_ln[:-1]], dim=0)
            x = x_ln + x_k * (prev - x_ln)
            state["x"].copy_(x_ln[-1])
            return x0 + RELUSQ(x @ ffn.kWt) @ ffn.vWt

        return layer
