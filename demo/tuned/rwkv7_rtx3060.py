"""RWKV7 variant tuned for Ampere+ sm_80+ CUDA (e.g. RTX 3060).

Measured on RTX 3060 (0.1B) the single biggest win is CUDA-Graph: replaying a
captured ``decode`` step and a per-T ``prefill`` graph collapses launch gaps
that dominate small models. This variant keeps the fp16 tilelang kernels (the
Ampere base path, which has native fp16 tensor cores) and layers CUDA-Graph
on top -- the same strategy as ``demo.tuned.rwkv7_mx450`` but WITHOUT the
Turing fp32-GEMM workaround (not needed on sm_80+).

    faster3a_2607   tl-fp16        tl-rtx3060
    1x1      4.78ms      10.38ms        2.20ms (CUDA-Graph decode)
    1x32     8.10ms      16.08ms        5.03ms (CUDA-Graph prefill)
    1x64     7.64ms      16.52ms        5.72ms (CUDA-Graph prefill)
    1x128    7.51ms      18.21ms       20.70ms (T>64 eager, fp16 GEMM)

Prefill is CUDA-graphed per exact T up to ``PREFILL_GRAPH_MAX_T`` (64); larger
T runs eager (launch overhead amortized, graph memory scales with T). The model
stays stateless: graphs replay against fixed-address shadow state and
``decode``/``prefill`` copy the caller's ``State`` in/out.

CUDA-Graph prefill requires the batch closures to update ``state["x"]`` in
place (``copy_``, not rebind) so tensor addresses stay fixed across replays --
that is why ``make_TMIX_batch``/``make_CMIX_batch`` are overridden here
(the fp16 base rebinds ``state["x"]``, which silently corrupts a replayed
graph).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from demo._rwkv7_base import LAYER_NORM, RELUSQ
from demo.graph_decode import GraphDecoder
from demo.prefill_graph import PrefillGraph
from demo.rwkv7_fp16 import RWKV7FP16
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7Weight

__all__ = ["RWKV7RTX3060"]


def _copy_state(dst: State, src: State) -> None:
    """Copy all state tensors from ``src`` into ``dst`` in place."""
    for ds, ss in zip(dst.tmix, src.tmix):
        for k in ds:
            ds[k].copy_(ss[k])
    for ds, ss in zip(dst.cmix, src.cmix):
        for k in ds:
            ds[k].copy_(ss[k])


class RWKV7RTX3060(RWKV7FP16):
    PREFILL_GRAPH_MAX_T = 64

    def __init__(
        self,
        w: RWKV7Weight,
        *,
        is_torch_compile: bool = True,
        use_graph: bool = True,
    ) -> None:
        super().__init__(w, is_torch_compile=is_torch_compile)
        self._graph = GraphDecoder(self) if use_graph else None
        self._prefill_graphs: dict[int, PrefillGraph] = {}

    def decode(self, token: Tensor, S: State) -> tuple[Tensor, State]:
        """Advance one token; CUDA-graph replays when enabled, eager otherwise."""
        if self._graph is None:
            return super().decode(token, S)

        g = self._graph
        _copy_state(g.state, S)
        logits = g.step(int(token.item()))
        _copy_state(S, g.state)
        return logits.clone(), S

    def prefill(self, tokens: Tensor, S: State) -> State:
        """Batched prefill; CUDA-graph replays per exact T when enabled."""
        if self._graph is None:
            return super().prefill(tokens, S)
        tok = tokens.reshape(-1)
        T = tok.numel()
        if T < 2 or T > self.PREFILL_GRAPH_MAX_T:
            return super().prefill(tok, S)
        g = self._prefill_graphs.get(T)
        if g is None:
            g = PrefillGraph(self, T)
            self._prefill_graphs[T] = g
        return g.step(tok, S)

    def make_TMIX_batch(self, i: int):
        # Same as the fp16 base but updates state["x"] in place (copy_) so the
        # captured prefill graph's tensor addresses stay fixed across replays.
        H, N = self.H, self.N
        b = self.w.blocks[i]
        att = b.att
        ln_pre = b.ln_pret
        rWt_stack = torch.stack(
            [att.receptance_weight.T, att.key_weight.T, att.value_weight.T], dim=0
        ).contiguous()
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
            prev = torch.cat([state["x"].unsqueeze(0), x[:-1]], dim=0)
            diff = prev - x
            xr = x + att.x_r * diff
            xw = x + att.x_w * diff
            xk = x + att.x_k * diff
            xv = x + att.x_v * diff
            xa = x + att.x_a * diff
            xg = x + att.x_g * diff
            state["x"].copy_(x[-1])

            rkv = ks.fused_rkv_gemm(xr, xk, xv, rWt_stack)
            r, k, v = rkv[0], rkv[1], rkv[2]

            if v_first is None:
                v_first = v
            else:
                v12 = xv @ v1 @ v2
                v = v + torch.sigmoid(v0 + v12) * (v_first - v)
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

            y, _ = ks.fused_dplr_T(state["rnn"], r, w, k, v, kk_norm, B)

            y_flat = F.group_norm(
                y.reshape(T_len, H * N), H, att.ln_x.w, att.ln_x.b, 64e-5
            )
            rkrk = (r * k * r_k).sum(dim=2, keepdim=True)
            y_out = (y_flat.view(T_len, H, N) + rkrk * v).reshape(T_len, H * N)
            g = torch.sigmoid(xg @ g1) @ g2
            return x0 + (y_out * g) @ oWt, v_first

        return layer

    def make_CMIX_batch(self, i: int):
        # Same as the fp16 base but updates state["x"] in place (copy_).
        b = self.w.blocks[i]
        ffn = b.ffn
        ln_pre = b.ln_prec
        x_k = ffn.x_k.reshape(-1)
        kWt, vWt = ffn.key_weight.T.contiguous(), ffn.value_weight.T.contiguous()

        def layer(x0: Tensor, state: dict[str, Tensor]) -> Tensor:
            x_ln = LAYER_NORM(x0, ln_pre)
            prev = torch.cat([state["x"].unsqueeze(0), x_ln[:-1]], dim=0)
            x = x_ln + x_k * (prev - x_ln)
            state["x"].copy_(x_ln[-1])
            return x0 + RELUSQ(x @ kWt) @ vWt

        return layer
