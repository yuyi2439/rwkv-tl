"""RWKV7 variant tuned for Turing sm_75 (e.g. laptop MX450).

Same decode path as ``demo.rwkv7_fp16.RWKV7FP16`` (bandwidth-bound fp16 kernels),
but the batched-prefill GEMMs run in fp32: on Turing, cuBLAS picks
pathological fp16 tensor-core kernels for the small prefill shapes
([T, C] @ [C, C] with T=32..128), making fp16 matmul/bmm ~4-8x SLOWER than
fp32 (measured on MX450: fp16 bmm ~1.3ms vs fp32 ~0.16ms). Ampere+ is
unaffected, so the base RWKV7 keeps fp16 there.

Small-T exception: for T<=16 the fused r/k/v projection goes through the
T-specialized tilelang fp16 m16n8k8 kernel instead (it beats the fp32 cuBLAS
bmm there -- measured 0.12 vs 0.21ms @ T=8); fp32 cuBLAS stays faster at
T>=32 (0.42 vs 0.83ms @ T=128).

Decode is CUDA-Graph accelerated by default (``use_graph=True``): a single
token step is captured once and replayed. The model stays stateless -- the
graph replays against its own fixed-address shadow state and ``decode`` copies
the caller's ``State`` in/out around each replay, so any ``State`` works.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from demo._rwkv7_base import LAYER_NORM, RELUSQ
from demo.graph_decode import GraphDecoder
from demo.prefill_graph import PrefillGraph
from demo.rwkv7_fp16 import RWKV7FP16
from rwkv_tl.kernels import fused_dplr_T, fused_rkv_gemm
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7Weight

__all__ = ["RWKV7MX450"]


def _copy_state(dst: State, src: State) -> None:
    """Copy all state tensors from ``src`` into ``dst`` in place."""
    for ds, ss in zip(dst.tmix, src.tmix):
        for k in ds:
            ds[k].copy_(ss[k])
    for ds, ss in zip(dst.cmix, src.cmix):
        for k in ds:
            ds[k].copy_(ss[k])


class RWKV7MX450(RWKV7FP16):
    # Small-T prefill is launch-bound (constant ~2175 launches regardless of T),
    # so it is CUDA-graphed per exact T up to this cap; larger T runs eager
    # (launch overhead is amortized there and graph memory scales with T).
    PREFILL_GRAPH_MAX_T = 64

    def __init__(
        self,
        w: RWKV7Weight,
        *,
        is_torch_compile: bool = True,
        use_graph: bool = True,
    ) -> None:
        super().__init__(w, is_torch_compile=is_torch_compile)
        # CUDA Graph single-token decode: the captured graph replays against its
        # own fixed-address shadow state; decode copies the caller's State
        # in/out around the replay, so the model stays stateless and any State
        # works (MX450 is a CUDA-only sm_75 variant, so the graph is default-on).
        self._graph = GraphDecoder(self) if use_graph else None
        self._prefill_graphs: dict[int, PrefillGraph] = {}

    def decode(self, token: int | Tensor, S: State) -> tuple[Tensor, State]:
        """Advance one token; CUDA-graph replays when enabled, eager otherwise."""
        if self._graph is None:
            return super().decode(token, S)

        g = self._graph
        _copy_state(g.state, S)
        logits = g.step(int(token.item()) if isinstance(token, Tensor) else int(token))
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
        # Batched TMIX for prefill, GEMMs in fp32 (see module docstring).
        H, N = self.HEAD_CNT, self.HEAD_DIM
        b = self.w.blocks[i]
        att = b.att
        rWt = att.receptance_weight.T
        kWt = att.key_weight.T
        vWt = att.value_weight.T
        ln_pre = b.ln_pret

        rWt_stack = torch.stack([rWt, kWt, vWt], dim=0).contiguous().float()
        # Small-T prefill uses the tilelang fp16 m16n8k8 rkv kernel: on Turing it
        # beats the fp32 cuBLAS bmm for T<=16 (measured 0.12 vs 0.21ms @ T=8);
        # fp32 cuBLAS stays faster at T>=32 (0.42 vs 0.83ms @ T=128).
        rWt_stack16 = rWt_stack.half()
        # .contiguous() matters: a transposed fp32 view is ~2.7x slower in cuBLAS
        # (measured [128,768]@[768,3072]: 1.52 vs 0.55ms non-contiguous vs contiguous).
        oWt = att.output_weight.T.contiguous().float()
        w0, w1, w2 = att.w0.reshape(-1), att.w1, att.w2
        a0, a1, a2 = att.a0.reshape(-1), att.a1, att.a2
        v0, v1, v2 = att.v0.reshape(-1), att.v1, att.v2
        g1, g2 = att.g1, att.g2
        k_k, k_a = att.k_k.reshape(-1).float(), att.k_a.reshape(-1).float()
        r_k = att.r_k
        x_r, x_w, x_k, x_v, x_a, x_g = (
            att.x_r.float(),
            att.x_w.float(),
            att.x_k.float(),
            att.x_v.float(),
            att.x_a.float(),
            att.x_g.float(),
        )
        w0, w1, w2 = w0.float(), w1.float(), w2.float()
        a0, a1, a2 = a0.float(), a1.float(), a2.float()
        v0, v1, v2 = v0.float(), v1.float(), v2.float()
        g1, g2 = g1.float(), g2.float()

        def layer(
            x0: Tensor, v_first: Tensor | None, state: dict[str, Tensor]
        ) -> tuple[Tensor, Tensor]:
            T_len = x0.shape[0]
            x = LAYER_NORM(x0, ln_pre).float()
            # token-shift: prev[t] = x[t-1], prev[0] = state["x"]
            prev = torch.cat([state["x"].unsqueeze(0), x[:-1]], dim=0)
            diff = prev - x
            xr = x + x_r * diff
            xw = x + x_w * diff
            xk = x + x_k * diff
            xv = x + x_v * diff
            xa = x + x_a * diff
            xg = x + x_g * diff
            # state["x"] stays fp16 (decode's fused_lerp6_rkv_copy requires it);
            # in-place so CUDA-Graph prefill keeps fixed buffers.
            state["x"].copy_(x[-1])

            if T_len <= 16:
                rkv = fused_rkv_gemm(xr.half(), xk.half(), xv.half(), rWt_stack16)
                r, k, v = rkv[0].float(), rkv[1].float(), rkv[2].float()
            else:
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

            # tilelang DPLR kernel is fp16 IO; cast back just for the recurrence.
            r, w, k, v, kk, a = (z.half() for z in (r, w, k, v, kk, a))
            r, w, k, v, kk, a = [z.view(T_len, H, N) for z in (r, w, k, v, kk, a)]
            den = torch.sqrt((kk * kk).sum(dim=2, keepdim=True))
            kk_norm = kk / torch.clamp(den, min=1e-12)
            B = -kk_norm * a

            y, _ = fused_dplr_T(state["rnn"], r, w, k, v, kk_norm, B)

            y_flat = F.group_norm(
                y.reshape(T_len, H * N), H, att.ln_x.w, att.ln_x.b, 64e-5
            )
            rkrk = (r * k * r_k).sum(dim=2, keepdim=True)
            y_out = (y_flat.view(T_len, H, N) + rkrk * v).reshape(T_len, H * N)
            g = torch.sigmoid(xg @ g1) @ g2
            out = x0 + (y_out.float() * g) @ oWt
            return out.half(), v_first

        return layer

    def make_CMIX_batch(self, i: int):
        # Batched CMIX for prefill, GEMMs in fp32 (see module docstring).
        b = self.w.blocks[i]
        ffn = b.ffn
        ln_pre = b.ln_prec
        x_k = ffn.x_k.reshape(-1)
        kWt = ffn.key_weight.T.contiguous().float()
        vWt = ffn.value_weight.T.contiguous().float()

        def layer(x0: Tensor, state: dict[str, Tensor]) -> Tensor:
            x_ln = LAYER_NORM(x0, ln_pre)
            prev = torch.cat([state["x"].unsqueeze(0), x_ln[:-1]], dim=0)
            x = x_ln + x_k * (prev - x_ln)
            state["x"].copy_(x_ln[-1])
            out = x0 + RELUSQ(x.float() @ kWt) @ vWt
            return out.half()

        return layer
