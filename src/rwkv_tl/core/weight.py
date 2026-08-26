import torch
import torch.nn.functional as F
from torch import Tensor

from rwkv_tl.quant import QTensor, stack_qtensors


def _load_wt(W, key: str) -> Tensor | QTensor:
    """A projection weight in kernel layout ``[in, out]``.

    Quantized checkpoints already store ``QTensor`` in this layout; plain
    checkpoints store ``[out, in]`` and are transposed here.
    """
    v = W[key]
    if isinstance(v, QTensor):
        return v
    return v.T.contiguous()


class LNWeight:
    w: Tensor
    b: Tensor

    def __init__(self, W, prefix: str):
        self.w = W[f"{prefix}.weight"]
        self.b = W[f"{prefix}.bias"]

    def __call__(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.w, self.b)


class RWKV7ATTWeight:
    """Attention block weights.

    ``rWt/kWt/vWt/oWt`` are stored **transposed** (``[in, out]``, i.e. the
    checkpoint's ``[out, in]`` transposed), because every compute path uses
    them as ``x @ W`` (matrix multiply with the input on the left). Storing
    the transposed form once avoids a per-path transpose copy. ``rkvWt`` is
    the three projections stacked into one ``[3, C, C]`` tensor so decode and
    prefill share a single r/k/v batch.

    The rank-in low-rank gates (``w1/a1/v1/g1``, ``[C, R]``) are stored in
    BOTH orientations: ``*1`` for the batched ``x @ w1`` form and ``*1t``
    (``[R, C]``) for the per-token ``w1t @ x`` form.
    """

    ln_pre: LNWeight
    ln_x: LNWeight

    x_rkvwag: Tensor

    v0: Tensor
    v1: Tensor
    v1t: Tensor
    v2t: Tensor
    w0: Tensor
    w1: Tensor
    w1t: Tensor
    w2t: Tensor
    a0: Tensor
    a1: Tensor
    a1t: Tensor
    a2t: Tensor
    g1: Tensor
    g1t: Tensor
    g2t: Tensor

    r_k: Tensor
    k_k: Tensor
    k_a: Tensor

    rkvWt: Tensor
    oWt: Tensor

    def __init__(self, W, prefix: str, ln_pre: LNWeight):
        self.ln_pre = ln_pre
        self.ln_x = LNWeight(W, f"{prefix}.ln_x")

        self.x_rkvwag = torch.stack(
            (
                W[f"{prefix}.x_r"].squeeze(),
                W[f"{prefix}.x_k"].squeeze(),
                W[f"{prefix}.x_v"].squeeze(),
                W[f"{prefix}.x_w"].squeeze(),
                W[f"{prefix}.x_a"].squeeze(),
                W[f"{prefix}.x_g"].squeeze(),
            ),
            dim=0,
        )

        self.v0 = W[f"{prefix}.v0"].squeeze()
        self.v1 = W[f"{prefix}.v1"]
        self.v1t = W[f"{prefix}.v1"].T.contiguous()
        self.v2t = W[f"{prefix}.v2"].T.contiguous()
        self.w0 = W[f"{prefix}.w0"].squeeze()
        self.w1 = W[f"{prefix}.w1"]
        self.w1t = W[f"{prefix}.w1"].T.contiguous()
        self.w2t = W[f"{prefix}.w2"].T.contiguous()
        self.a0 = W[f"{prefix}.a0"].squeeze()
        self.a1 = W[f"{prefix}.a1"]
        self.a1t = W[f"{prefix}.a1"].T.contiguous()
        self.a2t = W[f"{prefix}.a2"].T.contiguous()
        self.g1 = W[f"{prefix}.g1"]
        self.g1t = W[f"{prefix}.g1"].T.contiguous()
        self.g2t = W[f"{prefix}.g2"].T.contiguous()

        self.r_k = W[f"{prefix}.r_k"]
        self.k_k = W[f"{prefix}.k_k"]
        self.k_a = W[f"{prefix}.k_a"]

        # Transposed [in, out] (see class docstring): checkpoint stores [out, in].
        # Quantized checkpoints hold QTensor in the same layout already.
        r = _load_wt(W, f"{prefix}.receptance.weight")
        k = _load_wt(W, f"{prefix}.key.weight")
        v = _load_wt(W, f"{prefix}.value.weight")
        if isinstance(r, QTensor):
            self.rkvWt = stack_qtensors([r, k, v])
        else:
            self.rkvWt = torch.stack((r, k, v), dim=0)
        self.oWt = _load_wt(W, f"{prefix}.output.weight")


class RWKV7FFNWeight:
    """FFN block weights.

    ``kWt/vWt`` are stored **transposed** (``[in, out]``) like the attention
    matrix weights; every path uses them as ``x @ W``.
    """

    ln_pre: LNWeight

    x_k: Tensor
    kWt: Tensor
    vWt: Tensor

    def __init__(self, W, prefix: str, ln_pre: LNWeight):
        self.ln_pre = ln_pre
        self.x_k = W[f"{prefix}.x_k"].squeeze()
        self.kWt = _load_wt(W, f"{prefix}.key.weight")
        self.vWt = _load_wt(W, f"{prefix}.value.weight")


class RWKV7Block:
    att: RWKV7ATTWeight
    ffn: RWKV7FFNWeight

    def __init__(self, W, prefix: str):
        ln_pret = LNWeight(W, f"{prefix}.ln1")
        ln_prec = LNWeight(W, f"{prefix}.ln2")
        self.att = RWKV7ATTWeight(W, f"{prefix}.att", ln_pret)
        self.ffn = RWKV7FFNWeight(W, f"{prefix}.ffn", ln_prec)


class RWKV7Weight:
    dtype: torch.dtype
    device: torch.device
    L: int
    """Number of layers (blocks)."""
    C: int
    """Channel width (embedding size)."""

    emb: Tensor
    """Layer-normalized embedding (vocab size x C). Normalized once at load."""
    head: Tensor
    ln_in: LNWeight
    ln_out: LNWeight
    blocks: list[RWKV7Block]

    def __init__(
        self,
        model_path: str,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        # Checkpoints are bf16. Default converts to fp16 once at load
        # (fp16 tensor cores work on sm_75+, and fp16's 10-bit mantissa beats
        # bf16's 7-bit). Pass dtype=torch.bfloat16 to keep the raw checkpoint
        # dtype (no conversion) for the bf16 model.
        # Accumulation stays fp32 inside every kernel.
        if device is None:
            # A checkpoint saved from CUDA records cuda:0 locations; without
            # an explicit device, resolve to the default device so CPU-only
            # machines can load it too.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        W = torch.load(model_path, map_location=device, weights_only=False)
        W = {
            k: (v.to(dtype) if isinstance(v, torch.Tensor) else v) for k, v in W.items()
        }
        # fp16: [vocab, C] (used via torch.mv); QTensor: [C, vocab] kernel layout.
        self.head = W["head.weight"]
        self.ln_in = LNWeight(W, "blocks.0.ln0")
        self.ln_out = LNWeight(W, "ln_out")
        # Normalize the embedding once at load (LN0 fused in), so models can
        # use it directly without re-normalizing per construction.
        self.emb = self.ln_in(W["emb.weight"])

        self.dtype = dtype
        self.device = self.emb.device
        self.L = 1 + max(int(k.split(".")[1]) for k in W if k.startswith("blocks."))
        self.C = self.emb.shape[1]

        self.blocks = []
        for i in range(self.L):
            self.blocks.append(RWKV7Block(W, f"blocks.{i}"))
