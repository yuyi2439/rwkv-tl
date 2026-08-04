import torch
from torch import Tensor, nn


class LNWeight:
    w: Tensor
    b: Tensor

    def __init__(self, W, prefix: str):
        self.w = W[f"{prefix}.weight"]
        self.b = W[f"{prefix}.bias"]


class RWKV7ATTWeight(nn.Module):
    x_r: Tensor
    x_w: Tensor
    x_k: Tensor
    x_v: Tensor
    x_a: Tensor
    x_g: Tensor

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

    r_k: Tensor
    k_k: Tensor
    k_a: Tensor

    receptance_weight: Tensor
    key_weight: Tensor
    value_weight: Tensor
    output_weight: Tensor
    ln_x: LNWeight

    def __init__(self, W, prefix: str):
        super().__init__()
        self.x_r = W[f"{prefix}.x_r"].reshape(-1)
        self.x_w = W[f"{prefix}.x_w"].reshape(-1)
        self.x_k = W[f"{prefix}.x_k"].reshape(-1)
        self.x_v = W[f"{prefix}.x_v"].reshape(-1)
        self.x_a = W[f"{prefix}.x_a"].reshape(-1)
        self.x_g = W[f"{prefix}.x_g"].reshape(-1)

        self.w0 = W[f"{prefix}.w0"]
        self.w1 = W[f"{prefix}.w1"]
        self.w2 = W[f"{prefix}.w2"]
        self.a0 = W[f"{prefix}.a0"]
        self.a1 = W[f"{prefix}.a1"]
        self.a2 = W[f"{prefix}.a2"]
        self.v0 = W[f"{prefix}.v0"]
        self.v1 = W[f"{prefix}.v1"]
        self.v2 = W[f"{prefix}.v2"]
        self.g1 = W[f"{prefix}.g1"]
        self.g2 = W[f"{prefix}.g2"]

        self.r_k = W[f"{prefix}.r_k"]
        self.k_k = W[f"{prefix}.k_k"]
        self.k_a = W[f"{prefix}.k_a"]

        self.receptance_weight = W[f"{prefix}.receptance.weight"]
        self.key_weight = W[f"{prefix}.key.weight"]
        self.value_weight = W[f"{prefix}.value.weight"]
        self.output_weight = W[f"{prefix}.output.weight"]
        self.ln_x = LNWeight(W, f"{prefix}.ln_x")


class RWKV7FFNWeight:
    x_k: Tensor
    key_weight: Tensor
    value_weight: Tensor

    def __init__(self, W, prefix: str):
        self.x_k = W[f"{prefix}.x_k"]
        self.key_weight = W[f"{prefix}.key.weight"]
        self.value_weight = W[f"{prefix}.value.weight"]


class RWKV7Block(nn.Module):
    ln_pret: LNWeight
    ln_prec: LNWeight
    att: RWKV7ATTWeight
    ffn: RWKV7FFNWeight

    def __init__(self, W, prefix: str):
        super().__init__()
        self.ln_pret = LNWeight(W, f"{prefix}.ln1")
        self.ln_prec = LNWeight(W, f"{prefix}.ln2")
        self.att = RWKV7ATTWeight(W, f"{prefix}.att")
        self.ffn = RWKV7FFNWeight(W, f"{prefix}.ffn")


class RWKV7Weight(nn.Module):
    N_LAYER: int
    N_EMBD: int

    emb: Tensor
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
        super().__init__()
        # Checkpoints are bf16. Default converts to fp16 once at load
        # (Albatross approach): fp16 tensor cores work on sm_75+, and fp16's
        # 10-bit mantissa beats bf16's 7-bit. Pass dtype=torch.bfloat16 to
        # keep the raw checkpoint dtype (no conversion) for the bf16 model.
        # Accumulation stays fp32 inside every kernel.
        W = torch.load(model_path, map_location=device)
        W = {
            k: (v.to(dtype) if isinstance(v, torch.Tensor) else v) for k, v in W.items()
        }

        self.emb = W["emb.weight"]
        self.head = W["head.weight"]
        self.ln_in = LNWeight(W, "blocks.0.ln0")
        self.ln_out = LNWeight(W, "ln_out")

        self.N_LAYER = 1 + max(
            int(k.split(".")[1]) for k in W if k.startswith("blocks.")
        )
        self.N_EMBD = self.emb.shape[1]

        self.blocks = []
        for i in range(self.N_LAYER):
            self.blocks.append(RWKV7Block(W, f"blocks.{i}"))
