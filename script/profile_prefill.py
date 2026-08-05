"""Profile T=32 prefill to locate the bottleneck across TMIX/CMIX stages.

Uses torch.profiler to attribute GPU time per kernel group, then prints a
stage breakdown so we can decide whether DPLR or GEMV dominates prefill.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from demo import make_rwkv7
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7Weight

CKPT = os.environ.get("RWKV_CHECKPOINT_PATH")
T = 32
TOKENS = [(i * 1103515245 + 12345) % 65536 for i in range(T)]

if not CKPT:
    raise RuntimeError("RWKV_CHECKPOINT_PATH must be set")

with torch.device("cuda"):
    w = RWKV7Weight(CKPT)
    model_cls = make_rwkv7(w.device)
    model = model_cls(w)

# warmup
with torch.device("cuda"):
    S = State(model.w.L, model.w.C, 64, device="cuda")
    for _ in range(3):
        S.reset()
        model.prefill(torch.as_tensor(TOKENS, device=model.w.device), S)
torch.cuda.synchronize()

# profile
with torch.device("cuda"):
    S = State(model.w.L, model.w.C, 64, device="cuda")
    S.reset()
with (
    profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False
    ) as prof,
    record_function("prefill_T32"),
    torch.device("cuda"),
):
    model.prefill(torch.as_tensor(TOKENS, device=model.w.device), S)

# Group kernels by name keyword
from collections import defaultdict

groups = defaultdict(lambda: [0.0, 0])  # [total_us, count]
for row in prof.key_averages():
    name = row.key
    # Exclude profiler ranges/user annotations from kernel-stage accounting.
    if name == "prefill_T32" or name.startswith("ProfilerStep"):
        continue
    t = float(
        getattr(
            row,
            "self_cuda_time_total",
            getattr(row, "self_device_time_total", 0.0),
        )
    )
    if t <= 0:
        continue
    # categorize
    key = "other"
    nl = name.lower()
    if (
        "gemv" in nl
        or "mv" in nl
        or "gemm" in nl
        or "mm" in nl
        or "addmv" in nl
        or "bmm" in nl
    ):
        key = "GEMM/GEMV"
    elif "dplr" in nl or "fused_dplr" in nl:
        key = "DPLR"
    elif "lerp" in nl:
        key = "LERP"
    elif (
        "gate" in nl
        or "sigmoid" in nl
        or "w_gate" in nl
        or "v_gate" in nl
        or "a_kk" in nl
        or "neg_kk" in nl
    ):
        key = "gates"
    elif (
        "norm" in nl
        or "gn_rkrk" in nl
        or "l2norm" in nl
        or "layer_norm" in nl
        or "group_norm" in nl
        or "fused_l2norm" in nl
        or "fused_gn" in nl
    ):
        key = "norm"
    groups[key][0] += t
    groups[key][1] += int(getattr(row, "count", 0))

total = sum(v[0] for v in groups.values())
print(f"== T={T} prefill GPU time breakdown (total={total / 1e3:.2f}ms) ==")
if total <= 0:
    print("  (no CUDA self-time samples collected)")
else:
    for k, (us, cnt) in sorted(groups.items(), key=lambda x: -x[1][0]):
        print(f"  {k:12s}: {us / 1e3:7.3f}ms  ({us / total * 100:5.1f}%)  count={cnt}")

print("\n== Top 15 self-time kernels ==")
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=15))
