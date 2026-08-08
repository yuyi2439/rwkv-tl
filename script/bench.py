import importlib.util
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/yuyi2439/rwkv/rwkv-tl")

from demo import make_rwkv7
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7Weight

CKPT = sys.argv[1]
which = sys.argv[2]  # bf16 | mx450 | faster3a
FAST = "/home/yuyi2439/rwkv/Albatross/faster3a_2607/rwkv7_fast_v3a.py"
dev = torch.device("cuda")


def load_fast(module_path: Path, model_path: str):
    spec = importlib.util.spec_from_file_location("rwkv7_fast_v3a", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_PATH = model_path
    module.load_extensions()
    return module.RWKV7()


if which == "faster3a":
    model = load_fast(Path(FAST), CKPT)
    label = "faster3a_2607"
else:
    if which == "bf16":
        w = RWKV7Weight(CKPT, device=dev, dtype=torch.bfloat16)
        cls = make_rwkv7(dev, backend="bf16")
        label = "bf16+graph"
    elif which == "fp16":
        w = RWKV7Weight(CKPT, device=dev, dtype=torch.float16)
        cls = make_rwkv7(dev, backend="fp16")
        label = "fp16-base+graph"
    else:
        w = RWKV7Weight(CKPT, device=dev, dtype=torch.float16)
        cls = make_rwkv7(dev, backend="mx450")
        label = "mx450+graph"
    model = cls(w, is_torch_compile=False)


def run(T):
    if which == "faster3a":
        state = model.zero_state(1)
        tok = torch.arange(T, dtype=torch.long, device=dev).view(1, T)
        return lambda: model.forward(tok, state)
    S = State(model.w.L, model.w.C, 64, device=dev, dtype=model.w.dtype)
    if T == 1:
        tok = torch.tensor([7], dtype=torch.long, device=dev)
        return lambda: model.decode(tok, S)
    tok = torch.arange(T, dtype=torch.long, device=dev)
    return lambda: model.prefill(tok, S)


print(f"{label}  (0.4B)")
for T in (1, 8, 32, 64, 128):
    fn = run(T)
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    print(f"  T={T:3d}  {sorted(ts)[len(ts)//2]:8.3f} ms")
