# rwkv-tl

An RWKV7 operator library (built on TileLang) plus ready-to-use stateless
models. Import the package, point it at a checkpoint, and generate:

```python
import rwkv_tl

model = rwkv_tl.rwkv7("model-0.4b.pth")          # tilelang on CUDA, torch elsewhere
out = model.generate("Once upon a time", max_tokens=128)
print(out)
```

Everything is **stateless**: models never own runtime state. Pass a `State`
in and (optionally) get it back, or let `generate` create a fresh one.

## User API

Build a model from a checkpoint path (or a pre-loaded `RWKV7Weight`):

```python
model = rwkv_tl.rwkv7("model.pth")                       # backend auto-selected
model = rwkv_tl.rwkv7("model.pth", backend="torch")      # pure-PyTorch reference
model = rwkv_tl.RWKV7TL("model.pth")                     # explicit tilelang class
```

Text in, text out:

```python
out = model.generate("The meaning of life is", max_tokens=64, temperature=0.8)
```

Advanced users can get raw logits or drive the model token-by-token:

```python
S = model.new_state()
logits, S = model.logits("The meaning of life is", state=S)   # next-token distribution
logits, S = model.decode(token_id, S)                         # one token at a time
```

### State tune (prompt -> state, save/load)

```python
S = model.tune_state("You are a helpful assistant.")  # state tuned by a prompt
S.save("persona.pt")
S = rwkv_tl.State.load("persona.pt")
out = model.generate("Hello!", state=S, max_tokens=64)
```

## Demo

`script/demo_rwkv7.py` is a single-file, runnable walk-through of the whole
API: build a model from a checkpoint, state tune with save/load, inspect raw
logits, generate text, and decode token-by-token:

```bash
.venv/bin/python script/demo_rwkv7.py /path/to/rwkv7-0.1b.pth \
  --prompt "The meaning of life is" --max-tokens 64 \
  --tune-prompt "You are a helpful assistant." --save-state persona.pt
```

## Operator library

`rwkv_tl.kernel` is a TileLang operator library for building efficient RWKV
implementations. Every factory is **weight-bound**: it takes the compile-time
hyperparameters *and* the weights at construction and returns a callable that
only needs activations/state at call time:

```python
from rwkv_tl.kernel import ln_kernel, gemv_kernel

ln_pre = ln_kernel(C, DTYPE, ln_preW, ln_preB)   # weights captured here
x_ln = ln_pre(x0)                                 # call with activations only
```

Both granularities are supported:

- Fine-grained composable operators: `ln_kernel`, `ln_per_row_kernel`,
  `gemv_kernel`, `gemv_batch_kernel`.
- Coarse fused layer kernels: `cmix_decode_kernel`, `cmix_prefill_kernel`,
  `tmix_decode_kernel`, `tmix_prefill_kernel`.

The raw `@tilelang.jit` factories and shared macros (`gemv_macro`,
`gemv_main_macro`, ...) remain available for custom fused chains. Weights are
held by the wrapper, which is the hook for a future quantized-weight path
(int8/any4 storage + dequant fused into the kernels).

## Layout

```text
src/rwkv_tl/        # published library: models, State, Tokenizer, kernel/
  kernel/           # weight-bound tilelang operator factories
  model.py          # stateless RWKV7Model interface + text API
  rwkv7_tl.py       # tilelang fused model
  rwkv7_torch.py    # pure-PyTorch reference model
  state.py          # State (save/load included)
  tokenizer.py      # RWKV word tokenizer (vocab packaged inside the package)
script/             # chat, benchmark, profiling scripts
test/               # correctness and API tests
docs/               # benchmark reports and tuning notes (Chinese)
```

## Install and test

```bash
cd rwkv-tl
uv sync
.venv/bin/python -m pytest test/ -v
```

Kernel correctness tests need CUDA and `RWKV_CHECKPOINT_PATH`:

```bash
RWKV_CHECKPOINT_PATH=/path/to/rwkv7-g1d-0.1b.pth .venv/bin/python -m pytest test/ -v
```

The user-facing text API and the pure-torch backend also run on CPU.

`script/check_torch_vs_official.py` additionally validates the pure-torch
backend against the official RWKV-LM v7 demo (pure-torch path) on the same
checkpoint — logits must agree on argmax and top-5 for batched and per-token
decode:

```bash
.venv/bin/python script/check_torch_vs_official.py /path/to/rwkv7-0.1b.pth \
  --fast-path /path/to/RWKV-LM/RWKV-v7/rwkv_v7_demo.py
```

## Performance

Decode and prefill use fused tilelang kernels with fp16 compute and fp32
accumulation (DPLR state stays fp32), CUDA-Graph accelerated on CUDA by
default. Current numbers vs the Albatross reference implementation are in
`script/benchmark_rwkv7.md` and `docs/benchmarks/` (RTX 3060 / MX450).
`script/bench_tl_vs_torch.py` measures tl vs pure-torch on CUDA (prefill
sweep + decode), and `script/bench_tl_vs_fast.py` compares tl against the
Albatross faster3a_2607 reference implementation.
