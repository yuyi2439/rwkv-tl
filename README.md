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

Load a weight, then build the model (nothing is auto-detected; `backend` is
always explicit):

```python
import rwkv_tl
from rwkv_tl.core import RWKV7Weight

w = RWKV7Weight("model.pth", device="cuda")    # weight; device/dtype fixed here
model = rwkv_tl.rwkv7(w, backend="tl")         # one-call RWKV7TextModel
# decoupled equivalent:
model = rwkv_tl.rwkv7_model(w, backend="tl")   # token-level RWKV7Model
text = rwkv_tl.RWKV7TextModel(model)           # text-level wrapper
```

Text in, text out:

```python
text = model.generate("The meaning of life is",
                      max_new_tokens=64, temperature=0.8, stop="\n\n")
```

Chat (messages through the packaged chat template):

```python
answer = model.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
], max_new_tokens=128)
```

## Demo

`examples/` holds runnable walk-throughs: `basic_load_and_generate.py`
loads a checkpoint and generates text, `chat.py` chats through the template:

```bash
.venv/bin/python examples/basic_load_and_generate.py /path/to/rwkv7-0.1b.pth
.venv/bin/python examples/chat.py /path/to/rwkv7-0.1b.pth
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
  core/             # low-level/inference modules (model/state/tokenizer/
                    # weight/cuda_graph); no references outside core
  kernel/           # weight-bound tilelang operator factories
  text_model.py     # RWKV7TextModel (exposed): composes a token model
                    # (self.model) + tokenizer; tokenize/generate/chat
  rwkv7_tl.py       # tilelang fused model
  rwkv7_torch.py    # pure-PyTorch reference model
  asset/            # packaged data: vocab + chat template
script/             # chat, benchmark, profiling scripts
examples/           # runnable usage examples
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

## Performance

Decode and prefill use fused tilelang kernels with fp16 compute and fp32
accumulation (DPLR state stays fp32), CUDA-Graph accelerated on CUDA by
default. On the RTX 3060 (the current target card), small models (0.1B/0.4B)
are competitive with or ahead of the Albatross reference implementation;
large-model parity (1.5B) is explicitly not a performance goal.
`script/benchmark_rwkv7.py` measures tl vs pure-torch and vs the Albatross
faster3a_2607 reference on CUDA.
