---
name: tilelang-writer
description: >-
  How to write TileLang CUDA kernels with tilelang(0.1.13). Covers @tilelang.jit/out_idx, memory model and T.copy scopes,
  T.gemm 2D-grid tiling and budgets, T.Parallel vs thread binding,
  T.dynamic + T.macro pitfalls, per-element conditionals, and verified
  kernel-fusion patterns.
---

# TileLang Writer

Patterns for writing TileLang kernels (tilelang 0.1.13). When in doubt, write a minimal kernel, compile it, and benchmark.

## 1. Core structure

A kernel is a `@tilelang.jit` factory returning a `@T.prim_func`. It compiles
lazily on first call with concrete args and caches by `(C, DTYPE, ...)`.

```python
@tilelang.jit(out_idx=[3])          # param index 3 (out) is auto-allocated & returned
def fused_ln(C: int, DTYPE: str):
    N = T.dynamic("N")              # dynamic row count, bound at launch
    @T.prim_func
    def _impl(x: T.Tensor((N, C), DTYPE), w: T.Tensor((C,), DTYPE),
              b: T.Tensor((C,), DTYPE), out: T.Tensor((N, C), DTYPE)):
        with T.Kernel(N, threads=256) as n:
            for i in T.Parallel(C):
                out[n, i] = x[n, i]
    return _impl

out = fused_ln(768, "float16")(x, w, b)   # 3 inputs; out auto-allocated
```

- `out_idx=[k]` marks param `k` as output: do NOT pass it, tilelang allocates it.
- Without `out_idx`, every param is an input and the call binds positionally
  (a `*, out` keyword-only marker in the signature is ignored at runtime).
- Use `@functools.cache` / the jit's own caching to avoid recompiles.
- **Suggestion (not a requirement): pyright/pylance `reportInvalidTypeForm`.** The
  DSL's `T.Tensor((...), DTYPE)` call expressions in annotation positions and
  tilelang-only intrinsics trip pyright/pylance (VS Code). If the user reports
  `reportInvalidTypeForm` warnings on a tilelang file, the clean fix is to add
  `# pyright: reportInvalidTypeForm=false` at the top of that file. Only apply
  it when the warnings actually appear; it is a tooling workaround, not a
  mandatory header.

## 2. One prim_func = one host wrapper + N kernels

`with T.Kernel(...)` blocks compile to **separate `__global__` kernels** launched
in order from one host call. You can (and should) sequence phases this way —
it is *not* "one prim_func = one kernel".

```python
@T.prim_func
def _impl(...):
    with T.Kernel(T.ceildiv(HID, BN), T.ceildiv(N, BM), threads=128) as (bx, by):
        ...  # kernel 1 (grid shape A)
    with T.Kernel(T.ceildiv(C, BN), T.ceildiv(N, BM), threads=128) as (bx2, by2):
        ...  # kernel 2 (independent grid shape B)
```

- Each block keeps its own grid/thread shape.
- **Shared memory / registers do NOT survive across `with T.Kernel` blocks** —
  pass intermediates through a global tensor (`T.alloc_global(...)`).
- You CANNOT call a `@tilelang.jit` kernel from inside a prim_func (a host
  launch cannot be part of device IR: `'_thread._local' object has no
  attribute 'builder'`). Reuse logic with `T.macro` instead.

## 3. Memory model

| Allocation | Scope | Notes |
|---|---|---|
| `T.alloc_global((...)`, DTYPE)` | global | intermediate visible across T.Kernel blocks |
| `T.alloc_shared(...)` | shared | fast, per-block, freed at block end |
| `T.alloc_fragment((...), "float32")` | registers | per-thread; GEMM accumulator; disappears at kernel end |

### T.copy scope rules

`T.copy(src, dst)` works between any two of {global, shared, fragment} — the
instruction is auto-selected by direction:

- `global -> shared`: cp.async (default), **requires same dtype**.
- `shared -> global`: TMA on sm_90+, else SIMT; TMA requires same dtype.
- `global -> fragment` / `fragment -> global` / `fragment -> shared` / `shared -> fragment`:
  SIMT loop, **auto-casts dtype** (e.g. fp32 fragment -> fp16 global just works).
- `global -> global`: SIMT loop only (cp.async/TMA need shared); equivalent to
  `torch.copy_` and usually pointless.

**Pitfall:** a `T.copy` at the prim_func top level (outside any `with T.Kernel`)
fails with `Memory verification failed: Variable ... directly accessed by host
memory`. Every `T.copy` must live inside a `with T.Kernel` block.

**Pitfall:** `T.cast` is elementwise — `T.cast(buffer, dtype)` fails
(`T.cast` expects a PrimExpr, got Buffer). To down-convert a fragment, just
`T.copy(C_frag, h[...])` (auto-cast) or cast per element.

**Precision quirk:** a fragment declared `"float32"` can still come out
fp16-rounded when the surrounding computation stores to a DTYPE tensor (e.g.
a DPLR state written back as fp16-rounded despite being declared fp32). Treat
cross-call state precision as fp16-level unless verified; compute in fp32 and
only trust the final result's fp32 path.

## 4. T.gemm: tiling is everything

MMA path constraints (sm_86 uses `mma.sync.m16n8k16`):

- **M must be a multiple of 16**, **N multiple of 8**, **K multiple of 16**.
- Operands must be shared (or fragment) — never pass whole global matrices.
- Warp partition (Square policy): `num_warps = threads/32` must factor as
  `m_warp * n_warp` with `m_warp <= BM/16` and `n_warp <= BN/8`. Too many
  threads for the tile -> "No valid warp partition".

**Always use a 2D grid** (rows x output columns). A 1D grid (blocks over rows
only, serializing the whole output width inside each block) under-parallelizes
the GPU ~20x. The grid must tile **both** output dimensions:

```python
with T.Kernel(T.ceildiv(NOUT, BN), T.ceildiv(N, BM), threads=128) as (bu, bt):
    A_sh = T.alloc_shared((BM, BK), DTYPE)
    B_sh = T.alloc_shared((BK, BN), DTYPE)
    C_frag = T.alloc_fragment((BM, BN), "float32")
    T.clear(C_frag)
    for kk in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
        T.copy(x[bt * BM, kk * BK], A_sh)
        T.copy(W[kk * BK, bu * BN], B_sh)
        T.gemm(A_sh, B_sh, C_frag)
    # epilogue (relu2 / residual) straight from the register accumulator:
    for i, j in T.Parallel(BM, BN):
        v = C_frag[i, j]
        vv = T.max(v, T.float32(0.0))
        C_frag[i, j] = vv * vv
    T.copy(C_frag, out[bt * BM, bu * BN])   # fragment -> global, auto-casts
```

Budgets that couple `BM/BN/BK/stages/threads`:

- shared per block: `(BM*BK + BK*BN) * 2 bytes * num_stages <= 48KB`.
- registers per thread for the accumulator: `BM*BN*4 / threads` (keep well
  under ~200; large tiles spill and get slower).
- occupancy target: >= 2-4 blocks/SM (`blocks/SM = min(100KB/shared, 65536
  regs /(threads*regs), 1024/threads)`).
- Prefer `BN >= 64..128` (wider output tile = more MMA atoms in flight = ILP);
  keep `BK` modest (32-64). Measured sweet spot for `[N,768]x[768,3072]`:
  `BM=32, BN=128, BK=32, threads=128, stages=2`.

With a 2D grid + tuned tile, tilelang GEMM is **~parity with cuBLAS**
(pure GEMM), so the win is in **fusing epilogues** (relu2, residual, LN) — not
in beating cuBLAS at the raw GEMM.

**`T.clear` vs `clear_accum=True`:** when you accumulate across an outer K-tile
loop, `T.clear(C_frag)` once *before* the loop. `T.gemm(..., clear_accum=True)`
clears at the start of *every* call — inside a loop it zeroes the partial sum
and silently corrupts the result.

## 5. T.Parallel vs thread binding

- `tx = T.get_thread_binding(0)` is `threadIdx.x` — one scalar per thread.
  Manual SIMT: `i = bx*BLOCK + tx`.
- `for i in T.Parallel(C):` is a **logical loop**; tilelang distributes 0..C-1
  across the block's threads (contiguous chunks, each thread runs C/threads
  iterations). `i` is a logical index, not the thread id.
- `T.Parallel` **must be inside `with T.Kernel`** — outside it, tensors are
  "directly accessed by host memory".
- `T.Kernel(1, threads=BLOCK)` (single block) is only fine for tiny/memory-bound
  elementwise ops; compute-bound kernels need multi-block
  (`T.ceildiv(C, BLOCK)` blocks). `C // BLOCK` **floors and drops the tail** —
  always use `T.ceildiv(C, BLOCK)`.

## 5b. Warp-reduce kernels: threads = WARP, and the AMD lane width

`T.Kernel` defaults to 128 threads. Reduction kernels that only need one logical
warp (e.g. a per-row mean/var over a small dim) must request `threads=WARP`
(32). With the default 128, the extra 96 threads run warp shuffles while sitting
off the `threadIdx.x < 32` guard — that is undefined behavior and can produce
rare non-deterministic results that amplify through a recurrence.

On AMD, tilelang's `warp_reduce_sum` keeps **32-lane logical-warp semantics** on
both CDNA (wave64) and RDNA (wave32). Do NOT set `WARP` to the hardware
wavefront (64): the reduce would then cover only lanes 0-31 and silently drop
half the reduction. `SERIAL = DIM // WARP` stays 2 on every backend.

## 6. T.dynamic + T.macro: the object-identity pitfall

`T.dynamic("LEN")` creates a **fresh `tirx.Var` each call**. Two same-named
dynamics are *different objects*. If a macro declares its own `LEN` and the
outer prim_func declares another, the macro's `LEN` becomes an unbound free var
after inlining:

```text
InternalError: In PrimFunc _impl variables (LEN,) are used, but are not passed in as API arguments
```

(from `make_packed_api.cc` `UndefinedVars` — a var used in the body that is not
a param / buffer shape symbol).

**Fix: pass the SAME object into the macro factory.**

```python
def ln_macro(LEN, C, DTYPE, THREADS=256):     # LEN comes from the caller
    @T.macro
    def _impl(x: T.Tensor((LEN, C), DTYPE), ..., *, out: T.Tensor((LEN, C), DTYPE)):
        with T.Kernel(LEN, threads=THREADS) as n:
            ...
    return _impl

@tilelang.jit(out_idx=[3])
def ln(C, DTYPE):
    LEN = T.dynamic("LEN")
    @T.prim_func
    def _impl(x, w, b, out):
        ln_macro(LEN, C, DTYPE)(x, w, b, out=out)   # same LEN object
    return _impl
```

- A var used only in a type annotation (not the body) is also not captured by
  `get_func_nonlocals` — reference it in the body too.
- `assert LEN % X == 0` on a symbolic var **works**: it is lifted into the
  host launch shim (`host_kernel.cu`) as a runtime check
  ("Assertion failed; expected: ..., got: ...").

## 7. Illegal: runtime buffer switching

Binding a name to different buffers inside an `if` branch and using it after
the branch fails:

```python
# BROKEN: 'p' escapes the IfFrame
if n == 0:
    p = prev
else:
    p = x[n - 1, :]
for i in T.Parallel(C):
    out[n, i] = ... * p[i] ...
# -> "Immutable variable `p` is used outside its defining region!"
```

tilelang scopes names bound inside a control-flow frame (IfFrame/ForFrame).
Fix: use a **per-element conditional expression** (scalar, no buffer rebinding).
Clamp indices so a predicated select never reads out of bounds:

```python
with T.Kernel(LEN, threads=THREADS) as n:
    idx = T.max(n - 1, 0)                       # in-bounds even for n==0
    for i in T.Parallel(C):
        p_val = prev[i] if n == 0 else x[idx, i]  # scalar ternary
        out[n, i] = x[n, i] + weight[i] * (p_val - x[n, i])
```

## 8. Verified fusion strategy (RWKV7 CMIX)

- **Keep FFN up/down GEMMs as two separate 2D-grid kernels** (up+relu2 epilogue,
  down+residual epilogue). A *single* one-kernel up+relu2+down+residual forces a
  1D grid (hidden `[BM,4C]` exceeds shared, every block re-reads all weights)
  and is ~20x slower — keep it only as a reference.
- **Whole-chain fusion pays at small/medium T**: `fused_multi_cmix` (LN_pre +
  internal token-shift lerp + state copy + up + relu2 + down + residual in one
  prim_func = 5 kernels / 1 host call) beats the eager python-mediated CMIX
  ~1.4-2x at T=32..128, ~parity at T=256. Gains come from fewer kernels /
  fewer python dispatches / no `torch.cat` intermediate, not from faster GEMMs.
- Batched token-shift: `prev[n] = state for n==0 else x[n-1]` (section 7);
  update the carried state in the caller with
  `state.copy_(x_ln[-1])` (in-place for CUDA-Graph safety; single sequence B1Tn)
  or `x_ln.view(B, T, C)[:, -1, :]` for BnT.
- Batch (B>1) is only meaningful for the stateful parts (DPLR recurrence,
  token-shift, State layout `[B,C]`); GEMMs are batch-agnostic (flatten to
  `[B*T, C]`).

## 9. Pitfall checklist

- `C // BLOCK` drops the tail; use `T.ceildiv(C, BLOCK)`.
- `T.gemm` M must be `% 16 == 0`; B/K tiles shared; never whole global matrices.
- `T.Parallel` / `T.copy` must be inside `with T.Kernel`.
- `T.cast` is elementwise; use `T.copy` for buffer auto-cast.
- Macro closure dynamics must be the *same object* as the caller's.
- Names bound in an `if` cannot escape the branch; use scalar conditionals.
- `clear_accum` clears per gemm call — clear once outside a K-loop.
- 1D grid on a GEMM = ~20x under-parallelized; always 2D-grid output tiling.
- Benchmark with CUDA events + median, not single wall-clock shots (launch
  overhead ~8-12us dominates tiny kernels and makes single timings noisy).

## 10. Verification workflow

1. Write a minimal kernel; compile and run it standalone before integrating.
2. Check correctness against a torch reference (expect fp16 ~ULP error; whole
   chains ~0.1% relative).
3. Profile with `torch.profiler` to see which kernel dominates; tune
   BM/BN/BK/stages/threads against the budgets in section 4.
4. For prefill integration, benchmark the fused op vs the eager path at
   representative lengths; expect launch-count wins to shrink at large T.
