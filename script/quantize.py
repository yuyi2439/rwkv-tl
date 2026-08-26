"""Offline W8A16 quantizer: checkpoint -> quantized checkpoint (.q8.pth).

Quantizes the large projection matrices to ``QTensor`` (int8 + fp16
per-group scales, stored in the kernels' ``[in, out]`` layout); everything
else is kept in fp16. ``RWKV7Weight`` detects the quantized file and builds
``QTensor`` fields transparently.

Usage::

    python script/quantize.py --in model.pth --out model.q8.pth [--group 128]
"""

from __future__ import annotations

import argparse

import torch

from rwkv_tl.quant import DEFAULT_GROUP, quant_error, quantize

# Suffixes of 2D projection weights worth quantizing (att r/k/v/o + ffn k/v).
_QUANT_SUFFIXES = (
    ".receptance.weight",
    ".key.weight",
    ".value.weight",
    ".output.weight",
)
_HEAD_KEY = "head.weight"
_MIN_NUMEL = 256 * 256  # skip small matrices: quantizing them is pure loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--group", type=int, default=DEFAULT_GROUP)
    args = ap.parse_args()

    W = torch.load(args.inp, map_location="cpu", weights_only=False)
    n_q = n_keep = 0
    bytes_before = bytes_after = 0
    for k in list(W):
        v = W[k]
        if not isinstance(v, torch.Tensor) or v.dim() != 2:
            continue
        if not (k.endswith(_QUANT_SUFFIXES) or k == _HEAD_KEY):
            continue
        if v.numel() < _MIN_NUMEL:
            continue
        wt = v.T.contiguous().float()  # [out, in] -> [in, out] (kernel layout)
        qt = quantize(wt, args.group)
        err = quant_error(wt, qt)
        W[k] = qt
        n_q += 1
        b0 = v.numel() * v.element_size()
        b1 = qt.q.numel() + qt.s.numel() * 2
        bytes_before += b0
        bytes_after += b1
        print(
            f"{k:42s} {tuple(v.shape)!s:14s} -> int8 g={args.group} "
            f"rel={err['rel']:.4%} cos={err['cos']:.6f} "
            f"{b0 / 1e6:.1f}MB->{b1 / 1e6:.1f}MB"
        )
    for v in W.values():
        if isinstance(v, torch.Tensor):
            n_keep += 1

    torch.save(W, args.out)
    print(
        f"\nquantized {n_q} tensors "
        f"({bytes_before / 1e6:.0f}MB -> {bytes_after / 1e6:.0f}MB, "
        f"{(1 - bytes_after / bytes_before) * 100:.0f}% saved), "
        f"{n_keep} fp16 tensors kept -> {args.out}"
    )


if __name__ == "__main__":
    main()
