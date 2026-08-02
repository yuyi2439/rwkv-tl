"""Device capability detection helpers with caching."""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch


@functools.cache
def supports_native_bf16(device_type: str) -> bool:
    match device_type:
        case "cpu":
            return False
        case "cuda":
            return torch.cuda.is_bf16_supported(including_emulation=False)
        case "mps":
            return True
        case _:
            raise ValueError(f"Unknown device type: {device_type}")


def maybe_torch_compile(fn: Callable) -> Callable:
    """Decorator: lazily wrap a method with ``torch.compile`` on its device.

    The device is only known at runtime (``self.emb.device``), so the compile
    decision is made on the first call per instance and cached on the instance.
    The raw method stays reachable via ``self._eager_run_one`` (set by the
    caller from ``self.run_one.__wrapped__``).

    Usage::

        class Model:
            @maybe_torch_compile
            def run_one(self, token, state): ...
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        impl = self.__dict__.get("_run_one_impl")
        if impl is None:
            eager = fn.__get__(self, type(self))
            if supports_native_bf16(self.emb.device.type):
                impl = torch.compile(eager, fullgraph=True)
            else:
                impl = eager
            self.__dict__["_run_one_impl"] = impl
        return impl(*args, **kwargs)

    return wrapper
