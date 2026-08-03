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
    """Decorator: optionally wrap a method with ``torch.compile``.

    Whether compilation happens is decided per-instance via
    ``self._is_torch_compile``: if False the method runs eagerly; if True (and
    the device supports native bf16) the first call compiles the method and
    caches the compiled callable on the instance under
    ``self._{fn.__name__}_impl``.

    Usage::

        class Model:
            def __init__(self, *, is_torch_compile=True):
                self._is_torch_compile = is_torch_compile

            @maybe_torch_compile
            def decode(self, token, state): ...
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self._is_torch_compile:
            return fn(self, *args, **kwargs)

        cache_key = f"_{fn.__name__}_impl"
        impl = self.__dict__.get(cache_key)
        if impl is None:
            eager = fn.__get__(self, type(self))
            if supports_native_bf16(self.emb.device.type):
                impl = torch.compile(eager, fullgraph=True)
            else:
                impl = eager
            self.__dict__[cache_key] = impl
        return impl(*args, **kwargs)

    return wrapper
