"""Helpers for optional torch.compile of the decode path."""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch


def maybe_torch_compile(fn: Callable) -> Callable:
    """Decorator: optionally wrap a method with ``torch.compile``.

    Whether compilation happens is decided per-instance via
    ``self._is_torch_compile``: if False the method runs eagerly; if True the
    first call compiles the method (via the registered custom ops, so dynamo
    traces a single graph) and caches the compiled callable on the instance
    under ``self._{fn.__name__}_impl``.

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
            impl = torch.compile(eager, fullgraph=True)
            self.__dict__[cache_key] = impl
        return impl(*args, **kwargs)

    return wrapper
