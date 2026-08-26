"""KernelOp: a tilelang kernel with its weights bound at construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _strip_handle(name: str) -> str:
    """TIR buffer params are lowered with a ``_handle`` suffix; scalars are not."""
    return name.removesuffix("_handle")


def require_bind(bind: Mapping[str, Any], names: Sequence[str], name: str) -> None:
    """Validate a weight binding against the kernel's exact weight param names."""
    missing = [n for n in names if n not in bind]
    if missing:
        raise TypeError(f"{name} missing weights: {', '.join(missing)}")
    extra = set(bind) - set(names)
    if extra:
        raise TypeError(f"{name} unexpected weights: {', '.join(sorted(extra))}")


class KernelOp:
    """A lazily-compiled tilelang kernel with a subset of params bound.

    Args:
        factory: A ``@tilelang.jit`` factory (called with ``factory_args`` on
            first use to compile the kernel).
        factory_args: Hyperparameters for ``factory`` (C/DTYPE/H/ranks/...).
        bind: Param name -> weight tensor, captured at construction.
        call: Names of the params the caller supplies at call time (in order).
        name: Display name for error messages.
    """

    def __init__(
        self,
        factory: Callable,
        factory_args: Sequence[Any],
        *,
        bind: Mapping[str, Any],
        call: Sequence[str],
        name: str | None = None,
    ) -> None:
        self._factory = factory
        self._factory_args = tuple(factory_args)
        self._bind = dict(bind)
        self._call = tuple(call)
        self._name = name or getattr(factory, "__name__", "bound_kernel")
        self._kernel: Any = None
        self._order: list[str] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def kernel(self) -> Any:
        """The compiled tilelang kernel, compiling it on first access."""
        self._ensure_compiled()
        return self._kernel

    @property
    def bound(self) -> dict[str, Any]:
        """The weight tensors captured at construction."""
        return dict(self._bind)

    def _ensure_compiled(self) -> None:
        if self._kernel is not None:
            return
        kernel = self._factory(*self._factory_args)
        names = [_strip_handle(p.name) for p in kernel.prim_func.params]
        result = set(kernel.out_idx)
        self._order = [n for i, n in enumerate(names) if i not in result]
        self._kernel = kernel

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._ensure_compiled()
        assert self._order is not None

        if len(args) > len(self._call):
            raise TypeError(
                f"{self._name} takes at most {len(self._call)} positional "
                f"arguments ({', '.join(self._call)}), got {len(args)}"
            )
        given = dict(zip(self._call, args))
        given.update(kwargs)

        missing = [n for n in self._order if n not in self._bind and n not in given]
        if missing:
            raise TypeError(
                f"{self._name} missing required arguments: {', '.join(missing)}"
            )
        unexpected = set(given) - set(self._order)
        if unexpected:
            raise TypeError(
                f"{self._name} got unexpected arguments: {', '.join(sorted(unexpected))}"
            )

        return self._kernel(
            *(given[n] if n in given else self._bind[n] for n in self._order)
        )

    def __repr__(self) -> str:
        state = "compiled" if self._kernel is not None else "lazy"
        return f"KernelOp({self._name!r}, {state}, call={self._call!r})"
