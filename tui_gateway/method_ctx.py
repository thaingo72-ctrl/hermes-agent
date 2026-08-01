"""Explicit dependency context for split TUI JSON-RPC handlers.

The split ``methods_*`` modules are intentionally importable without importing
``server.py``.  Their handlers accept a context object as the first argument and
resolve server-owned state/services through that object.  ``HandlerRegistry``
adapts those contextful handlers to the JSON-RPC registry's ``(rid, params)``
call shape without mutating function globals.
"""

from collections.abc import Callable
from typing import Any, Protocol


class HandlerContext(Protocol):
    """Server dependencies consumed by split handler modules.

    ``server.py`` is the production context. Tests can pass a small fake context
    exposing only the attributes exercised by a handler.
    """

    _methods: dict[str, Callable[[Any, dict], dict]]

    def _ok(self, rid: Any, result: dict | None = None) -> dict: ...

    def _err(self, rid: Any, code: int, message: str) -> dict: ...

    def _profile_scoped(self, handler: Callable[[Any, dict], dict]) -> Callable[[Any, dict], dict]: ...


class ContextualHandler:
    """Callable adapter from JSON-RPC shape to a contextful handler."""

    def __init__(self, ctx: HandlerContext, fn: Callable[[HandlerContext, Any, dict], dict]) -> None:
        self.ctx = ctx
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", type(self).__name__)
        self.__doc__ = getattr(fn, "__doc__", None)

    def __call__(self, rid: Any, params: dict) -> dict:
        return self.fn(self.ctx, rid, params)


class HandlerRegistry:
    """Deferred @method registrar used by the methods_* split modules."""

    def __init__(self) -> None:
        self._pending: list[tuple[str, Callable[[HandlerContext, Any, dict], dict]]] = []

    def method(self, name: str):
        """Drop-in for server.py's ``@method`` decorator (defers registration)."""

        def dec(fn):
            self._pending.append((name, fn))
            return fn

        return dec

    def profile_scoped(self, fn):
        """Drop-in for server.py's ``@_profile_scoped`` (applied at install)."""
        fn._hermes_profile_scoped = True
        return fn

    def install(self, ctx: HandlerContext) -> None:
        """Register pending handlers against an explicit dependency context."""
        for name, fn in self._pending:
            real: Callable[[Any, dict], dict] = ContextualHandler(ctx, fn)
            if getattr(fn, "_hermes_profile_scoped", False):
                real = ctx._profile_scoped(real)
            ctx._methods[name] = real
