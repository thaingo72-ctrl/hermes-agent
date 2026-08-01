"""Remaining legacy tool JSON-RPC handlers for the TUI gateway.

Operation, system, and management RPCs are direct service-injected domains.
This legacy module is intentionally still installed by the server loop so any
future method_ctx-owned tool handlers have the same binding path.
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


def register(server) -> None:
    """Bind this module's handlers onto ``server``'s globals and registry."""
    _registry.install(server)
