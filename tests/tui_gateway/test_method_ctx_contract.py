"""Contracts for explicit TUI JSON-RPC handler dependency injection."""

from __future__ import annotations

import types
from pathlib import Path
from types import SimpleNamespace

from tui_gateway.method_ctx import HandlerRegistry
from tui_gateway import (
    methods_complete,
    methods_config,
    methods_prompt,
    methods_session,
    methods_tools,
)


def _ok(rid, result=None):
    return {"jsonrpc": "2.0", "id": rid, "result": result or {}}


def _err(rid, code, message):
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


class FakeContext(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(
            _methods={},
            _ok=_ok,
            _err=_err,
            _sessions={"s1": {"cols": 80}},
            _profile_scoped=self._profile_scoped,
            _load_cfg=lambda: {"custom_prompt": "fake prompt"},
            _respond=lambda rid, params, key, allow_expired=False: _ok(
                rid, {"status": "ok", "key": key, "value": params.get(key)}
            ),
        )
        self.profile_wrapped: list[str] = []

    def _sess_nowait(self, params, rid):
        session = self._sessions.get(params.get("session_id", ""))
        if session is None:
            return None, _err(rid, 4040, "no such session")
        return session, None

    def _profile_scoped(self, handler):
        self.profile_wrapped.append(handler.__name__)

        def wrapped(rid, params):
            return handler(rid, params)

        return wrapped


def test_registry_installs_plain_callables_without_rebinding_globals():
    ctx = FakeContext()
    registry = HandlerRegistry()

    @registry.method("example.echo")
    def echo(dep_ctx, rid, params):
        return dep_ctx._ok(rid, {"echo": params["value"]})

    registry.install(ctx)

    registered = ctx._methods["example.echo"]
    assert not isinstance(registered, types.FunctionType)
    assert registered("r1", {"value": "hi"}) == _ok("r1", {"echo": "hi"})
    assert registered.ctx is ctx
    assert "FunctionType" not in Path("tui_gateway/method_ctx.py").read_text()
    assert "__globals__" not in Path("tui_gateway/method_ctx.py").read_text()


def test_profile_scoped_handlers_are_wrapped_at_registration():
    ctx = FakeContext()
    registry = HandlerRegistry()

    @registry.method("example.profile")
    @registry.profile_scoped
    def profiled(dep_ctx, rid, params):
        return dep_ctx._ok(rid, {"profile": params.get("profile")})

    registry.install(ctx)

    assert "profiled" in ctx.profile_wrapped
    assert ctx._methods["example.profile"]("r2", {"profile": "work"}) == _ok(
        "r2", {"profile": "work"}
    )


def test_representative_split_handlers_work_with_fake_context():
    ctx = FakeContext()
    for module in (
        methods_complete,
        methods_config,
        methods_prompt,
        methods_session,
        methods_tools,
    ):
        module.register(ctx)

    assert ctx._methods["paste.collapse"]("c1", {"text": ""}) == _err(
        "c1", 4004, "empty paste"
    )
    assert ctx._methods["config.get"]("c2", {"key": "prompt"}) == _ok(
        "c2", {"prompt": "fake prompt"}
    )
    assert ctx._methods["clarify.respond"]("c3", {"answer": "A"}) == _ok(
        "c3", {"status": "ok", "key": "answer", "value": "A"}
    )
    assert ctx._methods["terminal.resize"](
        "c4", {"session_id": "s1", "cols": 132}
    ) == _ok("c4", {"cols": 132})
    assert ctx._methods["system.battery"]("c5", {})["result"]["available"] in {
        True,
        False,
    }


def test_config_set_is_owned_by_methods_config_not_server_py():
    server_text = Path("tui_gateway/server.py").read_text()
    methods_config_text = Path("tui_gateway/methods_config.py").read_text()

    assert "config.set" not in server_text
    assert "config.set intentionally stays in server.py" not in server_text
    assert "@method('config.set')" in methods_config_text
