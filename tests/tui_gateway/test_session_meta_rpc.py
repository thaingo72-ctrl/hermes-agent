from __future__ import annotations

import ast
import threading
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from tui_gateway import methods_session_meta as meta
from tui_gateway import server


TARGET_METHODS = {
    "message.react",
    "llm.oneshot",
    "handoff.request",
    "handoff.state",
    "handoff.fail",
    "session.usage",
    "session.context_breakdown",
}


def _ok(rid, result):
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code, message):
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _db_unavailable_error(rid, *, code):
    return _err(rid, code, "db unavailable")


def _rpc_services() -> meta.RpcServices:
    return meta.RpcServices(
        ok=_ok,
        err=_err,
        db_unavailable_error=_db_unavailable_error,
    )


def test_session_meta_handlers_are_owned_by_canonical_module():
    for name in TARGET_METHODS:
        registered = server._methods[name]
        assert isinstance(registered, partial)
        assert registered.func.__module__ == "tui_gateway.methods_session_meta"


def test_session_meta_service_bundles_are_frozen():
    services = _rpc_services()
    with pytest.raises(FrozenInstanceError):
        services.ok = lambda rid, result: result


def test_session_meta_module_avoids_legacy_rebinding_patterns():
    source = Path(meta.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_names = {"globals", "vars", "ContextVar", "FunctionType"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_names
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_names
        if isinstance(node, ast.arg):
            assert node.arg != "server"

    assert "HandlerRegistry" not in source
    assert "method_ctx" not in source


def test_llm_oneshot_uses_request_local_session_runtime_under_concurrency():
    entered = threading.Barrier(2)
    release = threading.Barrier(2)
    calls: list[tuple[str, dict | None]] = []
    calls_lock = threading.Lock()
    sessions = {
        "sid-a": {"agent": SimpleNamespace(provider="prov-a")},
        "sid-b": {"agent": SimpleNamespace(provider="prov-b")},
    }

    def get_session(session_id: str):
        return sessions.get(session_id)

    def main_runtime_from_agent(agent):
        return {"provider": agent.provider}

    def run_oneshot(**kwargs):
        entered.wait(timeout=2)
        release.wait(timeout=2)
        with calls_lock:
            calls.append((kwargs["user_input"], kwargs["main_runtime"]))
        return f"text:{kwargs['user_input']}"

    lookup = meta.SessionLookupServices(
        sess_nowait=lambda params, rid: (sessions[params["session_id"]], None),
        get_session=get_session,
    )
    llm = meta.LlmOneShotServices(
        run_oneshot=run_oneshot,
        main_runtime_from_agent=main_runtime_from_agent,
        warn=lambda *_args: None,
    )
    results: dict[str, dict] = {}

    def call(name: str, session_id: str) -> None:
        results[name] = meta.llm_oneshot(
            name,
            {"session_id": session_id, "input": name, "instructions": "summarize"},
            rpc=_rpc_services(),
            lookup=lookup,
            llm=llm,
        )

    threads = [
        threading.Thread(target=call, args=("a", "sid-a")),
        threading.Thread(target=call, args=("b", "sid-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert results["a"]["result"] == {"text": "text:a"}
    assert results["b"]["result"] == {"text": "text:b"}
    assert sorted(calls) == [
        ("a", {"provider": "prov-a"}),
        ("b", {"provider": "prov-b"}),
    ]


def test_concurrent_handoff_state_is_session_local():
    records = {
        "key-a": {"state": "pending", "platform": "telegram", "error": ""},
        "key-b": {"state": "completed", "platform": "slack", "error": ""},
    }
    sessions = {
        "sid-a": {"session_key": "key-a"},
        "sid-b": {"session_key": "key-b"},
    }
    entered = threading.Barrier(2)
    release = threading.Barrier(2)

    class FakeDb:
        def get_handoff_state(self, key):
            entered.wait(timeout=2)
            release.wait(timeout=2)
            return dict(records[key])

    @contextmanager
    def session_db(_session):
        yield FakeDb()

    lookup = meta.SessionLookupServices(
        sess_nowait=lambda params, rid: (sessions[params["session_id"]], None),
        get_session=lambda session_id: sessions.get(session_id),
    )
    storage = meta.SessionStorageServices(
        session_db=session_db,
        ensure_session_db_row=lambda _session: None,
    )
    results: dict[str, dict] = {}

    def call(name: str, session_id: str) -> None:
        results[name] = meta.handoff_state(
            name,
            {"session_id": session_id},
            rpc=_rpc_services(),
            lookup=lookup,
            storage=storage,
        )

    threads = [
        threading.Thread(target=call, args=("a", "sid-a")),
        threading.Thread(target=call, args=("b", "sid-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert results["a"]["result"] == {
        "state": "pending",
        "platform": "telegram",
        "error": "",
    }
    assert results["b"]["result"] == {
        "state": "completed",
        "platform": "slack",
        "error": "",
    }
