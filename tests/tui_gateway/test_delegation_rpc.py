from __future__ import annotations

import threading
import types
from pathlib import Path

import tui_gateway.server as srv
from tui_gateway import methods_delegation


def _missing_session(rid):
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": 4001, "message": "session not found"},
    }


def _services(session: dict | None = None, **overrides):
    root = Path(overrides.pop("root", "."))

    def sess_nowait(params, rid):
        if session is None:
            return None, _missing_session(rid)
        return session, None

    defaults = dict(
        sess_nowait=sess_nowait,
        current_transport=lambda: None,
        stdio_transport=object(),
        enqueue_prompt=lambda session, text, transport: session.__setitem__(
            "queued_prompt", {"text": text, "transport": transport}
        ),
        record_inflight_correction=lambda session, text: session.setdefault(
            "corrections", []
        ).append(text),
        spawn_trees_root=lambda: root,
        spawn_tree_session_dir=lambda session_id: root / session_id,
        append_spawn_tree_index=lambda session_dir, entry: None,
        read_spawn_tree_index=lambda session_dir: [],
        list_active_subagents=lambda: [],
        is_spawn_paused=lambda: False,
        get_max_spawn_depth=lambda: 3,
        get_max_concurrent_children=lambda: 4,
        set_spawn_paused=lambda paused: paused,
        interrupt_subagent=lambda subagent_id: subagent_id == "child-1",
        now=lambda: 123.0,
    )
    defaults.update(overrides)
    return methods_delegation.DelegationServices(**defaults)


def test_delegation_handlers_are_registered_as_explicit_delegation_callables():
    for name in (
        "delegation.status",
        "delegation.pause",
        "subagent.interrupt",
        "spawn_tree.save",
        "spawn_tree.list",
        "spawn_tree.load",
        "session.steer",
        "session.redirect",
        "terminal.resize",
    ):
        assert srv._methods[name].func.__module__ == "tui_gateway.methods_delegation"


def test_register_uses_only_explicit_frozen_services():
    assert methods_delegation.DelegationServices.__dataclass_params__.frozen is True

    methods = {}
    session = {
        "agent": types.SimpleNamespace(steer=lambda text: True),
        "history_lock": threading.Lock(),
    }
    methods_delegation.register(methods, services=_services(session))

    result = methods["session.steer"]("1", {"session_id": "sid", "text": "focus"})

    assert result["result"] == {"status": "queued", "text": "focus"}
    assert session["corrections"] == ["focus"]
    assert session["last_active"] == 123.0


def test_concurrent_steer_redirect_and_delegation_state_share_no_module_state():
    calls = []
    lock = threading.Lock()

    class Agent:
        _supports_active_turn_redirect = True

        def steer(self, text):
            with lock:
                calls.append(("steer", text))
            return True

        def redirect(self, text):
            with lock:
                calls.append(("redirect", text))
            return True

    session = {"agent": Agent(), "history_lock": threading.Lock()}
    services = _services(
        session,
        list_active_subagents=lambda: [{"id": "child-1"}],
        is_spawn_paused=lambda: True,
    )
    methods = {}
    methods_delegation.register(methods, services=services)

    outputs = []

    def call(method, text=None):
        params = {"session_id": "sid"}
        if text is not None:
            params["text"] = text
        outputs.append(methods[method](method, params))

    threads = [
        threading.Thread(target=call, args=("session.steer", "steer one")),
        threading.Thread(target=call, args=("session.redirect", "redirect one")),
        threading.Thread(target=call, args=("delegation.status",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert {entry[0] for entry in calls} == {"steer", "redirect"}
    assert session["corrections"] == ["steer one", "redirect one"]
    assert any(
        output.get("result", {}).get("active") == [{"id": "child-1"}]
        and output["result"]["paused"] is True
        for output in outputs
    )
