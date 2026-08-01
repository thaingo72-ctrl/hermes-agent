from __future__ import annotations

import functools
import inspect
import threading

from tui_gateway import methods_session, methods_session_state, server


SESSION_STATE_METHODS = {
    "session.status",
    "session.history",
    "session.undo",
    "session.compress",
    "session.save",
    "session.close",
    "session.branch",
    "session.interrupt",
}


def test_session_state_rpcs_are_owned_by_canonical_module():
    for name in SESSION_STATE_METHODS:
        handler = server._methods[name]
        assert isinstance(handler, functools.partial)
        assert handler.func.__module__ == methods_session_state.__name__


def test_legacy_session_module_does_not_register_session_state_rpcs():
    registered = {name for name, _handler in methods_session._registry._pending}
    assert SESSION_STATE_METHODS.isdisjoint(registered)


def test_session_state_module_uses_explicit_service_bundles():
    source = inspect.getsource(methods_session_state)
    assert "FunctionType" not in source
    assert "ContextVar" not in source
    assert "globals(" not in source
    assert "vars(" not in source
    for cls in (
        methods_session_state.RpcServices,
        methods_session_state.SessionAccessServices,
        methods_session_state.SessionViewServices,
        methods_session_state.ComputeHostServices,
        methods_session_state.CompressionServices,
        methods_session_state.BranchServices,
        methods_session_state.InterruptServices,
        methods_session_state.SessionStateServices,
    ):
        assert cls.__dataclass_params__.frozen is True


def test_session_close_releases_resume_lock_before_teardown(monkeypatch):
    claimed = {"session_key": "close-key"}
    teardown_started = threading.Event()
    release_teardown = threading.Event()
    probe_result = {}

    def fake_teardown(session, *, end_reason):
        teardown_started.set()
        probe_result["lock_free_during_teardown"] = server._session_resume_lock.acquire(
            blocking=False
        )
        if probe_result["lock_free_during_teardown"]:
            server._session_resume_lock.release()
        release_teardown.wait(timeout=2)
        return True

    monkeypatch.setattr(server, "_pop_session_by_id", lambda _sid: claimed)
    monkeypatch.setattr(server, "_teardown_popped_session", fake_teardown)

    thread = threading.Thread(
        target=lambda: server._methods["session.close"]("r", {"session_id": "sid"}),
        daemon=True,
    )
    thread.start()
    assert teardown_started.wait(timeout=2)
    release_teardown.set()
    thread.join(timeout=2)

    assert probe_result["lock_free_during_teardown"] is True


def test_session_interrupt_clears_stale_running_state(monkeypatch):
    sid = "interrupt-stale"

    class Agent:
        def __init__(self):
            self.interrupted = False

        def interrupt(self):
            self.interrupted = True

    agent = Agent()
    session = {
        "agent": agent,
        "history": [],
        "history_lock": threading.Lock(),
        "inflight_turn": {"text": "still shown"},
        "queued_prompt": {"text": "next"},
        "running": True,
        "session_key": "interrupt-key",
    }
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_tts_stream_stop", lambda: None)
    monkeypatch.setattr(server, "_clear_pending", lambda _sid=None: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: False)
    monkeypatch.setattr(
        "tools.approval.resolve_gateway_approval", lambda *args, **kwargs: None
    )
    try:
        resp = server._methods["session.interrupt"]("r", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert resp["result"] == {"status": "interrupted"}
    assert agent.interrupted is True
    assert session["running"] is False
    assert session["queued_prompt"] is None
    assert session["inflight_turn"] is None


def test_session_history_reads_snapshot_while_close_teardown_waits(monkeypatch):
    sid = "history-close"
    session = {
        "agent": None,
        "history": [{"role": "user", "content": "live"}],
        "history_lock": threading.Lock(),
        "running": False,
        "session_key": "",
    }
    server._sessions[sid] = session
    teardown_started = threading.Event()
    release_teardown = threading.Event()

    def fake_pop(_sid):
        return server._sessions.get(sid)

    def fake_teardown(_session, *, end_reason):
        teardown_started.set()
        release_teardown.wait(timeout=2)
        return True

    monkeypatch.setattr(server, "_pop_session_by_id", fake_pop)
    monkeypatch.setattr(server, "_teardown_popped_session", fake_teardown)

    thread = threading.Thread(
        target=lambda: server._methods["session.close"]("close", {"session_id": sid}),
        daemon=True,
    )
    thread.start()
    assert teardown_started.wait(timeout=2)

    resp = server._methods["session.history"]("history", {"session_id": sid})
    release_teardown.set()
    thread.join(timeout=2)
    server._sessions.pop(sid, None)

    assert resp["result"]["count"] == 1
    assert resp["result"]["messages"][0]["text"] == "live"
