"""Regression tests for the Codex time-to-first-byte (TTFB) watchdog.

The chatgpt.com/backend-api/codex endpoint has an intermittent failure mode
where it accepts the connection but never emits a single stream event. The
watchdog in ``interruptible_api_call`` kills such a connection at a short TTFB
cutoff (instead of waiting out the much longer wall-clock stale timeout) so the
retry loop can reconnect promptly. Once any stream event arrives, the TTFB
watchdog is satisfied and a separate idle watchdog handles streams that stop
emitting SSE events.

The "bytes flowing" signal is ``agent._codex_stream_last_event_ts``, set on
*any* event by ``codex_runtime.run_codex_stream`` — so reasoning-only or
tool-call-only turns (which emit no output-text deltas) are not mistaken for a
stall.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _make_codex_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    # The watchdog is gated on the codex_responses api_mode; assert/force it so
    # the test is robust to detection-logic changes elsewhere.
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    # Keep the wall-clock stale timeout high so any early kill is unambiguously
    # the TTFB path, not the stale-call path.
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 60.0
    )
    return agent


@pytest.mark.parametrize("estimated_tokens", [10_000, 50_001, 100_001])
def test_ttfb_default_is_flat_across_codex_context_sizes(estimated_tokens):
    """Context size is not a useful predictor of a dead-on-arrival Codex
    stream. Fail fast at the same bounded cutoff and let the retry reconnect."""
    from agent import chat_completion_helpers as h

    assert h.openai_codex_ttfb_timeout_default(estimated_tokens) == 45.0


def _mark_stream_event(agent, request_guard) -> None:
    if request_guard is not None:
        assert request_guard.mark_event(time.time()) is True
    else:
        agent._codex_stream_last_event_ts = time.time()


def _guard_test_agent():
    return SimpleNamespace(
        _codex_stream_last_event_ts=None,
        _interrupt_requested=False,
        _codex_streamed_text_parts=[],
        _fire_stream_delta=lambda text: None,
        _fire_reasoning_delta=lambda text: None,
        _touch_activity=lambda status: None,
        _client_log_context=lambda: "",
    )


def test_inactive_request_guard_blocks_delayed_worker_entry():
    """A worker that starts after its timeout returned is rejected before it can
    open a stream or bind itself to state belonging to a newer retry."""
    from agent import codex_runtime as runtime

    guard = runtime.CodexRequestGuard()
    guard.deactivate()
    creates: list[dict] = []
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: creates.append(kwargs))
    )

    with pytest.raises(InterruptedError):
        runtime.run_codex_stream(
            _guard_test_agent(),
            {"model": "gpt-5.6-sol"},
            client=client,
            request_guard=guard,
        )

    assert creates == []


def test_deactivated_request_guard_suppresses_late_callbacks(monkeypatch):
    """Once a timeout deactivates the request, late events cannot mutate shared
    stream state or emit text/reasoning/first-delta callbacks."""
    from agent import codex_runtime as runtime

    emitted_text: list[str] = []
    emitted_reasoning: list[str] = []
    first_deltas: list[bool] = []
    touches: list[str] = []
    agent = _guard_test_agent()
    agent._fire_stream_delta = lambda text: emitted_text.append(text)
    agent._fire_reasoning_delta = lambda text: emitted_reasoning.append(text)
    agent._touch_activity = lambda status: touches.append(status)
    guard = runtime.CodexRequestGuard()
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: []))

    def fake_consume(
        event_stream,
        *,
        model,
        on_text_delta,
        on_reasoning_delta,
        on_first_delta,
        on_event,
        interrupt_check,
    ):
        guard.deactivate()
        on_event(SimpleNamespace(type="response.output_text.delta"))
        on_text_delta("late text")
        on_reasoning_delta("late reasoning")
        on_first_delta()
        assert interrupt_check() is True
        return SimpleNamespace(status="completed", incomplete_details=None, error=None)

    monkeypatch.setattr(runtime, "_consume_codex_event_stream", fake_consume)
    with pytest.raises(InterruptedError, match="obsolete"):
        runtime.run_codex_stream(
            agent,
            {"model": "gpt-5.6-sol"},
            client=client,
            on_first_delta=lambda: first_deltas.append(True),
            request_guard=guard,
        )

    assert emitted_text == []
    assert emitted_reasoning == []
    assert first_deltas == []
    assert touches == []
    assert agent._codex_streamed_text_parts == []


def test_active_guarded_worker_publishes_stream_parts_on_completion(monkeypatch):
    """Only the active winning request publishes the compatibility text list."""
    from agent import codex_runtime as runtime

    older_parts = ["older request"]
    agent = _guard_test_agent()
    agent._codex_streamed_text_parts = older_parts
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: []))

    def fake_consume(event_stream, **callbacks):
        callbacks["on_text_delta"]("winning text")
        return SimpleNamespace(status="completed", incomplete_details=None, error=None)

    monkeypatch.setattr(runtime, "_consume_codex_event_stream", fake_consume)
    result = runtime.run_codex_stream(
        agent,
        {"model": "gpt-5.6-sol"},
        client=client,
        request_guard=runtime.CodexRequestGuard(),
    )

    assert result is not None
    assert result.status == "completed"
    assert agent._codex_streamed_text_parts == ["winning text"]
    assert agent._codex_streamed_text_parts is not older_parts


def test_request_guard_deactivation_does_not_wait_for_blocked_callback(monkeypatch):
    """A stuck display/TTS callback cannot block watchdog cancellation or delay
    client teardown; deactivation only owns request lifecycle state."""
    from agent import codex_runtime as runtime

    callback_entered = threading.Event()
    release_callback = threading.Event()
    deactivated = threading.Event()
    guard = runtime.CodexRequestGuard()
    agent = _guard_test_agent()

    def fire_text(text):
        callback_entered.set()
        assert release_callback.wait(2)

    agent._fire_stream_delta = fire_text
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: []))

    def fake_consume(event_stream, **callbacks):
        callbacks["on_text_delta"]("current text")
        return SimpleNamespace(status="completed", incomplete_details=None, error=None)

    monkeypatch.setattr(runtime, "_consume_codex_event_stream", fake_consume)
    worker_errors = []

    def run_stream():
        try:
            runtime.run_codex_stream(
                agent,
                {"model": "gpt-5.6-sol"},
                client=client,
                request_guard=guard,
            )
        except Exception as exc:
            worker_errors.append(exc)

    runner = threading.Thread(target=run_stream)
    runner.start()
    assert callback_entered.wait(2)

    def deactivate():
        guard.deactivate()
        deactivated.set()

    killer = threading.Thread(target=deactivate)
    killer.start()
    assert deactivated.wait(0.1) is True
    release_callback.set()
    runner.join(2)
    killer.join(2)
    assert not runner.is_alive()
    assert not killer.is_alive()
    assert len(worker_errors) == 1
    assert isinstance(worker_errors[0], InterruptedError)


def test_request_guard_completion_and_cancellation_have_one_winner():
    from agent import codex_runtime as runtime

    completion_first = runtime.CodexRequestGuard()
    assert completion_first.complete() is True
    assert completion_first.deactivate() is False
    assert completion_first.is_completed() is True

    cancellation_first = runtime.CodexRequestGuard()
    assert cancellation_first.deactivate() is True
    assert cancellation_first.complete() is False
    assert cancellation_first.is_completed() is False


def test_request_guard_serializes_stranger_abort_before_owner_close():
    from agent import codex_runtime as runtime

    guard = runtime.CodexRequestGuard()
    client = object()
    assert guard.register_client(client, owner_tid=11) is True
    abort_started = threading.Event()
    release_abort = threading.Event()
    owner_received = threading.Event()
    owner_result = []

    def stranger_abort():
        candidate, stranger = guard.client_for_close(caller_tid=22)
        assert candidate is client and stranger is True
        abort_started.set()
        assert release_abort.wait(2)
        guard.finish_stranger_abort(candidate)

    def owner_close():
        owner_result.append(guard.client_for_close(caller_tid=11))
        owner_received.set()

    stranger = threading.Thread(target=stranger_abort)
    owner = threading.Thread(target=owner_close)
    stranger.start()
    assert abort_started.wait(2)
    owner.start()
    assert owner_received.wait(0.1) is False
    release_abort.set()
    stranger.join(2)
    owner.join(2)

    assert owner_received.is_set()
    assert owner_result == [(client, False)]
    assert not stranger.is_alive()
    assert not owner.is_alive()


def test_request_guard_atomically_resolves_first_event_vs_ttfb_timeout():
    """Exactly one side wins: an admitted first event prevents a TTFB claim,
    while a claimed timeout prevents every later event from reviving the request."""
    from agent import codex_runtime as runtime

    event_first = runtime.CodexRequestGuard()
    assert event_first.mark_event(123.0) is True
    assert event_first.deactivate_if_no_event() is False
    assert event_first.is_active() is True
    assert event_first.last_event_ts() == 123.0

    timeout_first = runtime.CodexRequestGuard()
    assert timeout_first.deactivate_if_no_event() is True
    assert timeout_first.mark_event(456.0) is False
    assert timeout_first.last_event_ts() is None


def test_guarded_stream_requires_request_local_client():
    """A delayed guarded worker may not create or mutate a shared client after
    timeout; the interruptible caller must bind its request-local client first."""
    from agent import codex_runtime as runtime

    with pytest.raises(ValueError, match="request-local client"):
        runtime.run_codex_stream(
            _guard_test_agent(),
            {"model": "gpt-5.6-sol"},
            request_guard=runtime.CodexRequestGuard(),
        )


def test_late_client_created_after_ttfb_is_closed_without_opening_stream(
    tmp_path, monkeypatch
):
    """If client creation resumes after the watchdog already returned, the
    inactive request cannot register that client or enter the streaming runtime."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")
    creation_started = threading.Event()
    release_creation = threading.Event()
    client_closed = threading.Event()
    stream_opened = threading.Event()
    late_client = SimpleNamespace()

    def create_client(**kwargs):
        creation_started.set()
        assert release_creation.wait(10)
        return late_client

    def close_client(client, reason=None):
        assert client is late_client
        client_closed.set()

    def fake_stream(*args, **kwargs):
        stream_opened.set()
        return SimpleNamespace(status="completed")

    monkeypatch.setattr(agent, "_create_request_openai_client", create_client)
    monkeypatch.setattr(agent, "_close_request_openai_client", close_client)
    monkeypatch.setattr(agent, "_abort_request_openai_client", close_client)
    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    try:
        with pytest.raises(TimeoutError):
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        assert creation_started.is_set()
        assert not stream_opened.is_set()
    finally:
        release_creation.set()

    assert client_closed.wait(2)
    assert not stream_opened.is_set()


def test_ttfb_kills_when_no_stream_event(tmp_path, monkeypatch):
    """Backend accepts the connection but emits no event -> killed at the TTFB
    cutoff, well before the 60s wall-clock stale timeout, with a retryable
    TimeoutError and a ``codex_ttfb_kill`` close reason."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}
    guards = []

    def fake_hang(api_kwargs, client=None, on_first_delta=None, request_guard=None):
        guards.append(request_guard)
        # Never set _codex_stream_last_event_ts: simulate zero events arriving.
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    t0 = time.time()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        elapsed = time.time() - t0
        assert "TTFB" in str(excinfo.value)
        assert "codex_ttfb_kill" in closes
        assert len(guards) == 1
        assert guards[0] is not None
        assert guards[0].is_active() is False
        # ~1s cutoff + 2s join grace; must be far under the 60s stale timeout.
        assert elapsed < 15, f"TTFB watchdog took {elapsed:.1f}s"
    finally:
        stop["flag"] = True


def test_ttfb_default_tolerates_slow_first_event(tmp_path, monkeypatch):
    """With no env override, a healthy request whose first stream event is
    merely slow (~2s of backend admission / prefill) remains well inside the
    45s cutoff and is not killed."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    # Default behavior: no explicit TTFB override.
    monkeypatch.delenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("HERMES_CODEX_TTFB_MAX_SECONDS", raising=False)

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    sentinel = SimpleNamespace(ok=True)

    def fake_slow_first_event(api_kwargs, client=None, on_first_delta=None, request_guard=None):
        # Backend is alive but slow to admit: first event lands after ~2s,
        # well under the 45s default cutoff. Mark the first byte so the
        # no-byte detector sees activity, then return the response.
        time.sleep(2.0)
        _mark_stream_event(agent, request_guard)
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_slow_first_event)

    resp = h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
    assert resp is sentinel
    assert "codex_ttfb_kill" not in closes


def test_ttfb_includes_silent_hang_hint_for_gpt_5_5(tmp_path, monkeypatch):
    """The no-first-byte watchdog should surface the same actionable hint as the
    stale-call timeout path when the model matches the silent-hang heuristic."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")

    closes: list = []
    statuses: list[str] = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(agent, "_buffer_status", lambda msg: statuses.append(msg))
    monkeypatch.setattr(agent, "_emit_status", lambda msg: statuses.append(msg))
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None, request_guard=None):
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        message = str(excinfo.value)
        assert "gpt-5.4" in message
        assert "gpt-5.3-codex" in message
        assert "gpt-5.4-codex" in message
        assert "codex_ttfb_kill" in closes
        assert statuses, "expected a user-facing watchdog status"
        assert any("gpt-5.4" in s and "gpt-5.3-codex" in s for s in statuses)
    finally:
        stop["flag"] = True


def test_ttfb_high_env_is_capped_for_openai_codex(tmp_path, monkeypatch):
    """A stale local env value like 90s must not make openai-codex wait 90s
    before reconnecting when the backend emits no SSE frames."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("HERMES_CODEX_TTFB_MAX_SECONDS", "1")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None, request_guard=None):
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    t0 = time.time()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.4", "input": "hi"})
        elapsed = time.time() - t0
        assert "TTFB threshold: 1s" in str(excinfo.value)
        assert "codex_ttfb_kill" in closes
        assert elapsed < 15, f"TTFB watchdog ignored cap and took {elapsed:.1f}s"
    finally:
        stop["flag"] = True


def test_ttfb_does_not_kill_when_events_flow(tmp_path, monkeypatch):
    """Once a stream event has arrived, a generation that runs past the TTFB
    cutoff is NOT killed by the watchdog — it completes normally."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    sentinel = SimpleNamespace(ok=True)

    def fake_stream(api_kwargs, client=None, on_first_delta=None, request_guard=None):
        # Bytes flowing: mark stream activity right away, then keep generating
        # past the 1s TTFB cutoff before returning a real response.
        _mark_stream_event(agent, request_guard)
        if on_first_delta:
            on_first_delta()
        time.sleep(2.0)
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    resp = h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
    assert resp is sentinel
    assert "codex_ttfb_kill" not in closes


def test_event_idle_kills_after_first_event_then_silence(tmp_path, monkeypatch):
    """If Codex emits an opening SSE event and then goes silent, kill it via
    the stream-idle watchdog instead of waiting for the long non-stream stale
    timeout."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "1")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent,
        "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}

    def fake_stream(api_kwargs, client=None, on_first_delta=None, request_guard=None):
        _mark_stream_event(agent, request_guard)
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        assert "after first byte" in str(excinfo.value)
        assert "codex_stream_idle_kill" in closes
        assert "codex_ttfb_kill" not in closes
    finally:
        stop["flag"] = True


def test_ttfb_disabled_via_env_zero(tmp_path, monkeypatch):
    """Setting HERMES_CODEX_TTFB_TIMEOUT_SECONDS=0 disables the TTFB watchdog;
    a no-event stall then falls through to the (here, 60s) stale timeout, so a
    short hang is NOT killed by TTFB."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    sentinel = SimpleNamespace(ok=True)

    def fake_stream(api_kwargs, client=None, on_first_delta=None, request_guard=None):
        # No event marker, but only briefly — well under the 60s stale timeout.
        time.sleep(2.0)
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    resp = h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
    assert resp is sentinel
    assert "codex_ttfb_kill" not in closes


def test_large_codex_request_reconnects_on_zero_event_stall_by_default(
    tmp_path, monkeypatch
):
    """Large requests still need bounded recovery when Codex emits zero SSE
    events. Context size may lengthen the cutoff, but must not disable it."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")
    monkeypatch.delenv("HERMES_CODEX_TTFB_DISABLE_ABOVE_TOKENS", raising=False)
    monkeypatch.delenv("HERMES_CODEX_TTFB_STRICT", raising=False)

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client", lambda c, reason=None: closes.append(reason)
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client", lambda c, reason=None: closes.append(reason)
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None, request_guard=None):
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    large_input = "x" * 44_000  # ~11k estimated tokens, above the former gate.
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": large_input})
        assert "TTFB threshold: 1s" in str(excinfo.value)
        assert "codex_ttfb_kill" in closes
    finally:
        stop["flag"] = True


def test_large_codex_request_strict_ttfb_env_still_reconnects(tmp_path, monkeypatch):
    """Operators can force the old early-reconnect behavior for large inputs
    with HERMES_CODEX_TTFB_STRICT=1."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("HERMES_CODEX_TTFB_STRICT", "1")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client", lambda c, reason=None: closes.append(reason)
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client", lambda c, reason=None: closes.append(reason)
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None, request_guard=None):
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    large_input = "x" * 44_000
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": large_input})
        assert "TTFB threshold: 1s" in str(excinfo.value)
        assert "codex_ttfb_kill" in closes
    finally:
        stop["flag"] = True
