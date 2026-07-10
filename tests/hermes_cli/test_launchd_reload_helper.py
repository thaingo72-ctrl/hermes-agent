"""Tests for the detached launchd reload helper."""

import json
from types import SimpleNamespace

from gateway import status as gateway_status
from hermes_cli import _launchd_reload_helper as helper


def _write_runtime(path, *, pid=4242, active_agents=0):
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "gateway_state": "running",
                "active_agents": active_agents,
            }
        ),
        encoding="utf-8",
    )


def test_wait_for_gateway_idle_waits_until_originating_turn_finishes(tmp_path, monkeypatch):
    runtime_path = tmp_path / "gateway_state.json"
    _write_runtime(runtime_path, active_agents=1)
    monkeypatch.setattr(helper, "_pid_exists", lambda pid: pid == 4242)
    sleeps = []

    def finish_turn(_seconds):
        sleeps.append(1)
        _write_runtime(runtime_path, active_agents=0)

    monkeypatch.setattr(helper.time, "sleep", finish_turn)

    assert helper.wait_for_gateway_idle(
        4242,
        runtime_path=runtime_path,
        timeout_s=1.0,
        poll_s=0.01,
        settle_s=0.0,
    ) is True
    assert sleeps == [1]


def test_wait_for_gateway_idle_requires_quiet_settle_window(tmp_path, monkeypatch):
    runtime_path = tmp_path / "gateway_state.json"
    _write_runtime(runtime_path, active_agents=0)
    monkeypatch.setattr(helper, "_pid_exists", lambda pid: pid == 4242)
    ticks = iter([0.0, 0.0, 1.0, 2.0])
    monkeypatch.setattr(helper.time, "monotonic", lambda: next(ticks))
    sleeps = []
    monkeypatch.setattr(helper.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert helper.wait_for_gateway_idle(
        4242,
        runtime_path=runtime_path,
        timeout_s=10.0,
        poll_s=0.1,
        settle_s=2.0,
    ) is True
    assert sleeps == [0.1, 0.1]


def test_wait_for_gateway_idle_refuses_reload_while_turn_remains_active(tmp_path, monkeypatch):
    runtime_path = tmp_path / "gateway_state.json"
    _write_runtime(runtime_path, active_agents=1)
    monkeypatch.setattr(helper, "_pid_exists", lambda pid: pid == 4242)

    assert helper.wait_for_gateway_idle(
        4242,
        runtime_path=runtime_path,
        timeout_s=0.0,
        poll_s=0.01,
    ) is False


def test_wait_for_gateway_idle_allows_reload_after_old_gateway_exits(tmp_path, monkeypatch):
    runtime_path = tmp_path / "gateway_state.json"
    _write_runtime(runtime_path, active_agents=1)
    monkeypatch.setattr(helper, "_pid_exists", lambda _pid: False)

    assert helper.wait_for_gateway_idle(
        4242,
        runtime_path=runtime_path,
        timeout_s=1.0,
        poll_s=0.01,
    ) is True


def test_replacement_gateway_health_requires_live_pid(tmp_path, monkeypatch):
    runtime_path = tmp_path / "gateway_state.json"
    _write_runtime(runtime_path, pid=5252, active_agents=0)
    monkeypatch.setattr(helper, "_pid_exists", lambda _pid: False)

    assert helper._replacement_gateway_healthy(runtime_path, old_pid=4242) is False


def _run_helper(
    tmp_path,
    monkeypatch,
    *,
    planned_marker=True,
    bootout_rc=0,
    pid_exits=True,
    reload_timeout=1.0,
    pid_waits=None,
):
    monkeypatch.setattr(helper, "wait_for_gateway_idle", lambda *a, **k: True)
    pid_checks = []

    def pid_exists(pid):
        if pid == 5252:
            return True
        pid_checks.append(1)
        return not pid_exits or len(pid_checks) == 1

    monkeypatch.setattr(helper, "_pid_exists", pid_exists)
    def wait_for_pid_exit(pid, **kwargs):
        if pid_waits is not None:
            pid_waits.append((pid, kwargs))
        return pid_exits

    monkeypatch.setattr(
        helper,
        "_wait_for_pid_exit",
        wait_for_pid_exit,
        raising=False,
    )
    monkeypatch.setattr(
        gateway_status, "write_planned_stop_marker", lambda _pid: planned_marker
    )
    monkeypatch.setattr(gateway_status, "clear_planned_stop_marker", lambda: None)
    calls = []
    runtime_path = tmp_path / "gateway_state.json"

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if "bootout" in cmd:
            return SimpleNamespace(returncode=bootout_rc)
        if "bootstrap" in cmd:
            _write_runtime(runtime_path, pid=5252, active_agents=0)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    monkeypatch.setattr(helper.time, "sleep", lambda _seconds: None)
    pending_path = tmp_path / ".launchd-reload-pending"
    pending_path.write_text("pending", encoding="utf-8")
    result = helper.run_deferred_reload(
        gateway_pid=4242,
        target="gui/501/ai.hermes.gateway",
        domain="gui/501",
        label="ai.hermes.gateway",
        plist_path=tmp_path / "ai.hermes.gateway.plist",
        pending_path=pending_path,
        runtime_path=runtime_path,
        log_path=tmp_path / "launchd-reload.log",
        idle_timeout_s=10.0,
        reload_timeout_s=reload_timeout,
    )
    return result, calls, pending_path


def test_deferred_reload_aborts_when_planned_stop_marker_write_fails(tmp_path, monkeypatch):
    result, calls, pending_path = _run_helper(
        tmp_path, monkeypatch, planned_marker=False
    )

    assert result == 1
    assert calls == []
    assert pending_path.exists()


def test_deferred_reload_aborts_when_bootout_fails(tmp_path, monkeypatch):
    result, calls, pending_path = _run_helper(tmp_path, monkeypatch, bootout_rc=5)

    assert result == 1
    assert [cmd for cmd, _kwargs in calls] == [
        ["launchctl", "bootout", "gui/501/ai.hermes.gateway"]
    ]
    assert pending_path.exists()


def test_deferred_reload_aborts_if_originating_gateway_does_not_exit(
    tmp_path, monkeypatch
):
    result, _calls, pending_path = _run_helper(
        tmp_path, monkeypatch, pid_exits=False
    )

    assert result == 1
    assert pending_path.exists()


def test_deferred_reload_clears_pending_marker_only_after_success(tmp_path, monkeypatch):
    result, _calls, pending_path = _run_helper(tmp_path, monkeypatch)

    assert result == 0
    assert not pending_path.exists()


def test_deferred_reload_retries_bootstrap_until_label_is_registered(
    tmp_path, monkeypatch
):
    registered = iter([False, True])
    monkeypatch.setattr(helper, "_launchd_registered", lambda _label: next(registered))

    result, calls, _pending_path = _run_helper(tmp_path, monkeypatch)

    bootstrap_calls = [cmd for cmd, _kwargs in calls if "bootstrap" in cmd]
    assert result == 0
    assert len(bootstrap_calls) == 2


def test_deferred_reload_pid_exit_wait_includes_reload_budget(tmp_path, monkeypatch):
    pid_waits = []

    result, calls, _pending_path = _run_helper(
        tmp_path,
        monkeypatch,
        reload_timeout=77.0,
        pid_waits=pid_waits,
    )

    assert result == 0
    bootout_kwargs = next(kwargs for cmd, kwargs in calls if "bootout" in cmd)
    assert bootout_kwargs["timeout"] == 107.0
    assert pid_waits == [(4242, {"timeout_s": 107.0})]
