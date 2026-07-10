import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import gateway.run as gateway_run


class _ExitCalled(Exception):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


def _raise_exit(code: int) -> None:
    raise _ExitCalled(code)


def test_cli_run_gateway_force_exits_with_wedged_executor(tmp_path):
    """The installed service enters through hermes_cli.gateway.run_gateway,
    not gateway.run.main. A non-daemon executor worker must not strand that
    real entrypoint during a planned service restart."""
    project_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import concurrent.futures
        import threading

        import gateway.run as gateway_run
        import hermes_cli.gateway as gateway_cli

        gateway_cli._guard_official_docker_root_gateway = lambda: None
        gateway_cli._guard_named_profile_under_multiplexer = lambda force=False: None
        gateway_cli._guard_supervised_gateway_conflict = lambda force=False: None
        gateway_cli._guard_existing_gateway_process_conflict = lambda replace=False: None
        gateway_cli.supports_systemd_services = lambda: False

        async def fake_start_gateway(*args, **kwargs):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            executor.submit(threading.Event().wait)
            executor.shutdown(wait=False, cancel_futures=True)
            raise SystemExit(75)

        gateway_run.start_gateway = fake_start_gateway
        gateway_cli.run_gateway()
        """
    )
    env = os.environ.copy()
    env.update(
        {
            "HERMES_GATEWAY_EXIT_DIAG": "0",
            "HERMES_HOME": str(tmp_path / "hermes-home"),
            "PYTHONPATH": str(project_root),
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 75, result.stderr


def test_cli_run_gateway_force_exits_when_exception_diagnostics_fail(tmp_path):
    """Broken stderr must not prevent a bounded nonzero process exit."""
    project_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import concurrent.futures
        import sys
        import threading

        import gateway.run as gateway_run
        import hermes_cli.gateway as gateway_cli

        gateway_cli._guard_official_docker_root_gateway = lambda: None
        gateway_cli._guard_named_profile_under_multiplexer = lambda force=False: None
        gateway_cli._guard_supervised_gateway_conflict = lambda force=False: None
        gateway_cli._guard_existing_gateway_process_conflict = lambda replace=False: None
        gateway_cli.supports_systemd_services = lambda: False

        class BrokenStderr:
            def write(self, _data):
                raise OSError("stderr unavailable")

            def flush(self):
                raise OSError("stderr unavailable")

        async def fake_start_gateway(*args, **kwargs):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            executor.submit(threading.Event().wait)
            executor.shutdown(wait=False, cancel_futures=True)
            sys.stderr = BrokenStderr()
            raise RuntimeError("unexpected gateway failure")

        gateway_run.start_gateway = fake_start_gateway
        gateway_cli.run_gateway()
        """
    )
    env = os.environ.copy()
    env.update(
        {
            "HERMES_GATEWAY_EXIT_DIAG": "0",
            "HERMES_HOME": str(tmp_path / "hermes-home"),
            "PYTHONPATH": str(project_root),
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 1


def test_main_force_exits_zero_after_clean_shutdown(monkeypatch):
    async def fake_start_gateway(config=None):
        return True

    stdout = SimpleNamespace(flush=Mock())
    stderr = SimpleNamespace(flush=Mock())

    monkeypatch.setattr(gateway_run, "start_gateway", fake_start_gateway)
    monkeypatch.setattr(gateway_run.os, "_exit", _raise_exit)
    monkeypatch.setattr(gateway_run.sys, "argv", ["gateway.run"])
    monkeypatch.setattr(gateway_run.sys, "stdout", stdout)
    monkeypatch.setattr(gateway_run.sys, "stderr", stderr)

    with pytest.raises(_ExitCalled) as exc_info:
        gateway_run.main()

    assert exc_info.value.code == 0
    stdout.flush.assert_called_once_with()
    stderr.flush.assert_called_once_with()


def test_main_force_exits_one_after_failed_shutdown(monkeypatch):
    async def fake_start_gateway(config=None):
        return False

    stdout = SimpleNamespace(flush=Mock())
    stderr = SimpleNamespace(flush=Mock())

    monkeypatch.setattr(gateway_run, "start_gateway", fake_start_gateway)
    monkeypatch.setattr(gateway_run.os, "_exit", _raise_exit)
    monkeypatch.setattr(gateway_run.sys, "argv", ["gateway.run"])
    monkeypatch.setattr(gateway_run.sys, "stdout", stdout)
    monkeypatch.setattr(gateway_run.sys, "stderr", stderr)

    with pytest.raises(_ExitCalled) as exc_info:
        gateway_run.main()

    assert exc_info.value.code == 1
    stdout.flush.assert_called_once_with()
    stderr.flush.assert_called_once_with()


def test_main_terminates_via_os_exit_not_systemexit(monkeypatch):
    """The terminating call must be os._exit, NOT sys.exit — SystemExit is
    exactly what triggers the Py_FinalizeEx non-daemon-thread join hang this
    fixes (#53107). If main() ever regresses to sys.exit(), SystemExit would
    propagate instead of our os._exit sentinel and this test would fail.

    Test contributed by @AgenticSpark (PR #53122, duplicate of #53121)."""
    async def fake_start_gateway(config=None):
        return False

    stdout = SimpleNamespace(flush=Mock())
    stderr = SimpleNamespace(flush=Mock())

    monkeypatch.setattr(gateway_run, "start_gateway", fake_start_gateway)
    monkeypatch.setattr(gateway_run.os, "_exit", _raise_exit)
    monkeypatch.setattr(gateway_run.sys, "argv", ["gateway.run"])
    monkeypatch.setattr(gateway_run.sys, "stdout", stdout)
    monkeypatch.setattr(gateway_run.sys, "stderr", stderr)

    # Our os._exit sentinel must be what terminates main() — not SystemExit.
    with pytest.raises(_ExitCalled):
        gateway_run.main()


def test_main_routes_systemexit_through_os_exit(monkeypatch):
    """start_gateway raises SystemExit on the clean-fatal-config (#51228),
    planned-restart, and service-restart paths. main() must catch it and route
    the carried code through os._exit too, so those paths are equally wedge-proof
    (#53107) — a SystemExit propagating to interpreter finalization would join a
    stuck non-daemon worker and hang. Verifies the explicit code (e.g. 78) is
    preserved through the os._exit backstop."""
    async def fake_start_gateway(config=None):
        raise SystemExit(78)

    stdout = SimpleNamespace(flush=Mock())
    stderr = SimpleNamespace(flush=Mock())

    monkeypatch.setattr(gateway_run, "start_gateway", fake_start_gateway)
    monkeypatch.setattr(gateway_run.os, "_exit", _raise_exit)
    monkeypatch.setattr(gateway_run.sys, "argv", ["gateway.run"])
    monkeypatch.setattr(gateway_run.sys, "stdout", stdout)
    monkeypatch.setattr(gateway_run.sys, "stderr", stderr)

    with pytest.raises(_ExitCalled) as exc_info:
        gateway_run.main()

    # The SystemExit(78) must be converted to os._exit(78), not propagated.
    assert exc_info.value.code == 78
    stdout.flush.assert_called_once_with()
    stderr.flush.assert_called_once_with()


def test_main_systemexit_none_code_maps_to_zero(monkeypatch):
    """SystemExit() with no code (or None) is a clean exit → os._exit(0)."""
    async def fake_start_gateway(config=None):
        raise SystemExit()

    monkeypatch.setattr(gateway_run, "start_gateway", fake_start_gateway)
    monkeypatch.setattr(gateway_run.os, "_exit", _raise_exit)
    monkeypatch.setattr(gateway_run.sys, "argv", ["gateway.run"])
    monkeypatch.setattr(gateway_run.sys, "stdout", SimpleNamespace(flush=Mock()))
    monkeypatch.setattr(gateway_run.sys, "stderr", SimpleNamespace(flush=Mock()))

    with pytest.raises(_ExitCalled) as exc_info:
        gateway_run.main()

    assert exc_info.value.code == 0


def test_main_systemexit_str_code_maps_to_one(monkeypatch):
    """SystemExit with a str code (CPython prints it to stderr then exits 1).
    We can't print during os._exit, but the code must still map to 1 — matching
    CPython's handle_system_exit semantics for a non-int, non-None code."""
    async def fake_start_gateway(config=None):
        raise SystemExit("fatal: something went wrong")

    monkeypatch.setattr(gateway_run, "start_gateway", fake_start_gateway)
    monkeypatch.setattr(gateway_run.os, "_exit", _raise_exit)
    monkeypatch.setattr(gateway_run.sys, "argv", ["gateway.run"])
    monkeypatch.setattr(gateway_run.sys, "stdout", SimpleNamespace(flush=Mock()))
    monkeypatch.setattr(gateway_run.sys, "stderr", SimpleNamespace(flush=Mock()))

    with pytest.raises(_ExitCalled) as exc_info:
        gateway_run.main()

    assert exc_info.value.code == 1


def test_exit_backstop_releases_pid_file_and_runtime_lock(monkeypatch):
    """os._exit bypasses atexit, and the early SystemExit exit paths never run
    _stop_impl — so the force-exit backstop itself must release the PID file and
    runtime lock, or those early paths (#51228 fatal-config) would leak them.
    Both releases are idempotent, so this is safe on every exit path."""
    from gateway import status as gateway_status

    remove_pid = Mock()
    release_lock = Mock()
    monkeypatch.setattr(gateway_status, "remove_pid_file", remove_pid)
    monkeypatch.setattr(gateway_status, "release_gateway_runtime_lock", release_lock)
    monkeypatch.setattr(gateway_run.os, "_exit", _raise_exit)
    monkeypatch.setattr(gateway_run.sys, "stdout", SimpleNamespace(flush=Mock()))
    monkeypatch.setattr(gateway_run.sys, "stderr", SimpleNamespace(flush=Mock()))

    with pytest.raises(_ExitCalled) as exc_info:
        gateway_run._exit_after_graceful_shutdown(78)

    assert exc_info.value.code == 78
    remove_pid.assert_called_once_with()
    release_lock.assert_called_once_with()
