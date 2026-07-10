"""Detached, activity-aware launchd service reload helper.

This module is intentionally stdlib-heavy and process-independent. It is
spawned from a Hermes CLI command running inside the gateway's own process
tree, then waits for that originating gateway turn to finish before replacing
the launchd job. A fixed sleep is unsafe: the CLI command itself may be one
tool call in a much longer agent turn.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _wait_for_pid_exit(pid: int, *, timeout_s: float, poll_s: float = 0.2) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while _pid_exists(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(0.01, poll_s))
    return True


def _read_runtime_status(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def wait_for_gateway_idle(
    gateway_pid: int,
    *,
    runtime_path: Path,
    timeout_s: float,
    poll_s: float = 0.5,
    settle_s: float = 5.0,
) -> bool:
    """Wait until *gateway_pid* has no active agent turns.

    ``active_agents`` reaches zero just before the platform adapter sends the
    final response, so require a quiet settle window before allowing bootout.
    Missing/stale runtime state fails closed while the originating PID is
    alive. If that PID exits independently, the caller may safely abandon the
    reload without waiting out the deadline.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    idle_since: float | None = None
    while True:
        if not _pid_exists(gateway_pid):
            return True

        runtime = _read_runtime_status(runtime_path)
        runtime_pid = runtime.get("pid")
        active_agents = 1
        if runtime_pid == gateway_pid:
            try:
                active_agents = int(runtime.get("active_agents", 0) or 0)
            except (TypeError, ValueError):
                active_agents = 1

        now = time.monotonic()
        if runtime_pid == gateway_pid and active_agents <= 0:
            if idle_since is None:
                idle_since = now
            if now - idle_since >= max(0.0, settle_s):
                return True
        else:
            idle_since = None

        if now >= deadline:
            return False
        time.sleep(max(0.01, poll_s))


def _append_log(path: Path, message: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except OSError:
        pass


def _launchd_registered(label: str) -> bool:
    try:
        result = subprocess.run(
            ["launchctl", "list", label],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _replacement_gateway_healthy(runtime_path: Path, old_pid: int) -> bool:
    runtime = _read_runtime_status(runtime_path)
    try:
        replacement_pid = int(runtime.get("pid", 0) or 0)
    except (TypeError, ValueError):
        return False
    return (
        replacement_pid > 0
        and replacement_pid != old_pid
        and runtime.get("gateway_state") == "running"
        and _pid_exists(replacement_pid)
    )


def run_deferred_reload(
    *,
    gateway_pid: int,
    target: str,
    domain: str,
    label: str,
    plist_path: Path,
    pending_path: Path,
    runtime_path: Path,
    log_path: Path,
    idle_timeout_s: float,
    reload_timeout_s: float,
) -> int:
    """Reload a launchd job only after the originating gateway becomes idle."""
    if not wait_for_gateway_idle(
        gateway_pid,
        runtime_path=runtime_path,
        timeout_s=idle_timeout_s,
    ):
        _append_log(
            log_path,
            f"SKIPPED deferred reload for {target}: gateway PID {gateway_pid} "
            f"still had active agent work after {idle_timeout_s:.0f}s",
        )
        return 2

    # If the original process exited for another reason, do not risk killing a
    # replacement gateway that may already be handling a new turn. A later
    # explicit service start/restart can apply the rewritten plist.
    if not _pid_exists(gateway_pid):
        _append_log(
            log_path,
            f"SKIPPED deferred reload for {target}: originating gateway PID "
            f"{gateway_pid} already exited",
        )
        return 0

    # launchctl bootout delivers SIGTERM. Mark it planned immediately before
    # bootout so the gateway does not classify this maintenance operation as a
    # crash/unexpected signal or preserve restart-intent state incorrectly.
    try:
        from gateway.status import (
            clear_planned_stop_marker,
            write_planned_stop_marker,
        )

        if not write_planned_stop_marker(gateway_pid):
            raise OSError("planned-stop marker write returned false")
    except Exception as exc:
        _append_log(
            log_path,
            f"FAILED deferred reload for {target}: could not write planned-stop "
            f"marker for PID {gateway_pid}: {exc}",
        )
        return 1

    try:
        bootout = subprocess.run(
            ["launchctl", "bootout", target],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=max(30.0, reload_timeout_s + 30.0),
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        clear_planned_stop_marker()
        _append_log(
            log_path,
            f"FAILED deferred reload for {target}: launchctl bootout failed: {exc}",
        )
        return 1
    if bootout.returncode != 0:
        clear_planned_stop_marker()
        _append_log(
            log_path,
            f"FAILED deferred reload for {target}: launchctl bootout rc={bootout.returncode}",
        )
        return 1
    if not _wait_for_pid_exit(
        gateway_pid,
        timeout_s=max(30.0, reload_timeout_s + 30.0),
    ):
        clear_planned_stop_marker()
        _append_log(
            log_path,
            f"FAILED deferred reload for {target}: originating gateway PID "
            f"{gateway_pid} remained alive after bootout",
        )
        return 1

    deadline = time.monotonic() + max(1.0, reload_timeout_s)
    bootstrapped = False
    while True:
        if not bootstrapped:
            try:
                bootstrap = subprocess.run(
                    ["launchctl", "bootstrap", domain, str(plist_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=30,
                )
                bootstrapped = bootstrap.returncode == 0
                if not bootstrapped:
                    _append_log(
                        log_path,
                        f"bootstrap failed for {target}: rc={bootstrap.returncode}; retrying",
                    )
            except (subprocess.TimeoutExpired, OSError) as exc:
                _append_log(
                    log_path,
                    f"bootstrap failed for {target}: {exc}; retrying",
                )

        registered = _launchd_registered(label)
        if registered:
            # A non-zero bootstrap can mean the label was already loaded. Once
            # launchctl confirms registration, stop reissuing bootstrap while
            # the replacement runtime finishes starting.
            bootstrapped = True
        else:
            bootstrapped = False
            _append_log(
                log_path,
                f"bootstrap did not register {label}; retrying",
            )

        if registered and _replacement_gateway_healthy(runtime_path, gateway_pid):
            try:
                pending_path.unlink(missing_ok=True)
            except OSError as exc:
                _append_log(
                    log_path,
                    f"FAILED deferred reload for {target}: could not clear pending "
                    f"marker {pending_path}: {exc}",
                )
                return 1
            _append_log(
                log_path,
                f"Completed activity-aware launchd reload for {target} after "
                f"gateway PID {gateway_pid} became idle and exited",
            )
            return 0
        if time.monotonic() >= deadline:
            _append_log(
                log_path,
                f"FAILED deferred reload for {target}: replacement gateway was not "
                f"healthy after {reload_timeout_s:.0f}s",
            )
            return 1
        time.sleep(2.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-pid", required=True, type=int)
    parser.add_argument("--target", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--plist-path", required=True, type=Path)
    parser.add_argument("--pending-path", required=True, type=Path)
    parser.add_argument("--runtime-path", required=True, type=Path)
    parser.add_argument("--log-path", required=True, type=Path)
    parser.add_argument("--idle-timeout", type=float, default=1800.0)
    parser.add_argument("--reload-timeout", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_deferred_reload(
        gateway_pid=args.gateway_pid,
        target=args.target,
        domain=args.domain,
        label=args.label,
        plist_path=args.plist_path,
        pending_path=args.pending_path,
        runtime_path=args.runtime_path,
        log_path=args.log_path,
        idle_timeout_s=args.idle_timeout,
        reload_timeout_s=args.reload_timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
