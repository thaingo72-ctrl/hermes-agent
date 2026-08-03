#!/usr/bin/env python3
"""Per-file parallel test runner.

The minimum-viable replacement for pytest-xdist + a subprocess-isolation
plugin. Discovers test files under ``tests/`` (excluding integration/e2e
unless explicitly requested), then runs one ``python -m pytest <file>``
subprocess per file, with bounded parallelism (default: ``os.cpu_count()``).

Why per-file rather than per-test?
    Per-test spawn overhead (~250ms × 17k tests = 70min CPU minimum)
    swamped the actual work. Per-file spawn (~250ms × ~850 files = ~3.5min)
    fits in the budget while still giving every file a fresh Python
    interpreter — the only isolation boundary that actually matters
    (cross-file module-level state leakage was the original flake source;
    intra-file state is the test author's responsibility).

Why drop xdist entirely?
    xdist's persistent workers accumulate state across files, which is
    exactly the leakage we wanted to fix. xdist also adds complexity
    (loadfile vs loadscope, --max-worker-restart, internal control plane)
    that we don't need when the unit of work is "run pytest on one file".
    A subprocess.Popen pool gated by a semaphore is ~60 lines and does
    the job.

Usage:
    python scripts/run_tests_parallel.py [pytest_args...]

    Common pytest args pass through to each per-file pytest invocation
    (e.g. ``-q``, ``-v``, ``-x``, ``--tb=long``, ``-k 'pattern'``, ``--lf``)
    with no special separator — a bare ``-q`` "just works". Anything after
    a literal ``--`` is also passed through, and stacks with bare flags.

Environment:
    HERMES_TEST_WORKERS  Override worker count (default: os.cpu_count())
    HERMES_TEST_PATHS    Override discovery roots (colon-sep, default: 'tests')

Exit code: 0 if every file's pytest exited 0; 1 otherwise.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Default test discovery roots.
_DEFAULT_ROOTS = ["tests"]

# Directories to skip during discovery — these suites require real
# external services (a model gateway, a docker daemon with a prebuilt
# image, etc.) and are run in their own dedicated CI jobs:
#
#   tests/e2e/         — .github/workflows/tests.yml :: e2e job
#   tests/integration/ — historical; legacy --ignore flags
#   tests/docker/      — .github/workflows/docker.yml ::
#                        build-amd64 job (runs against the freshly-loaded
#                        nousresearch/hermes-agent:test image, via
#                        ``HERMES_TEST_IMAGE`` so the fixture skips
#                        rebuild). The full pytest-shard runner can't
#                        host these because the session-scoped
#                        ``built_image`` fixture would do a 3-7min
#                        ``docker build``,
#                        so the build is guaranteed to die in fixture
#                        setup. The dedicated job sidesteps both costs.
_SKIP_PARTS = {"integration", "e2e", "docker"}

# Per-file wall-clock cap. Override
# via --file-timeout or HERMES_TEST_FILE_TIMEOUT.
#
# Set to 300s (5 min) deliberately generous: the per-test subprocess
# isolation plugin spawns a fresh Python process per test, so a
# large-collection file pays N × (interpreter startup + import) of
# overhead before any test logic runs — and that overhead dilates under
# load on shared CI runners, producing false "no tests ran" timeouts on
# files that finish in ~100s on a quiet box. The Docker build matrix jobs
# take 7-10 min anyway, so this headroom costs nothing on total CI wall
# time while keeping a genuinely hung file bounded.
_DEFAULT_FILE_TIMEOUT_SECONDS = 300.0

# One-shot retry of failing test FILES. A file that exits non-zero is re-run
# once in a fresh subprocess; if the re-run passes, the file counts as passed
# but is loudly reported as FLAKY so it gets fixed rather than hidden.
# Deterministic failures fail both attempts — a real regression can never be
# laundered into green by this (it would have to flake in our favor twice in
# a row on the same runner, which is exactly the definition of a flake).
# Set to 0 to disable (env: HERMES_TEST_FILE_RETRIES).
_DEFAULT_FILE_RETRIES = 1

# Duration cache: maps relative file paths to last-observed subprocess
# wall-clock seconds. Used by ``--slice`` to distribute files across
# CI jobs by estimated total time, so no one job gets all the slow files.
_DURATIONS_FILE = "test_durations.json"


# Win32 Job Object definitions.  Keep these available on every platform so
# the containment contract can be unit-tested without a Windows host.
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _windows_last_error() -> int:
    get_last_error = getattr(ctypes, "get_last_error", lambda: 0)
    return int(get_last_error())


def _windows_api_error(action: str, error_code: int | None = None) -> OSError:
    if error_code is None:
        error_code = _windows_last_error()
    return OSError(error_code, f"Windows {action} failed")


class _WindowsJob:
    """A Win32 Job Object configured to hard-kill members on close."""

    def __init__(self, api: Any, handle: Any) -> None:
        self.api = api
        self.handle = handle

    @classmethod
    def create(cls, api: Any | None = None) -> "_WindowsJob":
        if api is None:
            loader = getattr(ctypes, "WinDLL", None)
            if loader is None:  # pragma: no cover - only reachable off Windows
                raise OSError("Win32 APIs are unavailable")
            api = loader("kernel32", use_last_error=True)
            assert api is not None
            api.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
            api.CreateJobObjectW.restype = ctypes.c_void_p
            api.SetInformationJobObject.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            api.SetInformationJobObject.restype = ctypes.c_int
            api.CloseHandle.argtypes = [ctypes.c_void_p]
            api.CloseHandle.restype = ctypes.c_int
            api.AssignProcessToJobObject.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            api.AssignProcessToJobObject.restype = ctypes.c_int
            api.GetCurrentProcess.argtypes = []
            api.GetCurrentProcess.restype = ctypes.c_void_p

        assert api is not None
        handle = api.CreateJobObjectW(None, None)
        if not handle:
            raise _windows_api_error("CreateJobObjectW")

        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        configured = api.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not configured:
            error_code = _windows_last_error()
            api.CloseHandle(handle)
            raise _windows_api_error("SetInformationJobObject", error_code)
        return cls(api, handle)

    def assign_current_process(self) -> None:
        """Contain this runner so all subsequently spawned children inherit."""
        process_handle = self.api.GetCurrentProcess()
        if not self.api.AssignProcessToJobObject(self.handle, process_handle):
            raise _windows_api_error("AssignProcessToJobObject(current process)")

    def close(self) -> None:
        if self.handle is None:
            return
        handle = self.handle
        self.handle = None
        if not self.api.CloseHandle(handle):
            raise _windows_api_error("CloseHandle(job)")


def _create_windows_runner_job(
    job_factory: Any = _WindowsJob.create,
) -> _WindowsJob | Any:
    """Put the runner in a KILL_ON_JOB_CLOSE Job before it spawns tests.

    The Job handle is intentionally kept open until process exit. Because the
    handle is non-inheritable, Windows closes the last handle even when the
    runner is hard-killed with TerminateProcess; every pytest descendant that
    inherited Job membership is then terminated by the kernel.
    """
    job = job_factory()
    try:
        job.assign_current_process()
    except BaseException:
        job.close()
        raise
    return job


def _approximately_count_tests(
    files: List[Path], repo_root: Path
) -> dict[Path, int]:
    """
    Make a decent estimate at individual tests per file.
    Running ``pytest --co -q`` is WAY too slow because it actually imports everything.

    Returns a mapping ``{file_path: test_count}``. Files with zero
    collected tests are omitted from the dict (not an error — e.g. the
    file only defines fixtures / conftest helpers).

    """

    results = {}

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            contents = f.read()
        results[path] = contents.count("def test_")

    return results


def _discover_files(roots: List[Path]) -> List[Path]:
    """Return every ``test_*.py`` under the given roots (sorted).

    Roots may be directories (recursed for ``test_*.py``) or explicit
    ``.py`` files (included as-is, even if they don't match the
    ``test_*`` prefix — caller knows what they want).

    Exclude any file whose path contains a component in ``_SKIP_PARTS``,
    UNLESS the user explicitly named it as a root (in which case the
    user's intent overrides the skip filter). This makes
    ``scripts/run_tests.sh tests/docker/`` work locally the same way
    ``pytest tests/docker/`` does — the CI-level skip exists to keep
    the sharded matrix from blowing up, not to block targeted runs.
    """
    seen: set[Path] = set()
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            # Explicit file: include it as-is, skip the _SKIP_PARTS filter
            # since the user named it directly.
            real = root.resolve()
            if real not in seen:
                seen.add(real)
                out.append(root)
            continue
        # If the explicit root itself sits inside a skipped dir (e.g.
        # the user said ``tests/docker``), the user has overridden the
        # skip for that subtree. Compute the set of skip-parts the user
        # opted into, and only filter files whose path crosses a
        # skip-part *outside* that opt-in.
        root_skip_overrides = {
            part for part in root.parts if part in _SKIP_PARTS
        }
        effective_skips = _SKIP_PARTS - root_skip_overrides
        for path in root.rglob("test_*.py"):
            if any(part in effective_skips for part in path.parts):
                continue
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            out.append(path)
    return sorted(out)


def _kill_tree(
    proc: Any,
    pgid: int | None = None,
) -> None:
    """Kill the pytest subprocess and every descendant it spawned.

    POSIX process-group SIGKILL targets the pgid captured immediately after
    ``Popen``; this still reaches descendants after the pytest leader exits.

    On Windows, ``taskkill /F /T`` handles per-file timeout and ordinary
    cleanup. Independently, the runner joins a Windows Job Object configured
    for hard termination with ``KILL_ON_JOB_CLOSE`` before spawning tests.
    With ``TerminateProcess``, Python SIGTERM handlers do not run. Windows
    instead closes the runner's non-inherited Job handle, and the kernel
    terminates every contained descendant.
    """
    if proc.pid is None:
        return

    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )  # windows-footgun: ok
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
    else:
        if pgid is not None:
            try:
                import signal as _signal

                os.killpg(pgid, _signal.SIGKILL)  # windows-footgun: ok
            except (ProcessLookupError, PermissionError, OSError):
                pass

    # Belt-and-suspenders: ensure subprocess.communicate() sees the exit.
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


class _ShutdownController:
    """Signal cleanup for active per-file pytest processes."""

    def __init__(
        self,
        shutdown_requested: threading.Event,
        active_processes: dict[int, tuple[Any, int | None]],
        active_processes_lock: threading.Lock,
    ) -> None:
        self.shutdown_requested = shutdown_requested
        self.active_processes = active_processes
        self.active_processes_lock = active_processes_lock
        self.received_signal: int | None = None
        # A Python signal handler can nest before its first assignment.  A
        # nonblocking Lock acquisition is the atomic one-winner operation;
        # the winner deliberately keeps it acquired for the runner lifetime.
        self._winner_guard: Any = threading.Lock()

    def handle_signal(self, signum: int, _frame: object) -> None:
        if not self._winner_guard.acquire(blocking=False):
            return
        self.received_signal = signum
        self.shutdown_requested.set()
        with self.active_processes_lock:
            processes = list(self.active_processes.values())
        for proc, pgid in processes:
            _kill_tree(proc, pgid=pgid)


def _run_one_file(
    file: Path,
    pytest_args: List[str],
    repo_root: Path,
    file_timeout: float,
    isolated_hermes_home: Path,
    shutdown_requested: threading.Event,
    active_processes: dict[int, tuple[Any, int | None]],
    active_processes_lock: threading.Lock,
    retries: int = 0,
) -> Tuple[Path, int, str, dict[str, int], float]:
    """Run ``python -m pytest <file> <pytest_args>`` in a fresh subprocess.

    Returns (file, returncode, captured_combined_output, summary_counts, subprocess_wall_seconds).

    ``retries`` > 0 enables the one-shot flake retry: a non-zero exit is
    re-run in a fresh subprocess; if the re-run passes, the file counts as
    passed but the output is prefixed with a FLAKY banner and the file/output
    are recorded in ``_FLAKY_RESULTS`` so the summary can call it out. A
    deterministic failure fails every attempt, so real regressions cannot
    be laundered green.

    ``summary_counts`` is the result of ``_parse_pytest_summary(output)`` —

    pytest exit codes (https://docs.pytest.org/en/stable/reference/exit-codes.html):
        0 = all tests passed
        1 = some tests failed
        2 = test execution interrupted
        3 = internal error
        4 = pytest CLI usage error
        5 = no tests collected

    We treat exit 5 as a pass: it just means every test in the file was
    skipped or filtered by a marker (e.g. ``-m 'not integration'`` skips
    files where every test is marked integration). That's intentional and
    not a failure mode.

    On per-file timeout (``file_timeout`` seconds) or any other exception
    during ``communicate()``, we kill the whole process group / process
    tree so grandchildren (uvicorn servers, async runtimes, etc.) do not
    orphan onto PID 1. This outer timeout exists only to
    bound a pathologically slow or hung file as a whole.
    """
    def _attempt_home(attempt_index: int) -> Path:
        home = isolated_hermes_home / f"attempt-{attempt_index}"
        home.mkdir(parents=True)
        return home

    file, rc, output, summary, subproc_wall = _run_one_file_once(
        file,
        pytest_args,
        repo_root,
        file_timeout,
        _attempt_home(0),
        shutdown_requested,
        active_processes,
        active_processes_lock,
    )
    attempt = 0
    while rc != 0 and attempt < retries and not shutdown_requested.is_set():
        attempt += 1
        first_output = output
        file, rc, output, summary, subproc_wall2 = _run_one_file_once(
            file,
            pytest_args,
            repo_root,
            file_timeout,
            _attempt_home(attempt),
            shutdown_requested,
            active_processes,
            active_processes_lock,
        )
        subproc_wall += subproc_wall2
        if rc == 0:
            output = (
                f"⚠ FLAKY: failed on attempt 1, passed on retry "
                f"(attempt {attempt + 1}). Fix the flake — do not ignore this.\n"
                f"--- first-attempt output ---\n{first_output}\n"
                f"--- retry output ---\n{output}"
            )
            with _flaky_lock:
                _FLAKY_RESULTS.append((file, output))
    return file, rc, output, summary, subproc_wall


# Files that failed once and passed on retry, with both attempts' output.
# Keeping the traceback is load-bearing: a self-healed flake without its
# failing assertion is only a filename, which forces another expensive full
# run to rediscover the race.
_FLAKY_RESULTS: List[Tuple[Path, str]] = []
_flaky_lock = threading.Lock()


def _run_one_file_once(
    file: Path,
    pytest_args: List[str],
    repo_root: Path,
    file_timeout: float,
    isolated_hermes_home: Path,
    shutdown_requested: threading.Event,
    active_processes: dict[int, tuple[Any, int | None]],
    active_processes_lock: threading.Lock,
) -> Tuple[Path, int, str, dict[str, int], float]:
    """Single attempt of a per-file pytest subprocess (see _run_one_file)."""
    if shutdown_requested.is_set():
        return file, 130, "runner interrupted before file started\n", {}, 0.0

    base_temp = isolated_hermes_home / "pytest-tmp"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--basetemp",
        str(base_temp),
        str(file),
        *pytest_args,
    ]
    child_env = os.environ.copy()
    child_env["HERMES_HOME"] = str(isolated_hermes_home)

    subproc_start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        start_new_session=sys.platform != "win32",
    )

    # ``start_new_session=True`` makes the POSIX child a session and process-
    # group leader, so its pgid is exactly its pid. Record that deterministic
    # value without a kernel lookup: a very fast pytest leader can exit before
    # ``os.getpgid`` runs while leaving descendants in the group.
    pgid: int | None = proc.pid if sys.platform != "win32" else None

    with active_processes_lock:
        active_processes[proc.pid] = (proc, pgid)
        stop_after_register = shutdown_requested.is_set()
    if stop_after_register:
        _kill_tree(proc, pgid=pgid)

    try:
        output, _ = proc.communicate(timeout=file_timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc, pgid=pgid)
        try:
            output, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            output = "(file timeout exceeded; output unavailable)"
        rc = 124  # de facto convention for "killed by timeout".
        output = (
            f"({file_timeout:.0f}s exceeded; "
            f"process tree SIGKILL'd)\n{output}"
        )
    except BaseException:
        # KeyboardInterrupt / runner crash — make sure no zombie
        # grandchildren outlive us.
        _kill_tree(proc, pgid=pgid)
        raise
    else:
        # Happy path: pytest exited on its own. Kill the group anyway in
        # case it left grandchildren behind; already-dead is a no-op.
        _kill_tree(proc, pgid=pgid)

        output += "\n"
    finally:
        with active_processes_lock:
            active_processes.pop(proc.pid, None)

    if rc == 5:
        # No tests collected in THIS file — legitimate per-file: a
        # platform-gated or fully-marker-filtered file (e.g. a win32-only
        # suite on Linux) collects nothing and must not fail the suite.
        # Tolerated here; the RUN-level guard in main() still fails when
        # NOTHING was collected across every file, so a broken invocation
        # (venv without pytest, -k that matches nothing) can't report green.
        rc = 0
    summary = _parse_pytest_summary(output)
    subproc_wall = time.monotonic() - subproc_start
    return file, rc, output, summary, subproc_wall


def _parse_pytest_summary(output: str) -> dict[str, int]:
    """Extract per-file test pass/fail/skip counts from pytest output.

    pytest prints a summary line like ``12 passed, 3 skipped, 1 failed in 2.1s``
    as the last non-empty line before the short test summary.  We scrape that
    line for the individual counts so the progress display can show test-level
    granularity instead of just file-level pass/fail.

    Returns a dict with keys ``passed``, ``failed``, ``skipped``, ``errors``,
    ``xfailed``, ``xpassed`` (only keys found in the output are present).
    """
    import re

    result: dict[str, int] = {}
    # Walk backwards from the end — the summary line is always near the tail.
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        # Match "N passed", "N failed", "N skipped", "N errors", "N xfailed", "N xpassed"
        for m in re.finditer(r"(\d+)\s+(passed|failed|skipped|errors|xfailed|xpassed)", line):
            result[m.group(2)] = int(m.group(1))
        # Also match "N error" (singular — pytest uses this sometimes).
        for m in re.finditer(r"(\d+)\s+error\b", line):
            result.setdefault("errors", result.get("errors", 0) + int(m.group(1)))
        if result:
            # Found the counts line — done.
            break
        # Stop at the short test summary header (if any) — everything above
        # that is individual failure details, not the counts line.
        if line.startswith("FAILED") or line.startswith("SHORT TEST SUMMARY"):
            break
    return result


def _format_file(file: Path, repo_root: Path) -> str:
    """Render a test-file path for display: strip the repo-root prefix
    when possible so output reads ``tests/acp/test_auth.py`` instead of
    ``/home/runner/work/hermes-agent/hermes-agent/tests/acp/test_auth.py``.

    Falls back to the absolute path for anything outside the repo root.
    """
    try:
        return str(file.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(file)


def _print_progress(
    tests_done: int,
    approx_total_tests: int,
    file: Path,
    rc: int,
    dur: float,
    repo_root: Path,
    tests_passed: int,
    tests_failed: int,
    test_counts: dict[Path, int],
    file_summary: dict[str, int] | None = None,
    subproc_wall: float | None = None,
) -> None:
    """Single-line live progress.

    When ``file_summary`` is provided (parsed from pytest output), the
    per-file parenthetical shows individual test pass/fail counts instead
    of just the total test count.

    ``subproc_wall`` is the actual subprocess wall-clock time (excluding
    queue-wait). When available, the display shows both the subprocess
    time and the queue-inclusive elapsed time.
    """
    status = "✓" if rc == 0 else "✗"
    pct = min((tests_done / approx_total_tests * 100), 100) if approx_total_tests else 0
    # Digit width for left-side counter padding (derived from total file count).
    fw = len(str(tests_passed + tests_failed))
    # Build per-file test count string.
    if file_summary:
        parts = []
        p = file_summary.get("passed", 0)
        f = file_summary.get("failed", 0)
        s = file_summary.get("skipped", 0)
        e = file_summary.get("errors", 0)
        if p:
            parts.append(f"{p}✓")
        if f:
            parts.append(f"{f}✗")
        if s:
            parts.append(f"{s}s")
        if e:
            parts.append(f"{e}e")
        # xfailed/xpassed are rare; include if present.
        xf = file_summary.get("xfailed", 0)
        xp = file_summary.get("xpassed", 0)
        if xf:
            parts.append(f"{xf}xf")
        if xp:
            parts.append(f"{xp}xp")
        test_str = " ".join(parts) + ", " if parts else ""
    else:
        n_tests = test_counts.get(file, 0)
        test_str = f"{n_tests} tests, " if n_tests else ""
    # Show subprocess time when available; fall back to queue-inclusive dur.
    if subproc_wall is not None:
        time_str = f"{subproc_wall:.1f}s"
    else:
        time_str = f"{dur:.1f}s"
    msg = (
        f"[{pct:5.1f}% | {tests_done:>5}/~{approx_total_tests}"
        f" | ✓{tests_passed:>{fw}} | ✗{tests_failed:>{fw}}] "
        f"{status} {_format_file(file, repo_root)} ({test_str}{time_str})"
    )
    # Truncate to terminal width if available (no clobbering ANSI lines).
    try:
        cols = os.get_terminal_size().columns
        if len(msg) > cols:
            msg = msg[: cols - 1] + "…"
    except OSError:
        pass
    print(msg, flush=True)


def _print_inline_failure(
    file: Path, output: str, repo_root: Path, pytest_passthrough: List[str]
) -> None:
    """Print a compact failure summary immediately when a file fails.

    Shows the tail of the pytest output (the failure section with stack
    traces) and a ready-to-run repro command, so the developer doesn't
    have to wait for the full run to finish before seeing what broke.
    """
    rel = _format_file(file, repo_root)
    # Build a repro command the developer can copy-paste.
    passthrough_str = " ".join(pytest_passthrough) if pytest_passthrough else ""
    repro = f"python -m pytest {rel}"
    if passthrough_str:
        repro += f" {passthrough_str}"

    # Grab just the failure lines (last ~30 lines of pytest output —
    # typically the FAILED summary + short test info).
    lines = output.rstrip().splitlines()
    tail = "\n".join(lines[-30:])

    print(flush=True)
    print(f"  ╔╍ Failed: {rel} ╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍", flush=True)
    for line in tail.splitlines():
        print(f"  ║ {line}", flush=True)
    print("  ║", flush=True)
    print(f"  ║  Repro: {repro}", flush=True)
    print("  ╚╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍", flush=True)
    print(flush=True)


def _load_durations(repo_root: Path) -> dict[str, float]:
    """Read the duration cache from the repo root.

    Returns a dict mapping relative file paths (e.g.
    ``tests/tools/test_code_execution.py``) to wall-clock seconds from
    the last run. Missing or corrupt file → empty dict (safe fallback).
    """
    path = repo_root / _DURATIONS_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print("[ERROR] Failed to load json durations file! {e}")
        return {}


def _save_durations(
    file_times: List[Tuple[Path, float]],
    repo_root: Path,
) -> None:
    """Write the duration cache so future ``--slice`` runs can use it.

    Merges with any existing cache so entries from files not in the
    current run (e.g. from a different slice) are preserved. Keys are
    repo-relative paths so the cache is portable across checkouts
    and CI runners.
    """
    data: dict[str, float] = _load_durations(repo_root)
    for f, t in file_times:
        key = _format_file(f, repo_root)
        data[key] = round(t, 3)
    path = repo_root / _DURATIONS_FILE
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compute_lpt_slices(
    files: List[Path],
    slice_count: int,
    durations: dict[str, float],
    repo_root: Path,
) -> List[List[Path]]:
    """Distribute files across N slices using LPT (Longest Processing Time first).

    Sorts files by estimated duration descending, then greedily assigns each
    file to the slice with the smallest accumulated time so far. This
    minimizes the makespan (max slice duration) and keeps CI jobs balanced.

    Files with no cached duration get a default estimate of 2.0s (roughly
    the P50 from profiling). This means first-time runs (no cache) still
    get reasonable distribution, and new files don't all land in one slice.

    Returns a list of N file-lists, one per slice (0-indexed).
    """
    if slice_count < 2:
        return [files]

    default_dur = 2.0
    file_durs: List[Tuple[Path, float]] = []
    for f in files:
        rel = _format_file(f, repo_root)
        dur = durations.get(rel, default_dur)
        file_durs.append((f, dur))

    # Sort longest first (LPT).
    file_durs.sort(key=lambda x: x[1], reverse=True)

    # Greedy assignment: for each file, add it to the slice with the
    # smallest current total.
    bucket_files: List[List[Path]] = [[] for _ in range(slice_count)]
    bucket_totals: List[float] = [0.0] * slice_count

    for f, dur in file_durs:
        min_idx = min(range(slice_count), key=lambda i: bucket_totals[i])
        bucket_files[min_idx].append(f)
        bucket_totals[min_idx] += dur

    return bucket_files


def _slice_files(
    files: List[Path],
    slice_index: int,
    slice_count: int,
    durations: dict[str, float],
    repo_root: Path,
) -> List[Path]:
    """Return the subset of *files* belonging to slice *slice_index*.

    Uses :func:`_compute_lpt_slices` for LPT distribution.

    ``slice_index`` is 1-indexed (1..slice_count) for ergonomics —
    ``--slice 1/4`` reads more naturally than ``--slice 0/4``.
    """
    if slice_count < 2:
        return files
    if not (1 <= slice_index <= slice_count):
        print(
            f"error: --slice index must be 1..{slice_count}, got {slice_index}",
            file=sys.stderr,
        )
        sys.exit(2)

    bucket_files = _compute_lpt_slices(files, slice_count, durations, repo_root)

    target = bucket_files[slice_index - 1]
    target_dur = sum(
        durations.get(_format_file(f, repo_root), 2.0) for f in target
    )
    total_dur = sum(
        durations.get(_format_file(f, repo_root), 2.0)
        for bucket in bucket_files
        for f in bucket
    )
    print(
        f"Slice {slice_index}/{slice_count}: {len(target)} files "
        f"(~{target_dur:.0f}s estimated of {total_dur:.0f}s total)",
        flush=True,
    )

    return target


def _make_stdio_glyph_safe() -> None:
    """Keep status glyphs from killing the runner on narrow console encodings.

    On native Windows, piped or legacy-console stdio defaults to a locale
    codec (usually cp1252) that cannot encode the ✓/✗ progress glyphs — the
    first per-file status line then dies with UnicodeEncodeError before a
    single test result is reported. Declare the runner's own output UTF-8
    (what CI and every modern terminal already are), with errors="replace"
    as the can't-crash backstop; where the encoding can't be changed, fall
    back to errors="replace" alone so glyphs degrade to "?" instead of
    killing the run. On already-UTF-8 stdio this is a no-op.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            try:
                reconfigure(errors="replace")
            except Exception:
                pass


def main() -> int:
    _make_stdio_glyph_safe()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=int(os.environ.get("HERMES_TEST_WORKERS") or (os.cpu_count() or 4) * 2),
        help="Parallel worker count (default: $HERMES_TEST_WORKERS or cpu_count*2)",
    )
    parser.add_argument(
        "--paths",
        default=os.environ.get("HERMES_TEST_PATHS", ":".join(_DEFAULT_ROOTS)),
        help="Colon-separated discovery roots (default: 'tests')",
    )
    parser.add_argument(
        "--include-integration",
        action="store_true",
        help="Don't skip integration/ e2e/ during discovery",
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=float(
            os.environ.get("HERMES_TEST_FILE_TIMEOUT", _DEFAULT_FILE_TIMEOUT_SECONDS)
        ),
        help=(
            "Per-file wall-clock cap in seconds. On timeout, the pytest "
            "subprocess and its full process tree are SIGKILL'd. "
            f"Default: {_DEFAULT_FILE_TIMEOUT_SECONDS}s ({round(_DEFAULT_FILE_TIMEOUT_SECONDS/60)} min), env: HERMES_TEST_FILE_TIMEOUT."
        ),
    )
    parser.add_argument(
        "--file-retries",
        type=int,
        default=int(
            os.environ.get("HERMES_TEST_FILE_RETRIES", _DEFAULT_FILE_RETRIES)
        ),
        help=(
            "Re-run a failing test FILE this many times in a fresh subprocess "
            "before declaring it failed. A pass-on-retry counts as passed but "
            "is reported as FLAKY in the summary. 0 disables. "
            f"Default: {_DEFAULT_FILE_RETRIES}, env: HERMES_TEST_FILE_RETRIES."
        ),
    )
    parser.add_argument(
        "--slice",
        metavar="I/N",
        help=(
            "Run only slice I of N (e.g. --slice 1/4). "
            "Files are distributed across slices using cached durations "
            "so each slice takes roughly equal wall time. "
            "Without a duration cache, files are distributed by count. "
            "Env: HERMES_TEST_SLICE (format: I/N)."
        ),
    )
    parser.add_argument(
        "--generate-slices",
        metavar="N",
        type=int,
        help=(
            "Discover test files, distribute them across N slices using "
            "LPT on cached durations, and print a JSON matrix to stdout "
            "then exit (no tests run). The JSON has the shape "
            "'{\"slices\": [{\"index\": 1, \"files\": [\"tests/foo.py\", ...]}, ...]}' "
            "so the CI generate job can feed it directly into a matrix."
        ),
    )
    parser.add_argument(
        "--files",
        metavar="LIST",
        help=(
            "Explicit colon-separated list of test files to run. Bypasses "
            "discovery entirely — used by CI matrix jobs that receive their "
            "file list from the generate job."
        ),
    )
    parser.add_argument(
        "paths_positional",
        nargs="*",
        metavar="PATH",
        help=(
            "Restrict discovery to these paths (directories or .py files). "
            "Mutually exclusive with --paths. Anything after a literal '--' "
            "separator is passed through to each per-file pytest invocation."
        ),
    )
    # Split argv into "our flags + positional paths" vs "pytest passthrough".
    #
    # Two ways to pass args through to the per-file pytest invocation:
    #   1. Explicit ``--`` separator: everything after it goes to pytest.
    #   2. Bare pytest flags anywhere before ``--``: any token starting with
    #      ``-`` that isn't one of OUR options is routed to pytest, so a bare
    #      ``-q`` / ``-v`` / ``-x`` / ``--tb=long`` / ``-k expr`` "just works"
    #      without the developer remembering the ``--``. This matches the
    #      docstring's promise and pytest muscle-memory.
    #
    # The subtlety bare-flag routing must handle: value-taking pytest flags
    # given in space-separated form (``-k expr``, ``-m mark``, ``-p plugin``,
    # ``-o name=val``). Naively, ``expr`` would look like a positional path and
    # clobber discovery. We peel the following token along with such flags so
    # it never reaches our positional ``paths``. ``=``-joined forms
    # (``-k=expr``, ``--tb=long``) are self-contained and need no lookahead.
    OUR_FLAGS = {
        "-j", "--jobs", "--paths", "--include-integration",
        "--file-timeout", "--file-retries", "--slice", "--generate-slices", "--files",
    }
    # pytest short flags that consume the NEXT token as their value.
    PYTEST_VALUE_FLAGS = {"-k", "-m", "-p", "-o", "-c", "-r", "-W"}

    def _is_our_flag(tok: str) -> bool:
        # Match exact (``-j``, ``--paths``), ``=``-joined (``--paths=x``),
        # and attached short-value (``-j4``) forms of our own options.
        if tok in OUR_FLAGS:
            return True
        head = tok.split("=", 1)[0]
        if head in OUR_FLAGS:
            return True
        # Attached short value, e.g. ``-j4`` → ``-j``.
        if len(tok) > 2 and tok[:2] in OUR_FLAGS and not tok[1] == "-":
            return True
        return False

    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        before, explicit_passthrough = argv[:sep], argv[sep + 1 :]
    else:
        before, explicit_passthrough = argv, []

    our_args: List[str] = []
    bare_passthrough: List[str] = []
    i = 0
    while i < len(before):
        tok = before[i]
        if tok.startswith("-") and not _is_our_flag(tok):
            bare_passthrough.append(tok)
            # Pull the value token for space-separated value flags.
            if tok in PYTEST_VALUE_FLAGS and i + 1 < len(before):
                bare_passthrough.append(before[i + 1])
                i += 2
                continue
        else:
            our_args.append(tok)
        i += 1

    args = parser.parse_args(our_args)

    # ── Node-id selectors → file + ``-k`` filter ────────────────────────────
    # This runner is FILE-granular: it spawns one ``pytest <file>`` per test
    # file. A pytest node id (``tests/foo.py::TestBar::test_baz``) is not an
    # existing path, so discovery silently dropped it and the run exited with
    # "No test files to run" — the selector looked accepted but nothing ran.
    # Translate instead: run the FILE and narrow with ``-k`` on the last
    # segment, which is what the caller meant.
    node_id_selectors: List[Tuple[str, str]] = []
    if args.paths_positional:
        translated: List[str] = []
        for raw in args.paths_positional:
            if "::" not in raw:
                translated.append(raw)
                continue
            file_part, _, selector = raw.partition("::")
            leaf = selector.rsplit("::", 1)[-1]
            # Strip a parametrized id (``test_x[case]``) down to the function
            # name; ``-k`` matches substrings, and brackets are -k syntax.
            leaf = leaf.split("[", 1)[0]
            node_id_selectors.append((raw, leaf))
            translated.append(file_part)
        if node_id_selectors:
            args.paths_positional = translated
            keys = [leaf for _, leaf in node_id_selectors]
            expr = " or ".join(dict.fromkeys(keys))
            for raw, leaf in node_id_selectors:
                print(
                    f"note: '{raw}' is a pytest node id; this runner is "
                    f"file-granular. Running the file with -k {leaf!r}.",
                    file=sys.stderr,
                )
            # Only inject -k when the caller didn't pass one themselves; their
            # explicit filter wins over our inferred one.
            if not any(
                t == "-k" or t.startswith("-k=") or (t.startswith("-k") and len(t) > 2)
                for t in bare_passthrough + explicit_passthrough
            ):
                bare_passthrough = bare_passthrough + ["-k", expr]

    # Bare flags run before any explicit ``--`` passthrough so ordering is
    # intuitive (``run_tests.sh tests/foo.py -q -- --tb=long`` → ``-q --tb=long``).
    pytest_passthrough = bare_passthrough + explicit_passthrough

    # Parse --slice (or HERMES_TEST_SLICE) early so we can exit on bad input
    # before doing any expensive discovery.
    slice_raw = args.slice or os.environ.get("HERMES_TEST_SLICE")
    slice_index: int | None = None
    slice_count: int = 1
    if slice_raw:
        try:
            idx_s, count_s = slice_raw.split("/", 1)
            slice_index = int(idx_s)
            slice_count = int(count_s)
        except (ValueError, AttributeError):
            print(f"error: --slice must be I/N (e.g. 1/4), got: {slice_raw!r}", file=sys.stderr)
            sys.exit(2)

    repo_root = Path(__file__).resolve().parent.parent

    # --files: explicit file list from the CI generate job — skip discovery.
    if args.files:
        files = [repo_root / f for f in args.files.split(":") if f.strip()]
        roots = []
    else:
        # Resolve discovery roots: positional path args override --paths if any
        # were supplied, otherwise --paths (which itself defaults to 'tests').
        if args.paths_positional:
            roots = [repo_root / p for p in args.paths_positional]
        else:
            roots = [repo_root / p for p in args.paths.split(":") if p]

        if args.include_integration:
            # Caller takes responsibility — typically used via explicit -k filter.
            global _SKIP_PARTS  # noqa: PLW0603 — config knob
            _SKIP_PARTS = set()

        files = _discover_files(roots)

    if not files:
        print("No test files to run", file=sys.stderr)
        return 1

    # --generate-slices: compute LPT distribution and emit JSON, then exit.
    if args.generate_slices is not None:
        durations = _load_durations(repo_root)
        slices = _compute_lpt_slices(
            files, args.generate_slices, durations, repo_root
        )
        matrix = {
            "slice": [
                {
                    "index": i + 1,
                    "files": ":".join(_format_file(f, repo_root) for f in bucket),
                }
                for i, bucket in enumerate(slices)
            ]
        }
        # Print to stdout so the CI step can capture it with $().
        print(json.dumps(matrix))
        return 0

    # Count individual tests per file
    test_counts = _approximately_count_tests(files, repo_root)
    approx_total_tests = sum(test_counts.values())

    # Apply slicing if requested — distribute files across CI jobs by
    # estimated duration so no one job gets all the slow files.
    if slice_index is not None:
        durations = _load_durations(repo_root)
        files = _slice_files(files, slice_index, slice_count, durations, repo_root)
        # Recount after slicing.
        test_counts = {f: test_counts[f] for f in files if f in test_counts}
        approx_total_tests = sum(test_counts.values())

    if roots:
        roots_str = [str(r.relative_to(repo_root)) if r.is_relative_to(repo_root) else str(r) for r in roots]
        print(
            f"Discovered {len(files)} test files (~{approx_total_tests} tests) under "
            f"{roots_str}; running with -j {args.jobs}",
            flush=True,
        )
    else:
        print(
            f"Running {len(files)} test files (~{approx_total_tests} tests) "
            f"with -j {args.jobs}",
            flush=True,
        )

    # Capture and print on completion (out-of-order is fine — keeps the
    # terminal clean rather than interleaving N parallel pytest outputs).
    failures: List[Tuple[Path, str, Dict[str, int]]] = []
    file_times: List[Tuple[Path, float]] = []  # (file, subprocess_wall) for distribution
    started = time.monotonic()
    files_done = 0
    tests_done = 0
    pass_count = 0
    fail_count = 0
    tests_passed = 0
    tests_failed = 0
    # Every collected outcome, not just pass/fail: a legitimately all-skipped
    # (platform-gated) file reports "2 skipped" and must NOT trip the
    # nothing-ran guard, whereas a file that died before collection reports
    # nothing at all and must.
    tests_collected = 0
    lock = threading.Lock()

    def _on_done(file: Path, started_at: float, fut: "Future[Tuple[Path, int, str, Dict[str, int], float]]") -> None:
        nonlocal files_done, tests_done, pass_count, fail_count, tests_passed, tests_failed
        nonlocal tests_collected
        n_tests = test_counts.get(file, 0)
        try:
            fpath, rc, output, summary, subproc_wall = fut.result()
        except Exception as exc:  # noqa: BLE001 — must always advance counter
            with lock:
                files_done += 1
                tests_done += n_tests
                fail_count += 1
                failures.append((file, f"runner crashed: {exc!r}", {}))
                _print_progress(
                    tests_done, approx_total_tests, file, 1,
                    time.monotonic() - started_at,
                    repo_root, tests_passed, tests_failed,
                    test_counts,
                    subproc_wall=0.0,
                )
            return
        with lock:
            files_done += 1
            tests_done += n_tests
            # Accumulate test-level counts from parsed summary.
            tests_passed += summary.get("passed", 0)
            tests_failed += summary.get("failed", 0)
            tests_collected += sum(
                summary.get(k, 0)
                for k in ("passed", "failed", "skipped", "errors", "xfailed", "xpassed")
            )
            file_times.append((fpath, subproc_wall))
            if rc == 0:
                pass_count += 1
            else:
                fail_count += 1
                failures.append((fpath, output, summary))
            _print_progress(
                tests_done, approx_total_tests, fpath, rc,
                time.monotonic() - started_at,
                repo_root, tests_passed, tests_failed,
                test_counts,
                file_summary=summary,
                subproc_wall=subproc_wall,
            )
            if rc != 0:
                _print_inline_failure(fpath, output, repo_root, pytest_passthrough)

    # On Windows, contain the runner itself before any pytest subprocess is
    # created. Descendants inherit Job membership from birth, and a hard
    # TerminateProcess of the runner closes the last non-inherited Job handle.
    _windows_runner_job: _WindowsJob | Any | None = None
    if sys.platform == "win32":
        try:
            _windows_runner_job = _create_windows_runner_job()
        except OSError as exc:
            print(f"ERROR: failed to establish Windows process containment: {exc}")
            return 1

    shutdown_requested = threading.Event()
    active_processes: dict[int, tuple[Any, int | None]] = {}
    active_processes_lock = threading.Lock()
    shutdown_controller = _ShutdownController(
        shutdown_requested,
        active_processes,
        active_processes_lock,
    )
    previous_handlers: dict[int, Any] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, shutdown_controller.handle_signal)
        except (OSError, ValueError):
            pass

    isolation_setup_error: OSError | None = None
    cleanup_error: OSError | None = None
    try:
        try:
            isolation_dir = tempfile.TemporaryDirectory(prefix="hermes-test-run-")
        except OSError as exc:
            isolation_setup_error = exc
        else:
            try:
                isolation_root = isolation_dir.name
                with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                    futures: List[Future] = []
                    for file_index, file in enumerate(files):
                        file_home = Path(isolation_root) / str(file_index)
                        t0 = time.monotonic()
                        fut = pool.submit(
                            _run_one_file,
                            file,
                            pytest_passthrough,
                            repo_root,
                            args.file_timeout,
                            file_home,
                            shutdown_requested,
                            active_processes,
                            active_processes_lock,
                            args.file_retries,
                        )
                        fut.add_done_callback(lambda f, file=file, t0=t0: _on_done(file, t0, f))
                        futures.append(fut)
                    # Block until everything's done. ThreadPoolExecutor.__exit__ waits
                    # for all submitted work, but doing it explicitly here makes the
                    # control flow obvious.
                    for fut in futures:
                        fut.result() if fut.exception() is None else None
            finally:
                try:
                    isolation_dir.cleanup()
                except OSError as exc:
                    cleanup_error = exc
    finally:
        # Restore first, then inspect received_signal. A signal delivered while
        # a not-yet-restored handler is still ours is therefore observed below;
        # after restoration, the caller's/default semantics apply instead.
        for sig, previous_handler in previous_handlers.items():
            signal.signal(sig, previous_handler)

    if isolation_setup_error is not None:
        print(f"ERROR: failed to create isolated test root: {isolation_setup_error}")
        return 1
    if cleanup_error is not None:
        print(f"ERROR: failed to remove isolated test root: {cleanup_error}")
        return 1
    if shutdown_controller.received_signal is not None:
        print(
            f"Interrupted by signal {shutdown_controller.received_signal}; "
            "active test trees cleaned up"
        )
        return 128 + shutdown_controller.received_signal

    elapsed = time.monotonic() - started
    print()
    pct = min(100, (tests_done / approx_total_tests * 100)) if approx_total_tests else 0
    print(f"=== Summary: {len(files)} files, {tests_passed} tests passed, {tests_failed} failed ({pct:.0f}% complete) in {elapsed:.1f}s ({args.jobs} workers) ===")

    # Zero tests collected across the WHOLE run is NOT a pass. Per-file rc=5
    # is deliberately tolerated above (platform-gated files), but if NOTHING
    # ran anywhere the invocation itself was broken — a venv without pytest, a
    # -k/-m filter that matched nothing, or collection erroring everywhere.
    # The summary line above reads green at a glance ("0 failed ... 100%
    # complete"), which has been misread as a successful verification, so say
    # it plainly AND fail the exit code.
    no_tests_ran_at_all = bool(files) and tests_collected == 0
    if no_tests_ran_at_all:
        print()
        print(
            "=== ✗ NO TESTS RAN — 0 collected across "
            f"{len(files)} file{'s' if len(files) != 1 else ''}. "
            "This is NOT a pass. ==="
        )
        print(
            "  Common causes: the selected venv has no pytest; a -k/-m filter "
            "matched nothing; or collection errored in every file."
        )
        print("  Check the per-file output above for the real error.")

    # Flaky files: failed once, passed on the automatic retry. Green, but
    # loudly reported so they get fixed instead of silently re-flaking.
    if _FLAKY_RESULTS:
        print()
        print(f"=== ⚠ {len(_FLAKY_RESULTS)} FLAKY file{'s' if len(_FLAKY_RESULTS) != 1 else ''} (failed once, passed on retry — fix these) ===")
        for f, output in _FLAKY_RESULTS:
            print(f"  {_format_file(f, repo_root)}")
            print(output.rstrip())

    # Save durations for future --slice runs. Each slice writes its own
    # partial test_durations.json; a CI merge step joins them later.
    # Locally, _save_durations merges with any existing cache so entries
    # from previous runs aren't lost.
    if file_times:
        _save_durations(file_times, repo_root)
        print(f"  Durations cached to {_DURATIONS_FILE} ({len(file_times)} files)")

    # Per-file time distribution (throwaway diagnostic — shows how
    # subprocess time is distributed so we can see if startup dominates).
    if file_times:
        times = sorted([t for _, t in file_times])
        total_subproc = sum(times)
        median_t = times[len(times) // 2]
        p50 = median_t
        p90 = times[int(len(times) * 0.90)]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[min(int(len(times) * 0.99), len(times) - 1)]
        max_t = times[-1]
        # How many files finish in <1s? That's roughly "just startup".
        fast = sum(1 for t in times if t < 1.0)
        fast_2s = sum(1 for t in times if t < 2.0)
        print()
        print("=== Per-file subprocess time distribution ===")
        print(f"  Files:   {len(times)}")
        print(f"  Total subprocess CPU-wall: {total_subproc:.1f}s  (runner wall: {elapsed:.1f}s, parallelism: {args.jobs}x)")
        print(f"  P50: {p50:.2f}s  P90: {p90:.2f}s  P95: {p95:.2f}s  P99: {p99:.2f}s  Max: {max_t:.2f}s")
        print(f"  <1s: {fast} files ({fast/len(times)*100:.0f}%)  <2s: {fast_2s} files ({fast_2s/len(times)*100:.0f}%)")
        # Top 10 slowest files — likely the ones dragging the run.
        slowest = sorted(file_times, key=lambda x: x[1], reverse=True)[:10]
        print("  Top 10 slowest:")
        for f, t in slowest:
            print(f"    {t:>6.2f}s  {_format_file(f, repo_root)}")

    if failures:
        print()
        print("=== Failure output ===")
        for file, output, _summary in failures:
            print()
            print(f"--- {_format_file(file, repo_root)} ---")
            print(output.rstrip())
        print()
        # Split: files with actual test failures vs non-zero exit for other reasons
        test_fail_files = [(f, s) for f, _o, s in failures if s.get("failed", 0) > 0]
        all_passed_but_nonzero = [(f, s) for f, _o, s in failures
                                  if s.get("failed", 0) == 0 and s.get("passed", 0) > 0]
        no_tests_ran = [(f, s) for f, _o, s in failures
                        if s.get("failed", 0) == 0 and s.get("passed", 0) == 0]
        if test_fail_files:
            total_tf = sum(s.get("failed", 0) for _, s in test_fail_files)
            print(f"=== {len(test_fail_files)} file{'s' if len(test_fail_files) != 1 else ''} with test failures ({total_tf} test{'s' if total_tf != 1 else ''} failed) ===")
            for file, s in test_fail_files:
                nf = s.get("failed", 0)
                print(f"  {_format_file(file, repo_root)}  ({nf} test{'s' if nf != 1 else ''} failed)")
        if all_passed_but_nonzero:
            print(f"=== {len(all_passed_but_nonzero)} file{'s' if len(all_passed_but_nonzero) != 1 else ''} where all tests passed but pytest exited non-zero (warnings-as-errors, hook failures, etc.) ===")
            for file, s in all_passed_but_nonzero:
                print(f"  {_format_file(file, repo_root)}  ({s.get('passed', 0)} passed)")
        if no_tests_ran:
            print(f"=== {len(no_tests_ran)} file{'s' if len(no_tests_ran) != 1 else ''} where no tests ran (collection/import error, timeout before collection, etc.) ===")
            for file, s in no_tests_ran:
                print(f"  {_format_file(file, repo_root)}")
        return 1

    if no_tests_ran_at_all:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
