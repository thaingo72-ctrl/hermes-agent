"""Verify scripts/run_tests_parallel.py kills test-spawned grandchildren.

Setup
-----
A test in this file spawns a long-lived Python grandchild that writes
its PID + a nonce to a tempfile, then exits without cleaning up.
With the old ``subprocess.run`` runner, that grandchild would orphan
and outlive the test (and the whole runner). With the current Popen +
``start_new_session`` + ``_kill_tree`` runner, the grandchild gets
SIGKILL'd via process-group kill when its file's pytest exits.

The leaker test always passes — its only job is to spawn a grandchild
and walk away. The verifier runs the runner over the leaker file in a
subprocess, then waits for the grandchild PID to disappear from the
kernel's process table.

POSIX-only: Windows has its own grandchild lifecycle (no shared session,
``taskkill /F /T`` semantics). Marked accordingly.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest


# Both tests share the same handoff file: the leaker writes here, the
# verifier reads here. We park it in $TMPDIR with a unique-per-run name
# so concurrent invocations of the suite don't clobber each other.
_HANDOFF_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "hermes-isolation-probe"
_HANDOFF_DIR.mkdir(exist_ok=True)


def _handoff_path_for(nonce: str) -> Path:
    return _HANDOFF_DIR / f"grandchild-{nonce}.json"


def _pid_alive(pid: int) -> bool:
    """POSIX: send signal 0 to probe whether ``pid`` is still alive.

    ``os.kill(pid, 0)`` raises ``ProcessLookupError`` if the process is
    gone, ``PermissionError`` if it exists but we can't signal it
    (someone else's pid). We treat PermissionError as "alive" because
    the process exists and that's all we need to know.
    """
    if sys.platform == "win32":  # pragma: no cover — POSIX-only test
        # On Windows we'd use OpenProcess + GetExitCodeProcess; this
        # test is skipped on Windows so the path is unreachable.
        raise RuntimeError("_pid_alive POSIX-only")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only probe")
@pytest.mark.live_system_guard_bypass
def test_grandchild_leak_is_killed_by_runner(tmp_path: Path) -> None:
    """Run the parallel runner over a probe file and verify cleanup.

    1. Materialize a probe file that spawns a long-lived grandchild and
       writes its PID to disk before exiting.
    2. Invoke ``scripts/run_tests_parallel.py`` against the probe file.
    3. Wait for the grandchild PID to vanish (poll for ~5s).
    4. Assert the runner exited cleanly AND the grandchild is dead.
    """
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    assert runner.exists(), f"runner missing at {runner}"

    # Probe lives in a temp dir, NOT under tests/, so the regular suite
    # never picks it up — only our explicit invocation does.
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    probe = probe_dir / "test_probe_leaker.py"
    nonce = f"{os.getpid()}-{int(time.time() * 1000)}"
    handoff = _handoff_path_for(nonce)
    if handoff.exists():
        handoff.unlink()

    probe_src = textwrap.dedent(f"""
        import json, os, subprocess, sys, time
        from pathlib import Path

        HANDOFF = Path({str(handoff)!r})

        def test_spawns_grandchild_and_walks_away():
            # Long-lived grandchild: detached, ignores SIGTERM (we want
            # SIGKILL or process-group kill to be the only thing that
            # works, simulating a misbehaving server).
            child = subprocess.Popen(
                [
                    sys.executable, "-c",
                    "import os, signal, sys, time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "sys.stdout.write(f'gc-pgid={{os.getpgid(0)}} gc-pid={{os.getpid()}}\\\\n'); "
                    "sys.stdout.flush(); "
                    "time.sleep(600)",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # IMPORTANT: do NOT pass start_new_session here. We want
                # the grandchild to inherit the pytest subprocess's
                # process group, so when the runner kills the group the
                # grandchild dies too.
            )
            # Read the first line so we can record gc's pgid in the
            # handoff, then walk away — don't close the pipe (would
            # signal EOF and let the child see SIGPIPE on next write).
            first_line = child.stdout.readline().decode().strip()
            HANDOFF.write_text(json.dumps({{
                "pid": child.pid,
                "diag": first_line,
                "test_pid": os.getpid(),
                "test_pgid": os.getpgid(0),
            }}))
            assert child.pid > 0
    """).strip()
    probe.write_text(probe_src + "\n")

    # Run the parallel runner against just the probe file. The runner
    # discovers under ``tests/`` by default, so we override via --paths.
    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--paths",
            str(probe_dir),
            "-j",
            "1",
            # Tight per-file timeout: the probe finishes in <1s, no
            # need for 10min.
            "--file-timeout",
            "30",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # The runner declares its stdio UTF-8 (see _make_stdio_glyph_safe);
        # decode the same way so ✓-glyph assertions hold on Windows, where
        # text=True alone would decode with the locale codec (cp1252).
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    assert handoff.exists(), (
        f"probe never wrote handoff file; runner output:\n{proc.stdout}"
    )
    handoff_data = json.loads(handoff.read_text())
    grandchild_pid = handoff_data["pid"]
    diag = handoff_data.get("diag", "(no diag)")
    test_pid = handoff_data.get("test_pid")
    test_pgid = handoff_data.get("test_pgid")
    handoff.unlink()

    # The runner must have exited cleanly (probe test passes).
    assert proc.returncode == 0, (
        f"runner exited {proc.returncode}; output:\n{proc.stdout}"
    )

    # The grandchild must be gone. Poll for a bit because process-group
    # SIGKILL + reaping isn't synchronous; on a loaded box it can take
    # a beat.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _pid_alive(grandchild_pid):
            break
        time.sleep(0.05)
    else:
        # Test cleanup: kill the leaked grandchild ourselves so a
        # FAILED assertion doesn't leave a sleep(600) running.
        try:
            os.kill(grandchild_pid, 9)
        except ProcessLookupError:
            pass
        pytest.fail(
            f"grandchild PID {grandchild_pid} survived runner exit; "
            f"diag={diag!r} test_pid={test_pid} test_pgid={test_pgid}; "
            f"runner output:\n{proc.stdout}"
        )


# ── Bare pytest-flag passthrough ─────────────────────────────────────────────
#
# The runner routes any token starting with ``-`` that isn't one of its own
# options (``-j``/``--jobs``, ``--paths``, ``--slice``, ``--file-timeout``,
# ``--generate-slices``, ``--files``, ``--include-integration``) straight
# through to each per-file pytest invocation — no ``--`` separator required.
# Before this, a bare ``-q`` errored out with "unrecognized arguments",
# forcing a retry on every run. These tests are behavior contracts, not
# snapshots: they assert that bare flags reach pytest and that value-taking
# flags (``-k expr``) keep their value instead of having it stolen by the
# positional-path discovery.


def _make_probe_dir(tmp_path: Path) -> Path:
    """Two trivial passing tests, one named test_alpha, one test_beta."""
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "test_flagprobe.py").write_text(
        "def test_alpha():\n    assert True\n\n"
        "def test_beta():\n    assert True\n"
    )
    return probe_dir


def _run_runner(probe_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    return subprocess.run(
        [sys.executable, str(runner), "--paths", str(probe_dir),
         "-j", "1", "--file-timeout", "30", *extra],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # The runner declares its stdio UTF-8 (see _make_stdio_glyph_safe);
        # decode the same way so ✓-glyph assertions hold on Windows, where
        # text=True alone would decode with the locale codec (cp1252).
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )




def test_bare_value_flag_keeps_its_value(tmp_path: Path) -> None:
    """``-k test_alpha`` reaches pytest as a selector, not as a path.

    The value token (``test_alpha``) must NOT be swallowed by the runner's
    positional-path discovery — if it were, discovery would look for a path
    named ``test_alpha``, find nothing, and the run would degrade. We assert
    the run succeeds AND only one of the two tests was selected (proving the
    ``-k`` filter actually applied inside pytest).
    """
    probe_dir = _make_probe_dir(tmp_path)
    proc = _run_runner(probe_dir, "-k", "test_alpha")
    assert proc.returncode == 0, proc.stdout
    # Exactly one test selected: the per-file summary shows "1✓" (1 passed).
    # test_beta is deselected by the -k filter.
    assert "1✓" in proc.stdout or "1 passed" in proc.stdout, proc.stdout
    assert "2✓" not in proc.stdout, (
        f"both tests ran — -k filter did not apply:\n{proc.stdout}"
    )




def test_positional_path_not_treated_as_flag(tmp_path: Path) -> None:
    """A positional path arg still overrides discovery (not routed to pytest)."""
    probe_dir = _make_probe_dir(tmp_path)
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    # Pass the probe dir positionally (no --paths), plus a bare -q.
    proc = subprocess.run(
        [sys.executable, str(runner), str(probe_dir), "-j", "1",
         "--file-timeout", "30", "-q"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout
    # Discovery found the probe file (2 tests), proving the positional path
    # was consumed as a root, not forwarded to pytest as a bad flag.
    assert "test_flagprobe.py" in proc.stdout, proc.stdout


def test_runner_overrides_caller_hermes_home_before_collection(tmp_path: Path) -> None:
    """Import-time cron writes must not use a caller-supplied Hermes home."""
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    caller_home = tmp_path / "caller-home"
    caller_jobs = caller_home / "cron" / "jobs.json"
    resolved_home_marker = tmp_path / "resolved-home.txt"
    probe = tmp_path / "test_cron_import_write.py"
    probe.write_text(
        textwrap.dedent(
            f"""
            import os
            from pathlib import Path

            from cron.jobs import JOBS_FILE, create_job

            CALLER_JOBS = Path({str(caller_jobs)!r})
            Path({str(resolved_home_marker)!r}).write_text(os.environ["HERMES_HOME"])
            create_job(
                prompt="runner isolation probe",
                schedule="every 1h",
                name="runner isolation probe",
            )


            def test_import_time_cron_write_uses_runner_home():
                assert JOBS_FILE.resolve() != CALLER_JOBS.resolve()
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["HERMES_HOME"] = str(caller_home)
    env["HOME"] = str(tmp_path / "fake-user-home")
    env["LOCALAPPDATA"] = str(tmp_path / "fake-local-app-data")
    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--files",
            str(probe),
            "--file-retries",
            "0",
            "-j",
            "1",
            "-q",
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout
    assert not caller_jobs.exists(), f"collection-time cron write escaped to {caller_jobs}"
    resolved_home = Path(resolved_home_marker.read_text(encoding="utf-8"))
    assert resolved_home.resolve() != caller_home.resolve()
    assert not resolved_home.exists(), f"isolated HERMES_HOME was not removed: {resolved_home}"


def test_file_retry_uses_fresh_hermes_home_for_each_attempt(tmp_path: Path) -> None:
    """A retry must not pass because the failed attempt poisoned runner state."""
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    probe = tmp_path / "test_home_poison_probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            ATTEMPT_HOME = Path(os.environ["HERMES_HOME"])
            POISON = ATTEMPT_HOME / "failed-attempt-marker"

            def test_failed_attempt_state_is_not_reused():
                if POISON.exists():
                    return
                POISON.write_text("poisoned by failed attempt")
                assert False, "failed attempt poisoned HERMES_HOME"
            """
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--files",
            str(probe),
            "--file-retries",
            "1",
            "-j",
            "1",
            "-q",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 1, proc.stdout
    assert "failed attempt poisoned HERMES_HOME" in proc.stdout
    assert "FLAKY file" not in proc.stdout


def test_parallel_files_get_distinct_collection_homes(tmp_path: Path) -> None:
    """Parallel collection-time writes must not share runner storage."""
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    probe_dir = tmp_path / "parallel-probes"
    probe_dir.mkdir()
    markers = [tmp_path / "home-a.txt", tmp_path / "home-b.txt"]
    for index, marker in enumerate(markers):
        (probe_dir / f"test_probe_{index}.py").write_text(
            textwrap.dedent(
                f"""
                import os
                from pathlib import Path
                from cron.jobs import create_job

                Path({str(marker)!r}).write_text(os.environ["HERMES_HOME"])
                create_job(
                    prompt="parallel runner isolation probe {index}",
                    schedule="every 1h",
                    name="parallel runner isolation probe {index}",
                )

                def test_probe():
                    assert os.environ["HERMES_HOME"]
                """
            ),
            encoding="utf-8",
        )

    caller_home = tmp_path / "caller-home"
    env = os.environ.copy()
    env["HERMES_HOME"] = str(caller_home)
    proc = subprocess.run(
        [sys.executable, str(runner), "--paths", str(probe_dir), "-j", "2", "-q"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout
    homes = [Path(marker.read_text(encoding="utf-8")) for marker in markers]
    assert homes[0].resolve() != homes[1].resolve()
    assert all(not home.exists() for home in homes)
    assert not (caller_home / "cron" / "jobs.json").exists()


def test_isolation_cleanup_failure_is_reported_and_fails_runner(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A leaked runner root must turn an otherwise-green run red."""
    from scripts import run_tests_parallel as runner_mod

    probe = tmp_path / "test_cleanup_probe.py"
    probe.write_text("def test_probe():\n    assert True\n", encoding="utf-8")
    isolation_root = tmp_path / "injected-isolation-root"

    class InjectedCleanupFailure:
        def __init__(self, *args, **kwargs):
            isolation_root.mkdir()
            self.name = str(isolation_root)

        def cleanup(self):
            raise OSError("injected isolation cleanup failure")

    def fake_run_one_file(file, *args, **kwargs):
        return file, 0, "1 passed in 0.01s\n", {"passed": 1}, 0.01

    monkeypatch.setattr(
        runner_mod.tempfile, "TemporaryDirectory", InjectedCleanupFailure
    )
    monkeypatch.setattr(runner_mod, "_run_one_file", fake_run_one_file)
    monkeypatch.setattr(runner_mod, "_save_durations", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path(runner_mod.__file__)),
            "--files",
            str(probe),
            "--file-retries",
            "0",
            "-j",
            "1",
            "-q",
        ],
    )

    result = runner_mod.main()
    output = capsys.readouterr().out

    assert result == 1
    assert "injected isolation cleanup failure" in output
    assert "cleaned up" not in output


def test_signal_arriving_during_handler_restore_is_not_missed(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A signal after test cleanup but before full restore still interrupts."""
    from scripts import run_tests_parallel as runner_mod

    probe = tmp_path / "test_late_signal_probe.py"
    probe.write_text("def test_probe():\n    assert True\n", encoding="utf-8")
    installed_handlers = {}
    signal_calls = []

    def fake_signal(sig, handler):
        signal_calls.append((sig, handler))
        if len(signal_calls) <= 2:
            installed_handlers[sig] = handler
        elif len(signal_calls) == 3:
            installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
        return handler

    def fake_run_one_file(file, *args, **kwargs):
        return file, 0, "1 passed in 0.01s\n", {"passed": 1}, 0.01

    monkeypatch.setattr(runner_mod.signal, "signal", fake_signal)
    monkeypatch.setattr(runner_mod, "_run_one_file", fake_run_one_file)
    monkeypatch.setattr(runner_mod, "_save_durations", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path(runner_mod.__file__)),
            "--files",
            str(probe),
            "--file-retries",
            "0",
            "-j",
            "1",
            "-q",
        ],
    )

    result = runner_mod.main()
    output = capsys.readouterr().out

    assert result == 128 + signal.SIGTERM
    assert "Interrupted by signal" in output
    assert len(signal_calls) == 4


def test_timeout_cleans_collection_home(tmp_path: Path) -> None:
    """The timeout path must kill pytest and remove its isolated cron store."""
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    home_marker = tmp_path / "timeout-home.txt"
    probe = tmp_path / "test_timeout.py"
    probe.write_text(
        textwrap.dedent(
            f"""
            import os
            from pathlib import Path
            import time
            from cron.jobs import create_job

            Path({str(home_marker)!r}).write_text(os.environ["HERMES_HOME"])
            create_job(
                prompt="timeout runner isolation probe",
                schedule="every 1h",
                name="timeout runner isolation probe",
            )

            def test_timeout():
                time.sleep(300)
            """
        ),
        encoding="utf-8",
    )

    caller_home = tmp_path / "caller-home"
    env = os.environ.copy()
    env["HERMES_HOME"] = str(caller_home)
    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--files",
            str(probe),
            "-j",
            "1",
            "--file-timeout",
            "1",
            "--file-retries",
            "0",
            "-q",
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 1, proc.stdout
    assert "1s exceeded" in proc.stdout
    isolated_home = Path(home_marker.read_text(encoding="utf-8"))
    assert not isolated_home.exists()
    assert not (caller_home / "cron" / "jobs.json").exists()


def test_posix_pgid_is_known_without_racy_kernel_lookup(
    monkeypatch, tmp_path: Path
) -> None:
    """A fast-exiting leader cannot make descendant cleanup lose its pgid."""
    from scripts import run_tests_parallel as runner_mod

    class FakeProc:
        pid = 4242
        returncode = 0

        def communicate(self, timeout=None):
            return "1 passed in 0.01s", None

        def kill(self):
            pass

    killed_pgids = []
    monkeypatch.setattr(runner_mod.sys, "platform", "linux")
    monkeypatch.setattr(runner_mod.subprocess, "Popen", lambda *args, **kwargs: FakeProc())
    monkeypatch.setattr(
        runner_mod.os,
        "getpgid",
        lambda _pid: (_ for _ in ()).throw(ProcessLookupError()),
    )
    monkeypatch.setattr(
        runner_mod,
        "_kill_tree",
        lambda _proc, pgid=None: killed_pgids.append(pgid),
    )

    result = runner_mod._run_one_file_once(
        tmp_path / "test_fast_exit.py",
        ["-q"],
        tmp_path,
        30,
        tmp_path / "isolated-home",
        threading.Event(),
        {},
        threading.Lock(),
    )

    assert result[1] == 0
    assert killed_pgids == [4242]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal/process-group probe")
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize(
    "sent_signal",
    [signal.SIGINT, signal.SIGTERM],
    ids=["sigint", "sigterm"],
)
def test_signal_cleans_active_process_tree_and_isolated_home(
    tmp_path: Path, sent_signal: int
) -> None:
    """SIGINT and SIGTERM must not orphan pytest or its temporary home."""
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    home_marker = tmp_path / "isolated-home.txt"
    pytest_pid_marker = tmp_path / "pytest-pid.txt"
    grandchild_pid_marker = tmp_path / "grandchild-pid.txt"
    probe = tmp_path / "test_hang.py"
    probe.write_text(
        textwrap.dedent(
            f"""
            import os
            from pathlib import Path
            import subprocess
            import sys
            import time

            Path({str(home_marker)!r}).write_text(os.environ["HERMES_HOME"])
            Path({str(pytest_pid_marker)!r}).write_text(str(os.getpid()))

            def test_hang_with_grandchild():
                child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
                Path({str(grandchild_pid_marker)!r}).write_text(str(child.pid))
                time.sleep(300)
            """
        ),
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        [
            sys.executable,
            str(runner),
            "--files",
            str(probe),
            "-j",
            "1",
            "--file-timeout",
            "300",
            "-q",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    pytest_pid = None
    grandchild_pid = None
    isolated_home = None
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if home_marker.exists() and pytest_pid_marker.exists() and grandchild_pid_marker.exists():
                break
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout is not None else ""
                pytest.fail(f"runner exited before signal probe was ready: {output}")
            time.sleep(0.05)
        else:
            pytest.fail("timed out waiting for signal probe handoff")

        isolated_home = Path(home_marker.read_text(encoding="utf-8"))
        pytest_pid = int(pytest_pid_marker.read_text(encoding="utf-8"))
        grandchild_pid = int(grandchild_pid_marker.read_text(encoding="utf-8"))
        proc.send_signal(sent_signal)
        proc.communicate(timeout=15)
        runner_returncode = proc.returncode
        time.sleep(0.2)

        pytest_alive = _pid_alive(pytest_pid)
        grandchild_alive = _pid_alive(grandchild_pid)
        isolated_home_exists = isolated_home.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=10)
        if pytest_pid is not None:
            try:
                os.killpg(pytest_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if isolated_home is not None:
            shutil.rmtree(isolated_home, ignore_errors=True)

    assert runner_returncode == 128 + sent_signal
    assert not pytest_alive, "pytest child survived runner signal"
    assert not grandchild_alive, "pytest grandchild survived runner signal"
    assert not isolated_home_exists, "per-file HERMES_HOME survived runner signal"


def test_windows_job_enables_kill_on_close_and_closes_once() -> None:
    """The containment handle must own and hard-kill every assigned process."""
    from scripts import run_tests_parallel as runner_mod

    job_type = getattr(runner_mod, "_WindowsJob", None)
    assert job_type is not None

    class FakeKernel32:
        def __init__(self):
            self.created = 0
            self.info_class = None
            self.limit_flags = None
            self.assigned = []
            self.closed = []

        def CreateJobObjectW(self, security_attributes, name):
            self.created += 1
            return 101

        def SetInformationJobObject(self, handle, info_class, info, size):
            self.info_class = info_class
            self.limit_flags = info._obj.BasicLimitInformation.LimitFlags
            return True

        def GetCurrentProcess(self):
            return 202

        def AssignProcessToJobObject(self, job_handle, process_handle):
            self.assigned.append((job_handle, process_handle))
            return True

        def CloseHandle(self, handle):
            self.closed.append(handle)
            return True

    api = FakeKernel32()
    job = job_type.create(api)
    job.assign_current_process()
    job.close()
    job.close()

    assert api.created == 1
    assert api.info_class == 9
    assert api.limit_flags == runner_mod._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    assert api.assigned == [(101, 202)]
    assert api.closed == [101]


def test_windows_job_preserves_configuration_error_before_close(monkeypatch) -> None:
    """Handle cleanup must not overwrite the Win32 error we report."""
    from scripts import run_tests_parallel as runner_mod

    error_state = {"code": 123}

    class FakeKernel32:
        def CreateJobObjectW(self, security_attributes, name):
            return 101

        def SetInformationJobObject(self, handle, info_class, info, size):
            return False

        def CloseHandle(self, handle):
            error_state["code"] = 6
            return True

    monkeypatch.setattr(
        runner_mod.ctypes,
        "get_last_error",
        lambda: error_state["code"],
        raising=False,
    )

    with pytest.raises(OSError) as exc_info:
        runner_mod._WindowsJob.create(FakeKernel32())

    assert exc_info.value.errno == 123


def test_windows_runner_job_is_assigned_before_tests_can_start() -> None:
    """The runner joins the Job before any child process can be submitted."""
    from scripts import run_tests_parallel as runner_mod

    events = []

    class FakeJob:
        def assign_current_process(self):
            events.append("assign-runner")

        def close(self):
            events.append("close")

    job = runner_mod._create_windows_runner_job(lambda: FakeJob())

    assert isinstance(job, FakeJob)
    assert events == ["assign-runner"]


def test_windows_runner_job_closes_if_assignment_fails() -> None:
    """Containment setup fails closed without leaking the Job handle."""
    from scripts import run_tests_parallel as runner_mod

    events = []

    class FakeJob:
        def assign_current_process(self):
            events.append("assign-runner")
            raise OSError("injected assignment failure")

        def close(self):
            events.append("close")

    with pytest.raises(OSError, match="injected assignment failure"):
        runner_mod._create_windows_runner_job(lambda: FakeJob())

    assert events == ["assign-runner", "close"]


def test_main_establishes_windows_job_before_submitting_tests(
    monkeypatch, tmp_path: Path
) -> None:
    """Windows containment must exist before the worker pool runs pytest."""
    from scripts import run_tests_parallel as runner_mod

    probe = tmp_path / "test_windows_containment_probe.py"
    probe.write_text("def test_probe():\n    assert True\n", encoding="utf-8")
    events = []
    fake_job = object()

    def fake_create_job():
        events.append("job-created-and-assigned")
        return fake_job

    def fake_run_one_file(file, *args, **kwargs):
        events.append("test-submitted")
        return file, 0, "1 passed in 0.01s\n", {"passed": 1}, 0.01

    monkeypatch.setattr(runner_mod.sys, "platform", "win32")
    monkeypatch.setattr(runner_mod, "_create_windows_runner_job", fake_create_job)
    monkeypatch.setattr(runner_mod, "_run_one_file", fake_run_one_file)
    monkeypatch.setattr(runner_mod, "_save_durations", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path(runner_mod.__file__)),
            "--files",
            str(probe),
            "--file-retries",
            "0",
            "-j",
            "1",
            "-q",
        ],
    )

    assert runner_mod.main() == 0
    assert events == ["job-created-and-assigned", "test-submitted"]


def test_kill_tree_docs_state_platform_hard_termination_semantics() -> None:
    """Docs must not imply Windows descendants receive Python SIGTERM."""
    from inspect import getdoc
    from scripts import run_tests_parallel as runner_mod

    documentation = getdoc(runner_mod._kill_tree) or ""
    assert "POSIX process-group SIGKILL" in documentation
    assert "Windows Job Object" in documentation
    assert "hard termination" in documentation
    assert "Python SIGTERM handlers do not run" in documentation


def test_nested_signal_before_received_signal_assignment_has_one_winner(
    monkeypatch,
) -> None:
    """The winner claim must be atomic before ``received_signal`` is assigned."""
    from scripts import run_tests_parallel as runner_mod

    shutdown_requested = threading.Event()
    active_processes_lock = threading.Lock()
    fake_proc = object()
    active_processes: dict[int, tuple[Any, int | None]] = {1: (fake_proc, None)}
    controller = runner_mod._ShutdownController(
        shutdown_requested,
        active_processes,
        active_processes_lock,
    )
    killed = []

    class PreAssignmentNestingGuard:
        def __init__(self):
            self.lock = threading.Lock()
            self.injected = False

        def acquire(self, blocking=True):
            acquired = self.lock.acquire(blocking=False)
            if acquired and not self.injected:
                self.injected = True
                controller.handle_signal(signal.SIGINT, None)
            return acquired

    guard = PreAssignmentNestingGuard()
    controller._winner_guard = guard
    monkeypatch.setattr(
        runner_mod,
        "_kill_tree",
        lambda proc, pgid=None: killed.append((proc, pgid)),
    )

    controller.handle_signal(signal.SIGTERM, None)

    assert guard.injected
    assert shutdown_requested.is_set()
    assert controller.received_signal == signal.SIGTERM
    assert killed == [(fake_proc, None)]


def test_repeated_signal_does_not_reenter_active_cleanup(monkeypatch) -> None:
    """A nested second signal must not run active-process cleanup twice."""
    from scripts import run_tests_parallel as runner_mod

    shutdown_requested = threading.Event()
    active_processes_lock = threading.Lock()
    fake_proc = object()
    active_processes: dict[int, tuple[Any, int | None]] = {1: (fake_proc, None)}
    controller = runner_mod._ShutdownController(
        shutdown_requested,
        active_processes,
        active_processes_lock,
    )
    killed = []

    def reentrant_kill(proc, pgid=None):
        killed.append((proc, pgid))
        if len(killed) == 1:
            controller.handle_signal(signal.SIGINT, None)

    monkeypatch.setattr(runner_mod, "_kill_tree", reentrant_kill)
    controller.handle_signal(signal.SIGTERM, None)

    assert shutdown_requested.is_set()
    assert controller.received_signal == signal.SIGTERM
    assert killed == [(fake_proc, None)]


def test_file_retry_self_heals_and_prints_both_attempts(tmp_path: Path) -> None:
    """A pass-on-retry is green, loud, and retains the failing traceback."""
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    marker = tmp_path / "ran-once"
    probe = tmp_path / "test_flaky_probe.py"
    probe.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path

            def test_flaky_once():
                marker = Path({str(marker)!r})
                if not marker.exists():
                    marker.write_text("failed once")
                    assert False, "simulated first-attempt flake"
                assert True
            """
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--files",
            str(probe),
            "--file-retries",
            "1",
            "-j",
            "1",
            "-q",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout
    assert "FLAKY file" in proc.stdout
    assert "simulated first-attempt flake" in proc.stdout
    assert "first-attempt output" in proc.stdout
    assert "retry output" in proc.stdout




# ---------------------------------------------------------------------------
# Zero-collection is not a pass; node ids are translated, not dropped.
#
# Both behaviors were real foot-guns: a run where NOTHING was collected printed
# "0 tests passed, 0 failed (100% complete)" (reads green), and a pytest node id
# (`file.py::Class::test`) was silently discarded by path discovery so the run
# ended with "No test files to run" while looking like an accepted selector.


def test_zero_collected_across_run_fails_and_says_so(tmp_path: Path) -> None:
    """A -k that matches nothing must FAIL, not report a green summary."""
    probe_dir = _make_probe_dir(tmp_path)
    proc = _run_runner(probe_dir, "-k", "zzz_matches_nothing")
    assert proc.returncode == 1, proc.stdout
    assert "NO TESTS RAN" in proc.stdout
    assert "NOT a pass" in proc.stdout




def test_node_id_selector_runs_the_named_test(tmp_path: Path) -> None:
    """``file.py::test_alpha`` runs that test instead of discovering nothing."""
    probe_dir = _make_probe_dir(tmp_path)
    target = probe_dir / "test_flagprobe.py"
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_tests_parallel.py"),
         f"{target}::test_alpha", "-j", "1", "--file-timeout", "30"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout
    assert "No test files to run" not in proc.stdout
    assert "node id" in proc.stdout  # explains the translation
    # Ran exactly the one selected test, not both in the file.
    assert "1 tests passed" in proc.stdout


def test_explicit_k_wins_over_node_id_inference(tmp_path: Path) -> None:
    """A caller's own ``-k`` is not overridden by the node-id translation."""
    probe_dir = _make_probe_dir(tmp_path)
    target = probe_dir / "test_flagprobe.py"
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_tests_parallel.py"),
         f"{target}::test_alpha", "-k", "test_beta",
         "-j", "1", "--file-timeout", "30"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=60,
    )
    # -k test_beta wins: one test ran, and it wasn't filtered to nothing.
    assert proc.returncode == 0, proc.stdout
    assert "1 tests passed" in proc.stdout
