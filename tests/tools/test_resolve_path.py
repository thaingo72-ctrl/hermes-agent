"""Tests for _resolve_path() — TERMINAL_CWD-aware path resolution in file_tools."""

from pathlib import Path

import agent.runtime_cwd as rc
import pytest


@pytest.fixture(autouse=True)
def _clear_default_runtime_cwd():
    rc.clear_session_cwd("default")
    yield
    rc.clear_session_cwd("default")

class TestResolvePath:
    """Verify _resolve_path respects TERMINAL_CWD for worktree isolation."""

    def test_relative_path_uses_terminal_cwd(self, monkeypatch, tmp_path):
        """Relative paths resolve against TERMINAL_CWD, not process CWD."""
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        from tools.file_tools import _resolve_path

        result = _resolve_path("foo/bar.py")
        assert result == (tmp_path / "foo" / "bar.py")


    def test_relative_path_prefers_recorded_session_cwd(self, monkeypatch, tmp_path):
        """The session's recorded cwd must win after the terminal changes directory."""
        start_dir = tmp_path / "start"
        live_dir = tmp_path / "worktree"
        start_dir.mkdir()
        live_dir.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(start_dir))

        from tools import file_tools

        task_id = "live-cwd"
        # The session's completed `cd` recorded the new directory.
        rc.record_session_cwd(task_id, str(live_dir))

        try:
            result = file_tools._resolve_path("nested/file.txt", task_id=task_id)
        finally:
            rc.clear_session_cwd(task_id)

        assert result == live_dir / "nested" / "file.txt"
