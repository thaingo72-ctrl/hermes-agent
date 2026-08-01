"""A session that settles into another git worktree re-anchors onto it.

The desktop already follows a session that moves (``followActiveSessionCwd``
refreshes the project tree and scopes into the new project), but it only ever
sees a move when the backend reports one on ``session.info``. These tests
exercise the backend half against real git worktrees on disk.
"""

from __future__ import annotations

import os
import subprocess

import pytest

import agent.runtime_cwd as rc
import tools.terminal_tool as terminal_tool
import tui_gateway.server as server


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo_with_worktree(tmp_path):
    """A real repo on ``main`` plus a linked worktree on ``feature``."""
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hi\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")

    worktree = tmp_path / "proj-feature"
    _git(repo, "worktree", "add", "-b", "feature", str(worktree))

    from tui_gateway import git_probe

    git_probe.invalidate()
    yield repo, worktree
    git_probe.invalidate()


@pytest.fixture
def session(repo_with_worktree):
    repo, _ = repo_with_worktree
    key = "sess-follow"
    rc.clear_session_cwd(key)
    yield {"session_key": key, "cwd": str(repo), "source": "desktop"}
    rc.clear_session_cwd(key)


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_persist_session_git_meta", lambda *_a: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _s: None)
    monkeypatch.setattr(server, "_is_local_terminal_backend", lambda: True)


def test_settling_in_a_worktree_reanchors_the_session(session, repo_with_worktree):
    """The whole reported bug: work goes to the worktree, the session says main."""
    _, worktree = repo_with_worktree
    rc.record_session_cwd(session["session_key"], str(worktree))

    assert server._reconcile_session_cwd_from_terminal(session) is True
    assert session["cwd"] == str(worktree)
    # The move is the session's workspace now, so it earns a persisted row.
    assert session["explicit_cwd"] is True


def test_a_subdirectory_of_the_same_checkout_is_not_a_move(session, repo_with_worktree):
    repo, _ = repo_with_worktree
    sub = repo / "src"
    sub.mkdir()
    rc.record_session_cwd(session["session_key"], str(sub))

    assert server._reconcile_session_cwd_from_terminal(session) is False
    assert session["cwd"] == str(repo)


def test_browsing_outside_a_repo_is_not_a_move(session, repo_with_worktree, tmp_path):
    """`cd /tmp` to read a log must not re-home the workspace."""
    repo, _ = repo_with_worktree
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    rc.record_session_cwd(session["session_key"], str(scratch))

    assert server._reconcile_session_cwd_from_terminal(session) is False
    assert session["cwd"] == str(repo)


def test_remote_backends_do_not_reanchor(session, repo_with_worktree, monkeypatch):
    """A remote cwd names a path on the host, not one this gateway can probe."""
    repo, worktree = repo_with_worktree
    monkeypatch.setattr(server, "_is_local_terminal_backend", lambda: False)
    rc.record_session_cwd(session["session_key"], str(worktree))

    assert server._reconcile_session_cwd_from_terminal(session) is False
    assert session["cwd"] == str(repo)


def test_settled_session_info_reports_the_worktree_branch(
    session, repo_with_worktree, monkeypatch
):
    """End of turn: the emitted session.info is what the desktop follows."""
    _, worktree = repo_with_worktree
    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(server, "_emit", lambda ev, sid, payload=None: emitted.append((ev, sid, payload or {})))
    rc.record_session_cwd(session["session_key"], str(worktree))

    server._emit_settled_session_info("sid-1", session, agent=None)

    assert len(emitted) == 1
    event, sid, payload = emitted[0]
    assert (event, sid) == ("session.info", "sid-1")
    assert payload["cwd"] == str(worktree)
    assert payload["branch"] == "feature"


def test_reconcile_ignores_a_foreign_sessions_record(session, repo_with_worktree):
    """cwd records are per session key — another chat's move must not leak in."""
    repo, worktree = repo_with_worktree
    rc.record_session_cwd("someone-else", str(worktree))

    assert server._reconcile_session_cwd_from_terminal(session) is False
    assert session["cwd"] == str(repo)
    rc.clear_session_cwd("someone-else")


def test_os_normalized_paths_are_not_a_move(session, repo_with_worktree):
    """A trailing-slash / unnormalized record is the same dir, not a relocation."""
    repo, _ = repo_with_worktree
    rc.record_session_cwd(session["session_key"], str(repo) + os.sep)

    assert server._reconcile_session_cwd_from_terminal(session) is False


def test_explicit_cwd_switch_cleans_old_environment_before_recording_new(
    tmp_path, monkeypatch
):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    session = {"session_key": "sess-switch", "cwd": str(old), "source": "desktop"}
    events: list[tuple[str, str]] = []
    cleaned: list[str] = []

    class FakeEnv:
        def cleanup(self):
            cleaned.append("default")

    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_persist_session_git_meta", lambda *_a: None)
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {"default": 1.0})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {"default": object()})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})

    import tools.file_tools as ft

    monkeypatch.setattr(ft, "_file_ops_cache", {"default": object()})

    def register(updated_session: dict) -> None:
        events.append(("register", updated_session["session_key"]))
        assert "default" not in terminal_tool._active_environments
        assert "default" not in ft._file_ops_cache
        assert updated_session["cwd"] == str(new.resolve())
        rc.record_session_cwd(
            updated_session["session_key"], updated_session["cwd"]
        )

    rc.record_session_cwd("sess-switch", str(old))
    monkeypatch.setattr(server, "_register_session_cwd", register)

    assert server._set_session_cwd(session, str(new)) == str(new.resolve())

    assert cleaned == ["default"]
    assert events == [("register", "sess-switch")]
    assert rc.get_session_cwd("sess-switch") == str(new.resolve())
