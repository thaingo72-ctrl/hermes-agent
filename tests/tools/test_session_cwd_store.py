"""Session-cwd record store.

The store is the single source of truth for per-session working directories:
every path that learns a session's live cwd records it under the raw session
key, and tool callers read that record instead of shared environment state.
"""

import pytest

import tools.terminal_tool as tt
import agent.runtime_cwd as rc


@pytest.fixture(autouse=True)
def _clean_store(monkeypatch):
    monkeypatch.setattr(rc, "_SESSION_CWDS", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})


class TestRecordSemantics:
    def test_records_are_keyed_by_raw_session_key(self):
        rc.record_session_cwd("sess-a", "/wt/a")
        rc.record_session_cwd("sess-b", "/wt/b")
        # No cross-talk: each session reads back exactly its own record.
        assert rc.get_session_cwd("sess-a") == "/wt/a"
        assert rc.get_session_cwd("sess-b") == "/wt/b"
        assert rc.get_session_cwd("sess-c") is None


    def test_clear_drops_only_the_named_session(self):
        rc.record_session_cwd("sess-a", "/wt/a")
        rc.record_session_cwd("sess-b", "/wt/b")
        rc.clear_session_cwd("sess-a")
        assert rc.get_session_cwd("sess-a") is None
        assert rc.get_session_cwd("sess-b") == "/wt/b"


class TestCleanupRouting:
    def test_prefers_an_existing_raw_task_key(self, monkeypatch):
        cleaned: list[str] = []

        class FakeEnv:
            def cleanup(self):
                cleaned.append("raw")

        monkeypatch.setattr(tt, "_active_environments", {"session-123": FakeEnv()})
        monkeypatch.setattr(tt, "_last_activity", {"session-123": 1.0})
        monkeypatch.setattr(tt, "_creation_locks", {"session-123": object()})
        monkeypatch.setattr(tt, "_resolve_container_task_id", lambda _task_id: "default")

        tt.cleanup_task_environment("session-123")

        assert cleaned == ["raw"]
        assert "session-123" not in tt._active_environments


class TestRecordWriteSites:
    def test_session_boundary_seeds_the_session_record_directly(self):
        """A registered workspace cwd IS the session's cwd until a `cd`."""
        rc.record_session_cwd("desktop-sess", "/wt/desktop")
        assert rc.get_session_cwd("desktop-sess") == "/wt/desktop"


    def test_reregistration_updates_the_record_directly(self):
        """ACP session/load switching project roots mid-session."""
        rc.record_session_cwd("acp-sess", "/proj/one")
        rc.record_session_cwd("acp-sess", "/proj/two")
        assert rc.get_session_cwd("acp-sess") == "/proj/two"

    def test_nonpersistent_cleanup_preserves_session_cwd(self):
        rc.record_session_cwd("sess-a", "/logical/cwd")

        tt.clear_task_env_overrides("sess-a")

        assert rc.get_session_cwd("sess-a") == "/logical/cwd"

    def test_true_session_teardown_clears_session_cwd(self):
        rc.record_session_cwd("sess-a", "/logical/cwd")

        tt.teardown_session_runtime_cwd("sess-a")

        assert rc.get_session_cwd("sess-a") is None

    def test_agent_close_releases_resources_without_clearing_runtime_cwd(self, monkeypatch):
        import threading

        import run_agent
        from run_agent import AIAgent

        calls = []
        monkeypatch.setattr(run_agent, "cleanup_vm", lambda task_id: calls.append(("vm", task_id)))
        monkeypatch.setattr(run_agent, "cleanup_browser", lambda task_id: calls.append(("browser", task_id)))
        monkeypatch.setattr(
            "tools.process_registry.process_registry.kill_all",
            lambda task_id=None: calls.append(("process", task_id)),
        )
        monkeypatch.setattr(
            "tools.computer_use.release_computer_use_session",
            lambda task_id: calls.append(("computer", task_id)),
        )

        agent = AIAgent.__new__(AIAgent)
        agent.session_id = "live-session"
        agent._active_children_lock = threading.Lock()
        agent._active_children = []
        agent.client = None
        agent._session_db = None
        agent._end_session_on_close = False
        agent._close_cached_request_openai_client = lambda reason: None
        rc.record_session_cwd("live-session", "/logical/cwd")

        agent.close()

        assert rc.get_session_cwd("live-session") == "/logical/cwd"
        assert ("vm", "live-session") in calls


class TestPostCommandDualWrite:
    """The env's post-command cwd tracking must mirror into the session record."""

    def _run(self, monkeypatch, task_id, env):
        import json
        monkeypatch.setattr(tt, "_active_environments", {task_id: env})
        monkeypatch.setattr(tt, "_last_activity", {})
        monkeypatch.setattr(
            tt, "_get_env_config",
            lambda: {"env_type": "local", "cwd": "/default", "timeout": 60,
                     "lifetime_seconds": 3600},
        )
        monkeypatch.setattr(
            tt, "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )
        return json.loads(tt.terminal_tool(command="cd /new/dir", task_id=task_id))

    def test_cd_result_is_recorded_under_the_session_key(self, monkeypatch):
        class FakeEnv:
            env = {}
            cwd = "/start"
            def execute(self, command, **kwargs):
                # Simulate the env's own post-command tracking (marker parse).
                self.cwd = "/new/dir"
                return {"output": "", "returncode": 0, "final_cwd": "/new/dir"}

        result = self._run(monkeypatch, "sess-a", FakeEnv())
        assert result["exit_code"] == 0
        assert rc.get_session_cwd("sess-a") == "/new/dir"
        # And ONLY that session's record was touched.
        assert rc.get_session_cwd("sess-b") is None

    def test_records_final_cwd_from_result_metadata_not_mutable_env(self, monkeypatch):
        """The terminal result owns final cwd; env.cwd is shared mutable state."""
        import json

        class FakeEnv:
            env = {}
            cwd = "/stomped/by/another/session"

            def execute(self, command, **kwargs):
                return {"output": "", "returncode": 0, "final_cwd": "/sess-a/final"}

        monkeypatch.setattr(tt, "_active_environments", {"sess-a": FakeEnv()})
        monkeypatch.setattr(tt, "_last_activity", {})
        monkeypatch.setattr(
            tt,
            "_get_env_config",
            lambda: {"env_type": "local", "cwd": "/default", "timeout": 60,
                     "lifetime_seconds": 3600},
        )
        monkeypatch.setattr(
            tt,
            "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )

        result = json.loads(tt.terminal_tool(command="pwd", task_id="sess-a"))

        assert result["exit_code"] == 0
        assert "final_cwd" not in result
        assert rc.get_session_cwd("sess-a") == "/sess-a/final"

    def test_barrier_controlled_two_session_final_cwd_isolation(self, monkeypatch):
        """Two sessions sharing one env must record their own result final_cwd."""
        import json
        import threading

        started = threading.Barrier(3)
        release = threading.Event()
        lock = threading.Lock()

        class FakeEnv:
            env = {}
            cwd = "/shared/initial"

            def execute(self, command, **kwargs):
                session = command.removeprefix("cmd-")
                started.wait(timeout=2)
                release.wait(timeout=2)
                with lock:
                    self.cwd = f"/shared/stomped/{session}"
                return {"output": "", "returncode": 0, "final_cwd": f"/{session}/final"}

        fake = FakeEnv()
        monkeypatch.setattr(tt, "_active_environments", {"default": fake})
        monkeypatch.setattr(tt, "_last_activity", {})
        monkeypatch.setattr(tt, "_resolve_container_task_id", lambda _task_id: "default")
        monkeypatch.setattr(
            tt,
            "_get_env_config",
            lambda: {"env_type": "local", "cwd": "/default", "timeout": 60,
                     "lifetime_seconds": 3600},
        )
        monkeypatch.setattr(
            tt,
            "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )

        results: dict[str, dict] = {}

        def run(session: str) -> None:
            results[session] = json.loads(
                tt.terminal_tool(command=f"cmd-{session}", task_id=session)
            )

        threads = [threading.Thread(target=run, args=(s,)) for s in ("sess-a", "sess-b")]
        for thread in threads:
            thread.start()
        started.wait(timeout=2)
        release.set()
        for thread in threads:
            thread.join(timeout=2)

        assert "final_cwd" not in results["sess-a"]
        assert "final_cwd" not in results["sess-b"]
        assert rc.get_session_cwd("sess-a") == "/sess-a/final"
        assert rc.get_session_cwd("sess-b") == "/sess-b/final"
        assert fake.cwd in {"/shared/stomped/sess-a", "/shared/stomped/sess-b"}

    def test_envs_without_result_cwd_keep_initialized_record(self, monkeypatch):
        class FakeEnv:
            env = {}
            def execute(self, command, **kwargs):
                return {"output": "", "returncode": 0}

        result = self._run(monkeypatch, "sess-a", FakeEnv())
        assert result["exit_code"] == 0
        assert rc.get_session_cwd("sess-a") == "/default"

    def test_base_environment_final_cwd_is_call_local_under_interleaving(self):
        import threading

        from tools.environments.base import BaseEnvironment

        first_parsed = threading.Event()
        allow_first_return = threading.Event()

        class InterleavingEnv(BaseEnvironment):
            def _run_bash(self, cmd_string, **_kwargs):
                return cmd_string

            def _wait_for_process(self, proc, **_kwargs):
                cwd = "/A" if "cmd-a" in proc else "/B"
                return {
                    "output": f"out\n{self._cwd_marker}{cwd}{self._cwd_marker}\n",
                    "returncode": 0,
                }

            def _update_cwd(self, result):
                parsed = super()._update_cwd(result)
                if parsed == "/A":
                    first_parsed.set()
                    allow_first_return.wait(timeout=2)
                return parsed

            def cleanup(self):
                pass

        env = InterleavingEnv(cwd="/start", timeout=10)
        results = {}

        thread_a = threading.Thread(
            target=lambda: results.setdefault(
                "a", env.execute("cmd-a", include_final_cwd=True)
            )
        )
        thread_a.start()
        assert first_parsed.wait(timeout=2)
        results["b"] = env.execute("cmd-b", include_final_cwd=True)
        allow_first_return.set()
        thread_a.join(timeout=2)

        assert results["b"]["final_cwd"] == "/B"
        assert results["a"]["final_cwd"] == "/A"
        assert env.cwd == "/B"


class TestFileToolsReadTheRecord:
    """File-tool path resolution prefers the session's own record."""

    def test_two_sessions_resolve_into_their_own_recorded_cwds(self, tmp_path, monkeypatch):
        import tools.file_tools as ft

        wt_a = tmp_path / "wt_a"
        wt_b = tmp_path / "wt_b"
        for d in (wt_a, wt_b):
            d.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setattr(ft, "_file_ops_cache", {})
        monkeypatch.setattr(tt, "_active_environments", {})

        # Each session ran commands that recorded its own cwd. No env alive,
        # no registered overrides — just the records.
        rc.record_session_cwd("sess-a", str(wt_a))
        rc.record_session_cwd("sess-b", str(wt_b))

        assert ft._resolve_path_for_task("f.py", task_id="sess-a") == (wt_a / "f.py")
        assert ft._resolve_path_for_task("f.py", task_id="sess-b") == (wt_b / "f.py")

    def test_record_beats_foreign_env_cwd_without_ownership_metadata(self, tmp_path, monkeypatch):
        """The leak-A scenario, solved structurally: the shared env's cwd is
        never consulted for path resolution — only the session's own record."""
        import tools.file_tools as ft

        wt_a = tmp_path / "wt_a"
        wt_b = tmp_path / "wt_b"
        for d in (wt_a, wt_b):
            d.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setattr(ft, "_file_ops_cache", {})

        class _Env:
            cwd = str(wt_b)  # another session's leftover cd on the shared env

        monkeypatch.setattr(tt, "_active_environments", {"default": _Env()})
        rc.record_session_cwd("sess-a", str(wt_a))

        resolved = ft._resolve_path_for_task("f.py", task_id="sess-a")
        assert resolved == (wt_a / "f.py")
        assert not str(resolved).startswith(str(wt_b))


class TestDelegateSeedsChildRecord:
    def test_child_record_seeded_from_parent_then_isolated(self):
        rc.record_session_cwd("parent-task", "/parent/worktree")
        # what delegate_tool does at spawn:
        rc.record_session_cwd("child-1", rc.get_session_cwd("parent-task"))

        assert rc.get_session_cwd("child-1") == "/parent/worktree"
        # child cds somewhere; parent record must be untouched.
        rc.record_session_cwd("child-1", "/child/scratch")
        assert rc.get_session_cwd("parent-task") == "/parent/worktree"
        assert rc.get_session_cwd("child-1") == "/child/scratch"


class TestCommandCwdReadsTheRecord:
    """_resolve_command_cwd: workdir > session record > default. Nothing else."""

    def test_record_beats_default(self):
        rc.record_session_cwd("sess-a", "/my/worktree")
        resolved = tt._resolve_command_cwd(
            workdir=None,
            default_cwd="/config/default",
            session_key="sess-a",
        )
        assert resolved == "/my/worktree"


    def test_other_sessions_record_is_not_consulted(self):
        rc.record_session_cwd("sess-b", "/other/worktree")
        resolved = tt._resolve_command_cwd(
            workdir=None,
            default_cwd="/config/default",
            session_key="sess-a",
        )
        assert resolved == "/config/default"

    def test_cd_then_next_command_runs_in_the_new_dir(self, tmp_path, monkeypatch):
        """E2E through terminal_tool: the record round-trips cd state."""
        import json

        class FakeEnv:
            env = {}
            cwd = "/start"
            def execute(self, command, **kwargs):
                self.last_cwd_arg = kwargs.get("cwd")
                if command.startswith("cd "):
                    self.cwd = command[3:]
                return {"output": "", "returncode": 0, "final_cwd": self.cwd}

        fake = FakeEnv()
        rc.record_session_cwd("sess-a", str(tmp_path))
        monkeypatch.setattr(tt, "_active_environments", {"sess-a": fake})
        monkeypatch.setattr(tt, "_last_activity", {})
        monkeypatch.setattr(
            tt, "_get_env_config",
            lambda: {"env_type": "local", "cwd": "/default", "timeout": 60,
                     "lifetime_seconds": 3600},
        )
        monkeypatch.setattr(
            tt, "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )

        json.loads(tt.terminal_tool(command="cd /project", task_id="sess-a"))
        assert rc.get_session_cwd("sess-a") == "/project"
        json.loads(tt.terminal_tool(command="pwd", task_id="sess-a"))
        assert fake.last_cwd_arg == "/project"

    def test_cd_survives_turn_finalize_for_terminal_and_file_resolution(
        self, tmp_path, monkeypatch
    ):
        """Clearing per-turn context must not clear the logical session cwd."""
        import json

        import tools.file_tools as ft
        from gateway.session_context import clear_session_vars, set_session_vars

        project = tmp_path / "project"
        project.mkdir()

        class FakeEnv:
            env = {}
            cwd = str(tmp_path)

            def execute(self, command, **kwargs):
                self.last_cwd_arg = kwargs.get("cwd")
                if command.startswith("cd "):
                    self.cwd = command[3:]
                return {"output": "", "returncode": 0, "final_cwd": self.cwd}

        fake = FakeEnv()
        monkeypatch.setattr(tt, "_active_environments", {"sess-a": fake})
        monkeypatch.setattr(tt, "_last_activity", {})
        monkeypatch.setattr(
            tt,
            "_get_env_config",
            lambda: {"env_type": "local", "cwd": str(tmp_path), "timeout": 60,
                     "lifetime_seconds": 3600},
        )
        monkeypatch.setattr(
            tt,
            "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )

        tokens = set_session_vars(session_key="sess-a", cwd=str(tmp_path))
        json.loads(tt.terminal_tool(command=f"cd {project}", task_id="sess-a"))
        clear_session_vars(tokens)

        json.loads(tt.terminal_tool(command="pwd", task_id="sess-a"))

        assert fake.last_cwd_arg == str(project)
        assert ft._resolve_path_for_task("next.txt", task_id="sess-a") == project / "next.txt"
