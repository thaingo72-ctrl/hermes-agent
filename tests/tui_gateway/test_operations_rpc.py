from __future__ import annotations

import functools
import threading
import time
import types

from tui_gateway import methods_operations, methods_tools, server


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "cols": 80,
        "slash_worker": None,
        **extra,
    }


def _call(method: str, params: dict) -> dict:
    return server.handle_request(
        {"jsonrpc": "2.0", "id": "1", "method": method, "params": params}
    )


def test_operation_handlers_are_registered_from_operations_module():
    names = {
        "slash.exec": methods_operations.slash_exec,
        "insights.get": methods_operations.insights_get,
        "rollback.list": methods_operations.rollback_list,
        "rollback.restore": methods_operations.rollback_restore,
        "rollback.diff": methods_operations.rollback_diff,
        "shell.exec": methods_operations.shell_exec,
    }

    for rpc_name, handler in names.items():
        registered = server._methods[rpc_name]
        assert isinstance(registered, functools.partial)
        assert registered.func is handler
        assert "services" in registered.keywords


def test_legacy_tools_module_no_longer_registers_operation_handlers():
    legacy_names = {name for name, _handler in methods_tools._registry._pending}

    assert not legacy_names & {
        "slash.exec",
        "insights.get",
        "rollback.list",
        "rollback.restore",
        "rollback.diff",
        "shell.exec",
        "system.battery",
        "process.stop",
        "process.list",
        "process.kill",
        "reload.mcp",
        "reload.env",
        "commands.catalog",
        "cli.exec",
        "command.resolve",
        "command.dispatch",
        "browser.manage",
        "plugins.list",
        "plugins.manage",
        "config.show",
        "tools.list",
        "tools.show",
        "tools.configure",
        "toolsets.list",
        "agents.list",
        "cron.manage",
        "learning.frames",
        "learning.detail",
        "learning.delete",
        "learning.edit",
        "skills.manage",
        "skills.reload",
    }


def test_rollback_restore_is_serialized_per_session():
    active = 0
    overlaps = 0
    guard = threading.Lock()

    class _Mgr:
        enabled = True

        def list_checkpoints(self, cwd):
            return [{"hash": "abc123"}]

        def restore(self, cwd, target, file_path=None):
            nonlocal active, overlaps
            with guard:
                active += 1
                if active > 1:
                    overlaps += 1
            time.sleep(0.05)
            with guard:
                active -= 1
            return {"success": True, "message": "restored"}

    server._sessions["rollback-lock"] = _session(
        agent=types.SimpleNamespace(_checkpoint_mgr=_Mgr())
    )
    results = []

    def run_restore(n: int) -> None:
        results.append(
            _call(
                "rollback.restore",
                {
                    "session_id": "rollback-lock",
                    "hash": "abc123",
                    "file_path": f"src/file{n}.py",
                },
            )
        )

    try:
        threads = [threading.Thread(target=run_restore, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
    finally:
        server._sessions.pop("rollback-lock", None)

    assert overlaps == 0
    assert len(results) == 2
    assert all(result["result"]["success"] is True for result in results)
