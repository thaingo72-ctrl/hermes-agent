"""Operational JSON-RPC handlers for the TUI gateway.

This module owns stateful operator commands that used to live in the legacy
``methods_tools`` rebinding seam. Dependencies are explicit and frozen so each
handler can be tested without mutating imported state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable


@dataclass(frozen=True)
class SlashOperationsServices:
    sess_nowait: Callable[[dict, Any], tuple[dict | None, dict | None]]
    live_slash_command_output: Callable[[str, dict | None, str, str], str | None]
    command_dispatch: Callable[[Any, dict], dict]
    resolve_model: Callable[[], str]
    worker_factory: Callable[..., Any]
    attach_worker: Callable[[str, dict, Any], None]
    mirror_slash_side_effects: Callable[[str, dict, str], str]
    sessions_lock: Any
    make_lock: Callable[[], Any]
    pending_input_commands: frozenset[str]
    worker_blocked_commands: frozenset[str]
    resolve_bundle_command_key: Callable[[str], str | None]
    resolve_command: Callable[[str], Any]
    get_skill_commands: Callable[[], dict]
    get_plugin_command_handler: Callable[[str], Callable[[str], Any] | None]
    resolve_plugin_command_result: Callable[[Any], Any]


@dataclass(frozen=True)
class InsightsOperationsServices:
    get_db: Callable[[], Any]
    db_unavailable_error: Callable[[Any], dict]
    time: Callable[[], float]


@dataclass(frozen=True)
class RollbackOperationsServices:
    sess: Callable[[dict, Any], tuple[dict | None, dict | None]]
    with_checkpoints: Callable[[dict, Callable[[Any, str], Any]], Any]
    resolve_checkpoint_hash: Callable[[Any, str, str], str]
    render_diff: Callable[[str, int], str]
    sessions_lock: Any
    make_lock: Callable[[], Any]


@dataclass(frozen=True)
class ShellOperationsServices:
    detect_hardline_command: Callable[[str], tuple[bool, str]]
    detect_dangerous_command: Callable[[str], tuple[bool, Any, str]]
    subprocess_run: Callable[..., Any]
    timeout_expired: type[BaseException]
    devnull: Any
    getcwd: Callable[[], str]
    windows_hide_flags: Callable[[], int]


@dataclass(frozen=True)
class OperationsServices:
    slash: SlashOperationsServices
    insights: InsightsOperationsServices
    rollback: RollbackOperationsServices
    shell: ShellOperationsServices


def _ok(rid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}


def _session_lock(session: dict, key: str, services) -> Any:
    with services.sessions_lock:
        return session.setdefault(key, services.make_lock())


def slash_exec(rid, params: dict, services: SlashOperationsServices) -> dict:
    session, err = services.sess_nowait(params, rid)
    if err:
        return err

    cmd = params.get("command", "").strip()
    if not cmd:
        return _err(rid, 4004, "empty command")

    cmd_text = cmd.lstrip("/") if cmd.startswith("/") else cmd
    cmd_parts = cmd_text.split(maxsplit=1)
    cmd_base = (cmd_parts[0] if cmd_parts else "").lower()
    cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
    sid = params.get("session_id", "")

    live_output = services.live_slash_command_output(sid, session, cmd_base, cmd_arg)
    if live_output is not None:
        return _ok(rid, {"output": live_output or "(no output)"})

    if cmd_base in services.pending_input_commands:
        return services.command_dispatch(
            rid,
            {
                "name": cmd_base,
                "arg": cmd_arg,
                "session_id": sid,
            },
        )

    if cmd_base in services.worker_blocked_commands:
        subcommand = cmd_arg.split(maxsplit=1)[0].lower() if cmd_arg else ""
        if subcommand in {"restore", "rewind"}:
            return _err(
                rid,
                4018,
                "snapshot restore mutates live config/state; use command.dispatch for /snapshot restore",
            )

    try:
        bundle_key = (
            services.resolve_bundle_command_key(cmd_base)
            if services.resolve_command(cmd_base) is None
            else None
        )
        if bundle_key is not None:
            return services.command_dispatch(
                rid,
                {
                    "name": bundle_key.lstrip("/"),
                    "arg": cmd_arg,
                    "session_id": sid,
                },
            )
    except Exception:
        pass

    try:
        cmd_key = f"/{cmd_base}"
        if cmd_key in services.get_skill_commands():
            return _err(rid, 4018, f"skill command: use command.dispatch for {cmd_key}")
    except Exception:
        pass

    plugin_handler = None
    if cmd_base:
        try:
            plugin_handler = services.get_plugin_command_handler(cmd_base)
        except Exception:
            plugin_handler = None

    if plugin_handler:
        try:
            result = services.resolve_plugin_command_result(plugin_handler(cmd_arg))
            return _ok(rid, {"output": str(result or "(no output)")})
        except Exception as exc:
            return _ok(rid, {"output": f"Plugin command error: {exc}"})

    worker = session.get("slash_worker")
    if not worker:
        spawn_lock = _session_lock(session, "_slash_spawn_lock", services)
        with spawn_lock:
            worker = session.get("slash_worker")
            if not worker:
                try:
                    model = getattr(session.get("agent"), "model", "")
                    if not model:
                        model = services.resolve_model()
                    worker = services.worker_factory(
                        session["session_key"],
                        model,
                        profile_home=session.get("profile_home"),
                    )
                    services.attach_worker(sid, session, worker)
                except Exception as exc:
                    return _err(rid, 5030, f"slash worker start failed: {exc}")

    try:
        output = worker.run(cmd)
        warning = services.mirror_slash_side_effects(sid, session, cmd)
        payload = {"output": output or "(no output)"}
        if warning:
            payload["warning"] = warning
        return _ok(rid, payload)
    except Exception as exc:
        try:
            worker.close()
        except Exception:
            pass
        session["slash_worker"] = None
        return _err(rid, 5030, str(exc))


def insights_get(rid, params: dict, services: InsightsOperationsServices) -> dict:
    days = params.get("days", 30)
    db = services.get_db()
    if db is None:
        return services.db_unavailable_error(rid)
    try:
        cutoff = services.time() - days * 86400
        rows = [
            s
            for s in db.list_sessions_rich(limit=500, compact_rows=True)
            if (s.get("started_at") or 0) >= cutoff
        ]
        return _ok(
            rid,
            {
                "days": days,
                "sessions": len(rows),
                "messages": sum(s.get("message_count", 0) for s in rows),
            },
        )
    except Exception as exc:
        return _err(rid, 5017, str(exc))


def rollback_list(rid, params: dict, services: RollbackOperationsServices) -> dict:
    session, err = services.sess(params, rid)
    if err:
        return err
    try:
        rollback_lock = _session_lock(session, "_rollback_lock", services)
        with rollback_lock:

            def go(mgr, cwd):
                if not mgr.enabled:
                    return _ok(rid, {"enabled": False, "checkpoints": []})
                return _ok(
                    rid,
                    {
                        "enabled": True,
                        "checkpoints": [
                            {
                                "hash": c.get("hash", ""),
                                "timestamp": c.get("timestamp", ""),
                                "message": c.get("message", ""),
                            }
                            for c in mgr.list_checkpoints(cwd)
                        ],
                    },
                )

            return services.with_checkpoints(session, go)
    except Exception as exc:
        return _err(rid, 5020, str(exc))


def rollback_restore(rid, params: dict, services: RollbackOperationsServices) -> dict:
    session, err = services.sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    file_path = params.get("file_path", "")
    if not target:
        return _err(rid, 4014, "hash required")
    if not file_path and session.get("running"):
        return _err(
            rid,
            4009,
            "session busy — /interrupt the current turn before full rollback.restore",
        )
    try:
        rollback_lock = _session_lock(session, "_rollback_lock", services)
        with rollback_lock:
            if not file_path and session.get("running"):
                return _err(
                    rid,
                    4009,
                    "session busy — /interrupt the current turn before full rollback.restore",
                )

            def go(mgr, cwd):
                resolved = services.resolve_checkpoint_hash(mgr, cwd, target)
                result = mgr.restore(cwd, resolved, file_path=file_path or None)
                if result.get("success") and not file_path:
                    removed = 0
                    with session["history_lock"]:
                        history = session.get("history", [])
                        last_user_idx = None
                        for i in range(len(history) - 1, -1, -1):
                            msg = history[i]
                            if msg.get("role") == "user" and not msg.get("display_kind"):
                                last_user_idx = i
                                break
                        if last_user_idx is not None:
                            removed = len(history) - last_user_idx
                            del history[last_user_idx:]
                        if removed:
                            session["history_version"] = (
                                int(session.get("history_version", 0)) + 1
                            )
                    result["history_removed"] = removed
                return result

            return _ok(rid, services.with_checkpoints(session, go))
    except Exception as exc:
        return _err(rid, 5021, str(exc))


def rollback_diff(rid, params: dict, services: RollbackOperationsServices) -> dict:
    session, err = services.sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    if not target:
        return _err(rid, 4014, "hash required")
    try:
        rollback_lock = _session_lock(session, "_rollback_lock", services)
        with rollback_lock:
            result = services.with_checkpoints(
                session,
                lambda mgr, cwd: mgr.diff(
                    cwd, services.resolve_checkpoint_hash(mgr, cwd, target)
                ),
            )
        raw = result.get("diff", "")[:4000]
        payload = {"stat": result.get("stat", ""), "diff": raw}
        rendered = services.render_diff(raw, session.get("cols", 80))
        if rendered:
            payload["rendered"] = rendered
        return _ok(rid, payload)
    except Exception as exc:
        return _err(rid, 5022, str(exc))


def shell_exec(rid, params: dict, services: ShellOperationsServices) -> dict:
    cmd = params.get("command", "")
    if not cmd:
        return _err(rid, 4004, "empty command")
    try:
        is_hardline, hardline_desc = services.detect_hardline_command(cmd)
        if is_hardline:
            return _err(
                rid,
                4005,
                f"blocked (hardline): {hardline_desc}. Use the agent for dangerous commands.",
            )
        is_dangerous, _, desc = services.detect_dangerous_command(cmd)
        if is_dangerous:
            return _err(
                rid,
                4005,
                f"blocked: {desc}. Use the agent for dangerous commands.",
            )
    except ImportError:
        return _err(
            rid,
            5001,
            "shell.exec unavailable: approval safety module not importable",
        )
    try:
        result = services.subprocess_run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=services.getcwd(),
            encoding="utf-8",
            errors="replace",
            stdin=services.devnull,
            creationflags=services.windows_hide_flags(),
        )
        return _ok(
            rid,
            {
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-2000:],
                "code": result.returncode,
            },
        )
    except services.timeout_expired:
        return _err(rid, 5002, "command timed out (30s)")
    except Exception as exc:
        return _err(rid, 5003, str(exc))


def register(
    methods: dict[str, Callable[[Any, dict], dict]],
    *,
    services: OperationsServices,
) -> None:
    methods["slash.exec"] = partial(slash_exec, services=services.slash)
    methods["insights.get"] = partial(insights_get, services=services.insights)
    methods["rollback.list"] = partial(rollback_list, services=services.rollback)
    methods["rollback.restore"] = partial(rollback_restore, services=services.rollback)
    methods["rollback.diff"] = partial(rollback_diff, services=services.rollback)
    methods["shell.exec"] = partial(shell_exec, services=services.shell)
