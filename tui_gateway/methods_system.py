"""System, process, reload, and command-control JSON-RPC handlers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, MutableMapping

from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)

MCP_RELOAD_MAX_PASSES = 3

TUI_HIDDEN: frozenset[str] = frozenset(
    {
        "sethome",
        "set-home",
        "commands",
        "approve",
        "deny",
    }
)

TUI_EXTRA: tuple[tuple[str, str, str], ...] = (
    ("/density", "Toggle compact display mode", "TUI"),
    ("/logs", "Show recent gateway log lines", "TUI"),
    (
        "/mouse",
        "Set mouse tracking preset [on|off|toggle|wheel|buttons|all]",
        "TUI",
    ),
    ("/sessions", "Switch between live TUI sessions", "TUI"),
)


@dataclass(frozen=True)
class SystemServices:
    sessions: MutableMapping[str, dict]
    sessions_lock: threading.RLock
    get_mcp_reload_lock: Callable[[], threading.Lock]
    get_mcp_reload_gen: Callable[[], int]
    set_mcp_reload_gen: Callable[[int], None]
    get_mcp_reload_loaded_rev: Callable[[], str]
    set_mcp_reload_loaded_rev: Callable[[str], None]
    compute_mcp_rev: Callable[[], str]
    skill_usage_lookup: Callable[[], tuple[Callable[[str], int], Callable[[str], str]]]
    sess: Callable[[dict, Any], tuple[dict | None, dict | None]]
    load_cfg: Callable[[], dict]
    load_enabled_toolsets: Callable[[], list[str] | None]
    session_uses_compute_host: Callable[[dict], bool]
    get_compute_host_supervisor: Callable[[], Any]
    emit: Callable[[str, str, dict], None]
    session_info: Callable[[Any, dict | None], dict]
    call_method: Callable[[str, Any, dict], dict]
    apply_model_switch: Callable[..., Any]
    resolve_session_platform: Callable[[], str]
    skill_scaffold_projection: Callable[[str], str]
    load_tool_progress_mode: Callable[[], str]
    get_db: Callable[[], Any]
    db_unavailable_error: Callable[..., dict]
    compress_session_history: Callable[..., tuple[int, Any]]
    sync_session_key_after_compress: Callable[[str, dict], None]
    send_compute_host_control: Callable[..., dict]
    apply_compute_host_metadata_mirror: Callable[[dict, dict | None], None]


def _ok(rid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}


def _session_processes(session: dict) -> list:
    """Background processes owned by this session (registry session_key match)."""
    from tools.process_registry import process_registry

    key = str(session.get("session_key") or "")
    owned = []
    for entry in process_registry.list_sessions():
        proc = process_registry.get(entry["session_id"])
        if proc is None or str(getattr(proc, "session_key", "") or "") != key:
            continue
        entry["output_tail"] = (proc.output_buffer or "")[-4000:]
        owned.append(entry)
    return owned


def _compute_mcp_rev(services: SystemServices) -> str:
    try:
        cfg = services.load_cfg()
        rev_src = json.dumps(
            {
                "mcp": cfg.get("mcp"),
                "mcp_servers": cfg.get("mcp_servers"),
                "tools": cfg.get("tools"),
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha1(rev_src.encode()).hexdigest()[:12]
    except Exception:
        return ""


def _finish_reload(
    rid,
    params: dict,
    *,
    coalesced: bool,
    services: SystemServices,
) -> dict:
    if bool(params.get("always", False)):
        try:
            from cli import save_config_value as save_cfg

            save_cfg("approvals.mcp_reload_confirm", False)
        except Exception as exc:
            logger.warning("Failed to persist mcp_reload_confirm=false: %s", exc)

    payload = {
        "status": "reloaded",
        "loaded_rev": services.get_mcp_reload_loaded_rev(),
    }
    if coalesced:
        payload["coalesced"] = True

    return _ok(rid, payload)


def _skill_usage_lookup():
    try:
        from tools.skill_usage import (
            _read_bundled_manifest_names,
            _read_hub_installed_names,
            activity_count,
            load_usage,
        )

        records = load_usage()
        bundled = _read_bundled_manifest_names()
        hub = _read_hub_installed_names()
    except Exception as e:
        logger.debug("skill usage lookup unavailable: %s", e)
        return (lambda _name: 0), (lambda _name: "local")

    def usage(name: str) -> int:
        try:
            return activity_count(records.get(name) or {})
        except Exception:
            return 0

    def origin(name: str) -> str:
        if name in hub:
            return "hub"
        if name in bundled:
            return "bundled"
        return "local"

    return usage, origin


def _cli_exec_blocked(argv: list[str]) -> str | None:
    if not argv:
        return "bare `hermes` is interactive — use `/hermes chat -q …` or run `hermes` in another terminal"
    a0 = argv[0].lower()
    if a0 == "setup":
        return "`hermes setup` needs a full terminal — run it outside the TUI"
    if a0 == "gateway":
        return "`hermes gateway` is long-running — run it in another terminal"
    if a0 == "sessions" and len(argv) > 1 and argv[1].lower() == "browse":
        return "`hermes sessions browse` is interactive — use /resume here, or run browse in another terminal"
    if a0 == "config" and len(argv) > 1 and argv[1].lower() == "edit":
        return "`hermes config edit` needs $EDITOR in a real terminal"
    return None


def _resolve_name(name: str) -> str:
    try:
        from hermes_cli.commands import resolve_command

        r = resolve_command(name)
        return r.name if r else name
    except Exception:
        return name


def system_battery(rid, params: dict, services: SystemServices) -> dict:
    try:
        from agent.battery import battery_category, read_battery

        batt = read_battery()
        return _ok(
            rid,
            {
                "available": batt.available,
                "percent": batt.percent,
                "plugged": batt.plugged,
                "category": battery_category(batt),
            },
        )
    except Exception:
        return _ok(
            rid,
            {"available": False, "percent": None, "plugged": None, "category": "dim"},
        )


def process_stop(rid, params: dict, services: SystemServices) -> dict:
    try:
        from tools.process_registry import process_registry

        return _ok(rid, {"killed": process_registry.kill_all()})
    except Exception as e:
        return _err(rid, 5010, str(e))


def process_list(rid, params: dict, services: SystemServices) -> dict:
    session, err = services.sess(params, rid)
    if err:
        return err
    try:
        return _ok(rid, {"processes": _session_processes(session or {})})
    except Exception as e:
        return _err(rid, 5010, str(e))


def process_kill(rid, params: dict, services: SystemServices) -> dict:
    session, err = services.sess(params, rid)
    if err:
        return err
    proc_id = str(params.get("process_id") or "")
    if not proc_id:
        return _err(rid, 4012, "process_id required")
    try:
        from tools.process_registry import process_registry

        proc = process_registry.get(proc_id)
        if proc is None or str(getattr(proc, "session_key", "") or "") != str(
            (session or {}).get("session_key") or ""
        ):
            return _err(rid, 4044, f"no such process: {proc_id}")
        return _ok(rid, process_registry.kill_process(proc_id))
    except Exception as e:
        return _err(rid, 5010, str(e))


def reload_mcp(rid, params: dict, services: SystemServices) -> dict:
    session = services.sessions.get(params.get("session_id", ""))
    try:
        user_confirm = bool(params.get("confirm", False))
        if not user_confirm:
            try:
                from hermes_cli.config import load_config

                cfg = load_config()
                approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
                confirm_required = True
                if isinstance(approvals, dict):
                    confirm_required = bool(approvals.get("mcp_reload_confirm", True))
            except Exception:
                confirm_required = True
            if confirm_required:
                return _ok(
                    rid,
                    {
                        "status": "confirm_required",
                        "message": (
                            "⚠️  /reload-mcp invalidates the prompt cache (next "
                            "message re-sends full input tokens). Reply `/reload-mcp "
                            "now` to proceed, or `/reload-mcp always` to proceed and "
                            "silence this prompt permanently."
                        ),
                    },
                )

        if session and services.session_uses_compute_host(session):
            try:
                ack = services.get_compute_host_supervisor().reload_mcp(
                    str(params.get("session_id") or ""),
                    request_id=f"reload-mcp-{rid}",
                )
            except Exception as exc:
                return _err(rid, 5019, f"compute-host reload_mcp failed: {exc}")
            return _ok(
                rid, {"status": "reloaded", "turn_isolation": True, "host_ack": ack}
            )

        from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools

        def refresh_session_agent() -> None:
            if not session:
                return
            agent = session["agent"]
            try:
                from tools.mcp_tool import refresh_agent_mcp_tools

                refresh_agent_mcp_tools(
                    agent,
                    enabled_override=services.load_enabled_toolsets(),
                    quiet_mode=True,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to refresh cached agent tools after /reload-mcp: %s",
                    exc,
                )
            services.emit(
                "session.info",
                params.get("session_id", ""),
                services.session_info(agent, session),
            )

        req_rev = str(params.get("rev") or "")

        def do_full_reload() -> None:
            loaded = services.compute_mcp_rev()
            for _ in range(MCP_RELOAD_MAX_PASSES):
                shutdown_mcp_servers()
                discover_mcp_tools()
                after = services.compute_mcp_rev()
                if after == loaded:
                    break
                loaded = after

            refresh_session_agent()
            services.set_mcp_reload_loaded_rev(loaded)
            services.set_mcp_reload_gen(services.get_mcp_reload_gen() + 1)

        reload_lock = services.get_mcp_reload_lock()
        if reload_lock.acquire(blocking=False):
            try:
                do_full_reload()
            finally:
                reload_lock.release()

            return _finish_reload(rid, params, coalesced=False, services=services)

        gen_before = services.get_mcp_reload_gen()

        with reload_lock:
            leader_completed = services.get_mcp_reload_gen() > gen_before
            rev_satisfied = (
                not req_rev or req_rev == services.get_mcp_reload_loaded_rev()
            )

            if leader_completed and rev_satisfied:
                refresh_session_agent()
                coalesced = True
            else:
                do_full_reload()
                coalesced = False

        return _finish_reload(rid, params, coalesced=coalesced, services=services)
    except Exception as e:
        return _err(rid, 5015, str(e))


def reload_env(rid, params: dict, services: SystemServices) -> dict:
    try:
        from hermes_cli.config import reload_env as reload_dotenv

        count = reload_dotenv()
        return _ok(rid, {"updated": int(count)})
    except Exception as e:
        return _err(rid, 5015, str(e))


def commands_catalog(rid, params: dict, services: SystemServices) -> dict:
    try:
        from hermes_cli.commands import (
            COMMAND_REGISTRY,
            SUBCOMMANDS,
            _build_description,
        )

        all_pairs: list[list[str]] = []
        canon: dict[str, str] = {}
        categories: list[dict] = []
        cat_map: dict[str, list[list[str]]] = {}
        cat_order: list[str] = []

        for cmd in COMMAND_REGISTRY:
            if cmd.name in TUI_HIDDEN or cmd.gateway_only:
                continue

            c = f"/{cmd.name}"
            canon[c.lower()] = c
            for a in cmd.aliases:
                canon[f"/{a}".lower()] = c

            desc = _build_description(cmd)
            all_pairs.append([c, desc])

            cat = cmd.category
            if cat not in cat_map:
                cat_map[cat] = []
                cat_order.append(cat)
            cat_map[cat].append([c, desc])

        for name, desc, cat in TUI_EXTRA:
            if name.lower() in canon:
                continue
            canon[name.lower()] = name
            all_pairs.append([name, desc])
            if cat not in cat_map:
                cat_map[cat] = []
                cat_order.append(cat)
            cat_map[cat].append([name, desc])

        warning = ""
        try:
            qcmds = services.load_cfg().get("quick_commands", {}) or {}
            if isinstance(qcmds, dict) and qcmds:
                bucket = "User commands"
                if bucket not in cat_map:
                    cat_map[bucket] = []
                    cat_order.append(bucket)
                for qname, qc in sorted(qcmds.items()):
                    if not isinstance(qc, dict):
                        continue
                    key = f"/{qname}"
                    canon[key.lower()] = key
                    qtype = qc.get("type", "")
                    if qtype == "exec":
                        default_desc = f"exec: {qc.get('command', '')}"
                    elif qtype == "alias":
                        default_desc = f"alias → {qc.get('target', '')}"
                    else:
                        default_desc = qtype or "quick command"
                    qdesc = str(qc.get("description") or default_desc)
                    qdesc = qdesc[:120] + ("…" if len(qdesc) > 120 else "")
                    all_pairs.append([key, qdesc])
                    cat_map[bucket].append([key, qdesc])
        except Exception as e:
            if not warning:
                warning = f"quick_commands discovery unavailable: {e}"

        skill_count = 0
        skills: dict[str, dict] = {}
        try:
            from agent.skill_commands import scan_skill_commands

            usage, origin_of = services.skill_usage_lookup()

            for k, info in sorted(scan_skill_commands().items()):
                d = str(info.get("description", "Skill"))
                all_pairs.append([k, d[:120] + ("…" if len(d) > 120 else "")])
                name = str(info.get("name") or k.lstrip("/"))
                skills[k] = {"usage": usage(name), "origin": origin_of(name)}
                skill_count += 1
        except Exception as e:
            warning = f"skill discovery unavailable: {e}"

        for cat in cat_order:
            categories.append({"name": cat, "pairs": cat_map[cat]})

        sub = {k: v[:] for k, v in SUBCOMMANDS.items()}
        return _ok(
            rid,
            {
                "pairs": all_pairs,
                "sub": sub,
                "canon": canon,
                "categories": categories,
                "skills": skills,
                "skill_count": skill_count,
                "warning": warning,
            },
        )
    except Exception as e:
        return _err(rid, 5020, str(e))


def cli_exec(rid, params: dict, services: SystemServices) -> dict:
    argv = params.get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        return _err(rid, 4003, "argv must be list[str]")
    hint = _cli_exec_blocked(argv)
    if hint:
        return _ok(rid, {"blocked": True, "hint": hint, "code": -1, "output": ""})
    try:
        from hermes_cli._subprocess_compat import windows_hide_flags

        r = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", *argv],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=min(int(params.get("timeout", 240)), 600),
            cwd=os.getcwd(),
            env=hermes_subprocess_env(inherit_credentials=True),
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        parts = [r.stdout or "", r.stderr or ""]
        out = "\n".join(p for p in parts if p).strip() or "(no output)"
        return _ok(rid, {"blocked": False, "code": r.returncode, "output": out[:48_000]})
    except subprocess.TimeoutExpired:
        return _err(rid, 5016, "cli.exec: timeout")
    except Exception as e:
        return _err(rid, 5017, str(e))


def command_resolve(rid, params: dict, services: SystemServices) -> dict:
    try:
        from hermes_cli.commands import resolve_command

        r = resolve_command(params.get("name", ""))
        if r:
            return _ok(
                rid,
                {
                    "canonical": r.name,
                    "description": r.description,
                    "category": r.category,
                },
            )
        return _err(rid, 4011, f"unknown command: {params.get('name')}")
    except Exception as e:
        return _err(rid, 5012, str(e))


def command_dispatch(rid, params: dict, services: SystemServices) -> dict:
    name, arg = params.get("name", "").lstrip("/"), params.get("arg", "")
    resolved = _resolve_name(name)
    if resolved != name:
        name = resolved
    session = services.sessions.get(params.get("session_id", ""))

    qcmds = services.load_cfg().get("quick_commands", {})
    if name in qcmds:
        qc = qcmds[name]
        if qc.get("type") == "exec":
            from tools.environments.local import build_subprocess_env

            sanitized_env = build_subprocess_env()
            from hermes_cli._subprocess_compat import windows_hide_flags

            r = subprocess.run(
                qc.get("command", ""),
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                stdin=subprocess.DEVNULL,
                env=sanitized_env,
                creationflags=windows_hide_flags(),
            )
            output = (
                (r.stdout or "")
                + ("\n" if r.stdout and r.stderr else "")
                + (r.stderr or "")
            ).strip()[:4000]
            if output:
                from agent.redact import redact_sensitive_text

                output = redact_sensitive_text(output)
            if r.returncode != 0:
                return _err(
                    rid,
                    4018,
                    output or f"quick command failed with exit code {r.returncode}",
                )
            return _ok(rid, {"type": "exec", "output": output})
        if qc.get("type") == "alias":
            return _ok(rid, {"type": "alias", "target": qc.get("target", "")})

    try:
        from hermes_cli.plugins import (
            get_plugin_command_handler,
            resolve_plugin_command_result,
        )

        handler = get_plugin_command_handler(name)
        if handler:
            result = resolve_plugin_command_result(handler(arg))
            return _ok(rid, {"type": "plugin", "output": str(result or "")})
    except Exception:
        pass

    try:
        from agent.skill_bundles import (
            build_bundle_invocation_message,
            get_skill_bundles,
            resolve_bundle_command_key,
        )
        from hermes_cli.commands import resolve_command

        bundle_key = (
            resolve_bundle_command_key(name)
            if resolve_command(name) is None
            else None
        )
    except Exception:
        bundle_key = None

    if bundle_key is not None:
        try:
            bundle_result = build_bundle_invocation_message(
                bundle_key,
                arg,
                task_id=session.get("session_key", "") if session else "",
                platform=services.resolve_session_platform(),
            )
        except Exception as exc:
            return _err(rid, 4018, f"bundle dispatch failed: {exc}")

        if not bundle_result:
            return _err(rid, 4018, f"failed to load bundle: {bundle_key}")

        msg, loaded_names, missing = bundle_result
        bundle_info = get_skill_bundles().get(bundle_key, {})
        bundle_name = bundle_info.get("name", bundle_key.lstrip("/"))
        notice = f"⚡ Loading bundle: {bundle_name} ({len(loaded_names)} skills)"
        if missing:
            notice += f"\nSkipped missing skills: {', '.join(missing)}"
        return _ok(
            rid,
            {
                "type": "send",
                "message": msg,
                "notice": notice,
                "display": services.skill_scaffold_projection(msg),
            },
        )

    try:
        from agent.skill_commands import (
            build_skill_invocation_message,
            scan_skill_commands,
        )

        cmds = scan_skill_commands()
        key = f"/{name}"
        if key in cmds:
            msg = build_skill_invocation_message(
                key, arg, task_id=session.get("session_key", "") if session else ""
            )
            if msg:
                return _ok(
                    rid,
                    {
                        "type": "skill",
                        "message": msg,
                        "name": cmds[key].get("name", name),
                        "display": services.skill_scaffold_projection(msg),
                    },
                )
    except Exception:
        pass

    if name in {"queue", "q"}:
        if not arg:
            return _err(rid, 4004, "usage: /queue <prompt>")
        return _ok(rid, {"type": "send", "message": arg})

    if name == "learn":
        from agent.learn_prompt import build_learn_prompt

        return _ok(rid, {"type": "send", "message": build_learn_prompt(arg)})
    if name == "init":
        from hermes_cli.init_command import build_init_prompt_for_cwd

        return _ok(rid, {"type": "send", "message": build_init_prompt_for_cwd(extra=arg)})
    if name == "moa":
        try:
            from hermes_cli.moa_config import moa_usage, normalize_moa_config

            if not arg:
                return _err(rid, 4004, moa_usage())
            if not session:
                return _err(rid, 4001, "no active session")
            sid = params.get("session_id", "")
            moa_cfg = normalize_moa_config(services.load_cfg().get("moa") or {})
            preset = moa_cfg["default_preset"]
            agent = session.get("agent")
            session["moa_one_shot_restore"] = {
                "override": session.get("model_override"),
                "model": getattr(agent, "model", None) if agent else None,
                "provider": getattr(agent, "provider", None) if agent else None,
            }
            if agent is not None:
                try:
                    services.apply_model_switch(
                        sid,
                        session,
                        f"{preset} --provider moa",
                        confirm_expensive_model=False,
                        pin_session_override=True,
                        persist_override=False,
                    )
                except Exception as exc:
                    session.pop("moa_one_shot_restore", None)
                    return _err(rid, 5030, f"moa unavailable: {exc}")
            else:
                session["model_override"] = {
                    "provider": "moa",
                    "model": preset,
                    "base_url": "moa://local",
                    "api_key": "moa-virtual-provider",
                    "api_mode": "chat_completions",
                }
            return _ok(
                rid,
                {
                    "type": "send",
                    "notice": f"MoA one-shot queued with preset {preset}; previous model will be restored after this turn.",
                    "message": arg,
                },
            )
        except Exception as exc:
            return _err(rid, 5030, f"moa unavailable: {exc}")

    if name == "focus":
        from hermes_cli.focus_view import (
            format_focus_status,
            format_focus_toggle_message,
            resolve_focus_arg,
        )

        display_focus = services.load_cfg().get("display")
        d_focus: dict = display_focus if isinstance(display_focus, dict) else {}
        cur_focus = bool(d_focus.get("focus_view", False))
        action, target = resolve_focus_arg(arg, cur_focus)
        if action == "usage":
            return _err(rid, 4004, "usage: /focus [on|off|status]")
        if action == "status":
            saved = d_focus.get("focus_saved_tool_progress") or services.load_tool_progress_mode()
            return _ok(
                rid,
                {"type": "exec", "output": format_focus_status(cur_focus, saved)},
            )
        res = services.call_method(
            "config.set",
            rid,
            {
                "key": "focus",
                "value": "on" if target else "off",
                "session_id": params.get("session_id", ""),
            },
        )
        if "error" in res:
            return res
        payload = res.get("result") or {}
        return _ok(
            rid,
            {
                "type": "exec",
                "output": format_focus_toggle_message(
                    bool(target), payload.get("tool_progress") or "all"
                ),
            },
        )

    if name == "retry":
        if not session:
            return _err(rid, 4001, "no active session to retry")
        if session.get("running"):
            return _err(
                rid, 4009, "session busy — /interrupt the current turn before /retry"
            )
        history = session.get("history", [])
        if not history:
            return _err(rid, 4018, "no previous user message to retry")
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            if msg.get("role") == "user" and not msg.get("display_kind"):
                last_user_idx = i
                break
        if last_user_idx is None:
            return _err(rid, 4018, "no previous user message to retry")
        content = history[last_user_idx].get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if not content:
            return _err(rid, 4018, "last user message is empty")
        with session["history_lock"]:
            session["history"] = history[:last_user_idx]
            session["history_version"] = int(session.get("history_version", 0)) + 1
        return _ok(rid, {"type": "send", "message": content})

    if name == "steer":
        if not arg:
            return _err(rid, 4004, "usage: /steer <prompt>")
        agent = session.get("agent") if session else None
        if agent and hasattr(agent, "steer"):
            try:
                accepted = agent.steer(arg)
                if accepted:
                    return _ok(
                        rid,
                        {
                            "type": "exec",
                            "output": f"⏩ Steer queued — arrives after the next tool call: {arg[:80]}{'...' if len(arg) > 80 else ''}",
                        },
                    )
            except Exception:
                pass
        return _ok(rid, {"type": "send", "message": arg})

    if name == "goal":
        if not session:
            return _err(rid, 4001, "no active session")
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            return _err(rid, 5030, f"goals unavailable: {exc}")

        sid_key = session.get("session_key") or ""
        if not sid_key:
            return _err(rid, 4001, "no session key")

        try:
            goals_cfg = services.load_cfg().get("goals") or {}
            max_turns = int(goals_cfg.get("max_turns", 20) or 20)
        except Exception:
            max_turns = 20
        mgr = GoalManager(session_id=sid_key, default_max_turns=max_turns)

        lower = arg.strip().lower()
        if not arg.strip() or lower == "status":
            return _ok(rid, {"type": "exec", "output": mgr.status_line()})
        if lower == "pause":
            state = mgr.pause(reason="user-paused")
            out = "No goal set." if state is None else f"⏸ Goal paused: {state.goal}"
            return _ok(rid, {"type": "exec", "output": out})
        if lower == "resume":
            state = mgr.resume()
            if state is None:
                return _ok(rid, {"type": "exec", "output": "No goal to resume."})
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": (
                        f"▶ Goal resumed: {state.goal}\n"
                        "Send any message to continue, or wait — I'll take the next step on the next turn."
                    ),
                },
            )
        if lower in {"clear", "stop", "done"}:
            had = mgr.has_goal()
            mgr.clear()
            return _ok(
                rid,
                {"type": "exec", "output": "✓ Goal cleared." if had else "No active goal."},
            )

        try:
            state = mgr.set(arg)
        except ValueError as exc:
            return _err(rid, 4004, f"invalid goal: {exc}")

        notice = (
            f"⊙ Goal set ({state.max_turns}-turn budget): {state.goal}\n"
            "I'll keep working until the goal is done, you pause/clear it, or the budget is exhausted.\n"
            "Controls: /goal status · /goal pause · /goal resume · /goal clear"
        )
        return _ok(rid, {"type": "send", "notice": notice, "message": state.goal})

    if name == "undo":
        if not session:
            return _err(rid, 4001, "no active session to undo")
        if session.get("running"):
            return _err(
                rid, 4009, "session busy — /interrupt the current turn before /undo"
            )
        db = services.get_db()
        if db is None:
            return services.db_unavailable_error(rid, code=5008)
        session_key = session.get("session_key", "")
        if not session_key:
            return _err(rid, 4001, "no session key for undo")
        n = 1
        arg_str = (arg or "").strip()
        if arg_str:
            try:
                n = int(arg_str.split()[0])
            except (ValueError, IndexError):
                return _err(rid, 4004, f"undo: invalid count {arg_str!r} — use /undo or /undo N")
        if n < 1:
            n = 1
        try:
            recents = db.list_recent_user_messages(session_key, limit=max(n, 10))
        except Exception as e:
            return _err(rid, 5008, f"undo: failed to load history: {e}")
        if not recents:
            return _err(rid, 4018, "no user messages to undo")
        target_idx = min(n - 1, len(recents) - 1)
        target_id = recents[target_idx]["id"]
        try:
            result = db.rewind_to_message(session_key, target_id)
        except ValueError as e:
            return _err(rid, 4004, f"undo: {e}")
        except Exception as e:
            return _err(rid, 5008, f"undo: {e}")
        try:
            active = db.get_messages_as_conversation(session_key, repair_alternation=True)
        except Exception:
            active = []
        with session["history_lock"]:
            session["history"] = list(active)
            session["history_version"] = int(session.get("history_version", 0)) + 1
        agent = session.get("agent")
        if agent is not None:
            mm = getattr(agent, "_memory_manager", None)
            if mm is not None:
                try:
                    mm.on_session_switch(
                        session_key,
                        parent_session_id="",
                        reset=False,
                        rewound=True,
                    )
                except Exception:
                    pass
            if hasattr(agent, "_invalidate_system_prompt"):
                try:
                    agent._invalidate_system_prompt()
                except Exception:
                    pass
            if hasattr(agent, "_last_flushed_db_idx"):
                try:
                    agent._last_flushed_db_idx = len(active)
                except Exception:
                    pass
        target_msg = result.get("target_message") or {}
        target_text = target_msg.get("content") or ""
        if isinstance(target_text, list):
            parts = [
                p.get("text", "") for p in target_text
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            target_text = "\n".join(t for t in parts if t)
        if not isinstance(target_text, str):
            target_text = ""
        rewound_count = result.get("rewound_count", 0)
        turns_undone = target_idx + 1
        turn_word = "turn" if turns_undone == 1 else "turns"
        notice = (
            f"↶ Undid {turns_undone} {turn_word} ({rewound_count} message(s)). "
            "Edit and resubmit, or send a new message."
        )
        return _ok(rid, {"type": "prefill", "message": target_text, "notice": notice})

    if name in {"snapshot", "snap"}:
        subcommand = arg.split(maxsplit=1)[0].lower() if arg else ""
        if subcommand in {"restore", "rewind"}:
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": (
                        "/snapshot restore is blocked in the TUI because it changes "
                        "config/state on disk while the live agent has cached settings. "
                        "Run it in the classic CLI, then restart the TUI."
                    ),
                },
            )

    if name in {"compress", "compact"}:
        if not session:
            return _err(rid, 4001, "no active session to compress")
        if session.get("running"):
            return _err(
                rid, 4009, "session busy — /interrupt the current turn before /compress"
            )
        from agent.conversation_compression import (
            finalize_context_engine_compression_notification,
        )

        sid = params.get("session_id", "")
        if services.session_uses_compute_host(session):
            command = f"/{name}" + (f" {arg}" if arg else "")
            try:
                ack = services.send_compute_host_control(
                    sid,
                    route_name="slash.compress",
                    command=command,
                    wait=True,
                )
            except Exception as exc:
                return _err(rid, 5019, f"compute-host slash.compress failed: {exc}")
            if ack.get("type") in {"control.error", "error"}:
                return _err(
                    rid,
                    4009,
                    str(ack.get("message") or "compute-host slash.compress failed"),
                )
            services.apply_compute_host_metadata_mirror(session, ack)
            return _ok(rid, {"type": "exec", "output": str(ack.get("output") or "")})
        try:
            from agent.manual_compression_feedback import summarize_manual_compression
            from agent.model_metadata import estimate_request_tokens_rough

            with session["history_lock"]:
                before_messages = list(session.get("history", []))
                history_version = int(session.get("history_version", 0))
            before_count = len(before_messages)
            agent = session["agent"]
            sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
            tools = getattr(agent, "tools", None) or None
            before_tokens = (
                estimate_request_tokens_rough(
                    before_messages, system_prompt=sys_prompt, tools=tools
                )
                if before_count
                else 0
            )
            removed, usage = services.compress_session_history(
                session,
                arg.strip() or None,
                approx_tokens=before_tokens,
                before_messages=before_messages,
                history_version=history_version,
            )
            with session["history_lock"]:
                after_messages = list(session.get("history", []))
            sys_prompt_after = getattr(agent, "_cached_system_prompt", "") or sys_prompt
            tools_after = getattr(agent, "tools", None) or tools
            after_tokens = (
                estimate_request_tokens_rough(
                    after_messages,
                    system_prompt=sys_prompt_after,
                    tools=tools_after,
                )
                if after_messages
                else 0
            )
            services.sync_session_key_after_compress(sid, session)
            summary = summarize_manual_compression(
                before_messages,
                after_messages,
                before_tokens,
                after_tokens,
                compression_state=getattr(agent, "context_compressor", None),
            )
            services.emit("session.info", sid, services.session_info(session.get("agent"), session))
            finalize_context_engine_compression_notification(agent, committed=True)
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": "\n".join(
                        filter(
                            None,
                            [
                                summary["headline"],
                                summary["token_line"],
                                summary.get("note"),
                            ],
                        )
                    ),
                },
            )
        except Exception as e:
            if e.__class__.__name__ == "CompressionLockHeld":
                from agent.manual_compression_feedback import (
                    describe_compression_lock_skip,
                )

                return _ok(
                    rid,
                    {
                        "type": "exec",
                        "output": describe_compression_lock_skip(getattr(e, "holder", "")),
                    },
                )
            finalize_context_engine_compression_notification(session["agent"], committed=False)
            return _err(rid, 5009, f"compress failed: {e}")

    return _err(rid, 4018, f"not a quick/plugin/bundle/skill command: {name}")


def register(
    methods: dict[str, Callable[[Any, dict], dict]],
    *,
    services: SystemServices,
) -> None:
    def bind(handler: Callable[[Any, dict, SystemServices], dict]):
        return lambda rid, params: handler(rid, params, services)

    methods["system.battery"] = bind(system_battery)
    methods["process.stop"] = bind(process_stop)
    methods["process.list"] = bind(process_list)
    methods["process.kill"] = bind(process_kill)
    methods["reload.mcp"] = bind(reload_mcp)
    methods["reload.env"] = bind(reload_env)
    methods["commands.catalog"] = bind(commands_catalog)
    methods["cli.exec"] = bind(cli_exec)
    methods["command.resolve"] = bind(command_resolve)
    methods["command.dispatch"] = bind(command_dispatch)
