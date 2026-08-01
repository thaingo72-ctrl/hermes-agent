"""Config / projects / setup JSON-RPC handlers for the TUI gateway."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping

from hermes_constants import DEFAULT_INDICATOR_STYLE, INDICATOR_STYLES
from tui_gateway import git_probe

CONFIG_METHODS = (
    "config.get",
    "config.set",
    "projects.discover_repos",
    "projects.record_repos",
    "projects.tree",
    "projects.project_sessions",
    "projects.list",
    "projects.get",
    "projects.create",
    "projects.update",
    "projects.add_folder",
    "projects.remove_folder",
    "projects.set_primary",
    "projects.archive",
    "projects.delete",
    "projects.set_active",
    "projects.for_cwd",
    "setup.status",
    "setup.runtime_check",
)

REGISTERED_METHODS: dict[str, Callable[[Any, dict], dict]] = {}


@dataclass(frozen=True)
class ConfigServices:
    ok: Callable[[Any, dict], dict]
    err: Callable[[Any, int, str], dict]
    hermes_home: Path
    sessions: dict[str, dict]
    sessions_lock: RLock
    detail_modes: frozenset[str]
    detail_section_names: tuple[str, ...]
    load_cfg: Callable[[], dict]
    load_cfg_raw: Callable[[], dict]
    save_cfg: Callable[[dict], None]
    write_config_key: Callable[[str, Any], None]
    compute_mcp_rev: Callable[[], str]
    get_db: Callable[[], Any]
    completion_cwd: Callable[[dict | None], str]
    git_branch_for_cwd: Callable[[str], str]
    git_common_repo_root_for_cwd: Callable[[str], str]
    resolve_cwd_git: Callable[[str], Any]
    resolve_model: Callable[[], str]
    load_service_tier: Callable[[], str | None]
    load_busy_input_mode: Callable[[], str]
    load_approval_mode: Callable[[], str]
    load_tool_progress_mode: Callable[[], str]
    coerce_statusbar: Callable[[Any], str]
    display_mouse_tracking: Callable[[dict], str]
    apply_model_switch: Callable[..., dict]
    start_agent_build: Callable[[str, dict], None]
    wait_agent: Callable[[dict, Any], dict | None]
    persist_live_session_runtime: Callable[[dict | None], None]
    session_info: Callable[..., dict]
    emit: Callable[[str, str, dict], None]
    broadcast_global_event: Callable[[str, dict | None], None]
    resolve_skin: Callable[[], dict]
    note_skin_broadcast: Callable[[], None]
    validate_personality: Callable[[str, dict | None], tuple[str, str]]
    apply_personality_to_session: Callable[[str, dict | None, str, str], tuple[bool, dict | None]]


def _ok(services: ConfigServices, rid: Any, result: dict) -> dict:
    return services.ok(rid, result)


def _err(services: ConfigServices, rid: Any, code: int, msg: str) -> dict:
    return services.err(rid, code, msg)


def _session_for_params(services: ConfigServices, params: dict) -> dict | None:
    with services.sessions_lock:
        return services.sessions.get(params.get("session_id", ""))


def _session_items(services: ConfigServices) -> list[tuple[str, dict]]:
    with services.sessions_lock:
        return list(services.sessions.items())


def _config_get(rid: Any, params: dict, services: ConfigServices) -> dict:
    key = params.get("key", "")
    if key == "provider":
        try:
            from hermes_cli.models import list_available_providers, normalize_provider

            model = services.resolve_model()
            parts = model.split("/", 1)
            return _ok(
                services,
                rid,
                {
                    "model": model,
                    "provider": normalize_provider(parts[0]) if len(parts) > 1 else "unknown",
                    "providers": list_available_providers(),
                },
            )
        except Exception as e:
            return _err(services, rid, 5013, str(e))
    if key == "profile":
        from hermes_constants import display_hermes_home

        return _ok(services, rid, {"home": str(services.hermes_home), "display": display_hermes_home()})
    if key == "project":
        cfg_terminal = services.load_cfg().get("terminal") or {}
        raw = str(params.get("cwd", "") or cfg_terminal.get("cwd", "") or "").strip()
        cwd = services.completion_cwd({"cwd": raw} if raw else {})
        return _ok(services, rid, {"cwd": cwd, "branch": services.git_branch_for_cwd(cwd)})
    if key == "full":
        return _ok(services, rid, {"config": services.load_cfg()})
    if key == "prompt":
        return _ok(services, rid, {"prompt": services.load_cfg().get("custom_prompt", "")})
    if key == "skin":
        return _ok(services, rid, {"value": (services.load_cfg().get("display") or {}).get("skin", "default")})
    if key == "indicator":
        raw = (services.load_cfg().get("display") or {}).get("tui_status_indicator", "")
        norm = str(raw).strip().lower()
        return _ok(services, rid, {"value": norm if norm in INDICATOR_STYLES else DEFAULT_INDICATOR_STYLE})
    if key == "personality":
        return _ok(services, rid, {"value": (services.load_cfg().get("display") or {}).get("personality") or "none"})
    if key == "reasoning":
        cfg = services.load_cfg()
        session = _session_for_params(services, params)
        reasoning_config = None
        if session is not None:
            if isinstance(session.get("create_reasoning_override"), dict):
                reasoning_config = session.get("create_reasoning_override")
            else:
                agent = session.get("agent")
                agent_reasoning = getattr(agent, "reasoning_config", None)
                if isinstance(agent_reasoning, dict):
                    reasoning_config = agent_reasoning

        if isinstance(reasoning_config, dict):
            effort = "none" if reasoning_config.get("enabled") is False else str(reasoning_config.get("effort") or "medium")
        else:
            raw_effort = (cfg.get("agent") or {}).get("reasoning_effort", "")
            effort = "none" if raw_effort is False else str(raw_effort or "medium")
        display = "show" if bool((cfg.get("display") or {}).get("show_reasoning", True)) else "hide"
        return _ok(services, rid, {"value": effort, "display": display})
    if key == "fast":
        session = _session_for_params(services, params)
        tier = None
        if session is not None:
            agent = session.get("agent")
            if agent is not None:
                tier = getattr(agent, "service_tier", None)
            elif session.get("create_service_tier_override") is not None:
                tier = session["create_service_tier_override"]
        if tier is None:
            tier = services.load_service_tier()
        return _ok(services, rid, {"value": "fast" if tier == "priority" else "normal"})
    if key == "busy":
        return _ok(services, rid, {"value": services.load_busy_input_mode()})
    if key in {"approval_mode", "approvals.mode"}:
        try:
            return _ok(services, rid, {"value": services.load_approval_mode()})
        except Exception as e:
            return _err(services, rid, 5001, str(e))
    if key == "details_mode":
        raw = str((services.load_cfg().get("display") or {}).get("details_mode", "collapsed") or "collapsed").strip().lower()
        return _ok(services, rid, {"value": raw if raw in services.detail_modes else "collapsed"})
    if key == "thinking_mode":
        allowed = frozenset({"collapsed", "truncated", "full"})
        cfg = services.load_cfg()
        raw = str((cfg.get("display") or {}).get("thinking_mode", "") or "").strip().lower()
        if raw in allowed:
            value = raw
        else:
            dm = str((cfg.get("display") or {}).get("details_mode", "collapsed") or "collapsed").strip().lower()
            value = "full" if dm == "expanded" else "collapsed"
        return _ok(services, rid, {"value": value})
    if key == "density":
        on = bool((services.load_cfg().get("display") or {}).get("tui_compact", False))
        return _ok(services, rid, {"value": "on" if on else "off"})
    if key == "theme":
        display = services.load_cfg().get("display")
        raw = str(display.get("tui_theme", "auto") if isinstance(display, dict) else "auto").strip().lower()
        return _ok(services, rid, {"value": raw if raw in {"auto", "light", "dark"} else "auto"})
    if key == "statusbar":
        display = services.load_cfg().get("display")
        raw = display.get("tui_statusbar", "top") if isinstance(display, dict) else "top"
        return _ok(services, rid, {"value": services.coerce_statusbar(raw)})
    if key == "focus":
        display = services.load_cfg().get("display")
        on = bool(display.get("focus_view", False)) if isinstance(display, dict) else False
        return _ok(services, rid, {"value": "on" if on else "off", "tool_progress": services.load_tool_progress_mode()})
    if key == "mouse":
        display = services.load_cfg().get("display")
        return _ok(services, rid, {"value": services.display_mouse_tracking(display)})
    if key == "mtime":
        cfg_path = services.hermes_home / "config.yaml"
        try:
            mtime = cfg_path.stat().st_mtime if cfg_path.exists() else 0
        except Exception:
            return _ok(services, rid, {"mtime": 0})
        return _ok(services, rid, {"mtime": mtime, "mcp_rev": services.compute_mcp_rev()})
    return _err(services, rid, 4002, f"unknown config key: {key}")


def _config_set(rid: Any, params: dict, services: ConfigServices) -> dict:
    key, value = params.get("key", ""), params.get("value", "")
    session = _session_for_params(services, params)

    if key == "model":
        try:
            if not value:
                return _err(services, rid, 4002, "model value required")
            if session:
                from hermes_cli.model_switch import parse_model_switch_args

                if session.get("running"):
                    parsed = parse_model_switch_args(value)
                    try:
                        pending_model = parsed.model_input
                    except Exception:
                        pending_model = str(value)
                    session["pending_model_switch"] = {
                        "raw": value,
                        "confirm_expensive_model": bool(params.get("confirm_expensive_model", False)),
                        "display_model": pending_model,
                        "display_provider": (getattr(parsed, "explicit_provider", "") or "").strip(),
                    }
                    return _ok(
                        services,
                        rid,
                        {
                            "key": key,
                            "value": pending_model,
                            "warning": "",
                            "confirm_required": False,
                            "confirm_message": "",
                            "scope": "session",
                            "deferred": True,
                        },
                    )
                parsed_flags = parse_model_switch_args(value)
                explicit_provider = parsed_flags.explicit_provider
                if session.get("agent") is None and not explicit_provider.strip():
                    session_id = params.get("session_id", "")
                    services.start_agent_build(session_id, session)
                    init_err = services.wait_agent(session, rid)
                    if init_err:
                        return init_err
                    if session.get("agent") is None:
                        return _err(services, rid, 5032, "agent initialization failed")
                result = services.apply_model_switch(
                    params.get("session_id", ""),
                    session,
                    value,
                    confirm_expensive_model=bool(params.get("confirm_expensive_model", False)),
                    parsed_flags=parsed_flags,
                )
            else:
                result = services.apply_model_switch(
                    "",
                    {"agent": None},
                    value,
                    confirm_expensive_model=bool(params.get("confirm_expensive_model", False)),
                )
            return _ok(
                services,
                rid,
                {
                    "key": key,
                    "value": result["value"],
                    "warning": result["warning"],
                    "confirm_required": result.get("confirm_required", False),
                    "confirm_message": result.get("confirm_message", ""),
                    "scope": result.get("scope", "session"),
                },
            )
        except Exception as e:
            return _err(services, rid, 5001, str(e))

    if key == "fast":
        raw = str(value or "").strip().lower()
        agent = session.get("agent") if session else None
        if agent is not None:
            current_fast = getattr(agent, "service_tier", None) == "priority"
        elif session is not None and session.get("create_service_tier_override") is not None:
            current_fast = session["create_service_tier_override"] == "priority"
        else:
            current_fast = services.load_service_tier() == "priority"
        if raw in {"status"}:
            return _ok(services, rid, {"key": key, "value": "fast" if current_fast else "normal"})
        if raw in {"", "toggle"}:
            nv = "normal" if current_fast else "fast"
        elif raw in {"fast", "on"}:
            nv = "fast"
        elif raw in {"normal", "off"}:
            nv = "normal"
        else:
            return _err(services, rid, 4002, f"unknown fast mode: {value}")

        overrides = None
        if nv == "fast":
            from hermes_cli.models import resolve_fast_mode_overrides

            session_override = (session or {}).get("model_override") or {}
            target_model = getattr(agent, "model", None) if agent is not None else (session_override.get("model") if isinstance(session_override, dict) else None) or services.resolve_model()
            if not target_model:
                return _err(services, rid, 4002, "fast mode is not available without a selected model")
            overrides = resolve_fast_mode_overrides(target_model)
            if overrides is None:
                return _err(services, rid, 4002, "fast mode is not available for this model")
        if session is not None:
            session["create_service_tier_override"] = "priority" if nv == "fast" else ""
        else:
            services.write_config_key("agent.service_tier", nv)
        if agent is not None:
            agent.service_tier = "priority" if nv == "fast" else None
            current_overrides = dict(getattr(agent, "request_overrides", {}) or {})
            current_overrides.pop("service_tier", None)
            current_overrides.pop("speed", None)
            if nv == "fast":
                current_overrides.update(overrides)
            agent.request_overrides = current_overrides
            services.persist_live_session_runtime(session)
            services.emit("session.info", params.get("session_id", ""), services.session_info(agent, session))
        return _ok(services, rid, {"key": key, "value": nv})

    if key == "busy":
        raw = str(value or "").strip().lower()
        if raw in {"", "status"}:
            return _ok(services, rid, {"key": key, "value": services.load_busy_input_mode()})
        if raw not in {"queue", "steer", "interrupt"}:
            return _err(services, rid, 4002, f"unknown busy mode: {value}")
        services.write_config_key("display.busy_input_mode", raw)
        return _ok(services, rid, {"key": key, "value": raw})

    if key == "verbose":
        cycle = ["off", "new", "all", "verbose"]
        cur = session.get("tool_progress_mode", services.load_tool_progress_mode()) if session else services.load_tool_progress_mode()
        if value and value != "cycle":
            nv = str(value).strip().lower()
            if nv not in cycle:
                return _err(services, rid, 4002, f"unknown verbose mode: {value}")
        else:
            try:
                idx = cycle.index(cur)
            except ValueError:
                idx = 2
            nv = cycle[(idx + 1) % len(cycle)]
        services.write_config_key("display.tool_progress", nv)
        if session:
            session["tool_progress_mode"] = nv
            agent = session.get("agent")
            if agent is not None:
                agent.verbose_logging = nv == "verbose"
        return _ok(services, rid, {"key": key, "value": nv})

    if key == "focus":
        from hermes_cli.focus_view import FOCUS_TOOL_PROGRESS_MODE, normalize_tool_progress_mode, resolve_focus_arg

        cfg_f = services.load_cfg()
        d_f = cfg_f.get("display") if isinstance(cfg_f.get("display"), dict) else {}
        cur_focus = bool(d_f.get("focus_view", False))
        action, target = resolve_focus_arg(str(value or ""), cur_focus)
        if action == "usage":
            return _err(services, rid, 4002, f"unknown focus value: {value} (use on|off|status)")
        if action == "status" or target is None:
            return _ok(services, rid, {"key": key, "value": "on" if cur_focus else "off", "tool_progress": services.load_tool_progress_mode()})
        if target:
            saved = normalize_tool_progress_mode((d_f.get("focus_saved_tool_progress") or services.load_tool_progress_mode()) if cur_focus else services.load_tool_progress_mode())
            services.write_config_key("display.focus_saved_tool_progress", saved)
            services.write_config_key("display.tool_progress", FOCUS_TOOL_PROGRESS_MODE)
            effective = FOCUS_TOOL_PROGRESS_MODE
        else:
            saved = normalize_tool_progress_mode(d_f.get("focus_saved_tool_progress") or "all")
            services.write_config_key("display.tool_progress", saved)
            effective = saved
        services.write_config_key("display.focus_view", bool(target))
        if session:
            session["focus_view"] = bool(target)
            session["tool_progress_mode"] = effective
            agent_f = session.get("agent")
            if agent_f is not None:
                try:
                    agent_f.tool_progress_mode = effective
                except Exception:
                    pass
        return _ok(services, rid, {"key": key, "value": "on" if target else "off", "tool_progress": effective})

    if key in {"approval_mode", "approvals.mode"}:
        raw = str(value or "").strip().lower()
        if raw not in {"manual", "smart", "off"}:
            return _err(services, rid, 4002, f"unknown approval mode: {value}; pick one of manual|smart|off")
        services.write_config_key("approvals.mode", raw)
        for sid, sess in _session_items(services):
            agent = sess.get("agent")
            if agent is not None:
                services.emit("session.info", sid, services.session_info(agent, sess))
        return _ok(services, rid, {"key": "approvals.mode", "value": raw})

    if key == "yolo":
        return _set_yolo(rid, params, value, session, services)

    if key == "reasoning":
        return _set_reasoning(rid, params, value, session, services)

    if key == "details_mode":
        nv = str(value or "").strip().lower()
        if nv not in services.detail_modes:
            return _err(services, rid, 4002, f"unknown details_mode: {value}")
        cfg = services.load_cfg_raw()
        display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
        sections = display.get("sections") if isinstance(display.get("sections"), dict) else {}
        display["details_mode"] = nv
        for section in services.detail_section_names:
            sections[section] = nv
        display["sections"] = sections
        cfg["display"] = display
        services.save_cfg(cfg)
        return _ok(services, rid, {"key": key, "value": nv})

    if key.startswith("details_mode."):
        section = key.split(".", 1)[1]
        if section not in services.detail_section_names:
            return _err(services, rid, 4002, f"unknown section: {section}")
        cfg = services.load_cfg_raw()
        display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
        sections_cfg = display.get("sections") if isinstance(display.get("sections"), dict) else {}
        nv = str(value or "").strip().lower()
        if not nv:
            sections_cfg.pop(section, None)
            display["sections"] = sections_cfg
            cfg["display"] = display
            services.save_cfg(cfg)
            return _ok(services, rid, {"key": key, "value": ""})
        if nv not in services.detail_modes:
            return _err(services, rid, 4002, f"unknown details_mode: {value}")
        sections_cfg[section] = nv
        display["sections"] = sections_cfg
        cfg["display"] = display
        services.save_cfg(cfg)
        return _ok(services, rid, {"key": key, "value": nv})

    if key == "thinking_mode":
        nv = str(value or "").strip().lower()
        if nv not in {"collapsed", "truncated", "full"}:
            return _err(services, rid, 4002, f"unknown thinking_mode: {value}")
        services.write_config_key("display.thinking_mode", nv)
        services.write_config_key("display.details_mode", "expanded" if nv == "full" else "collapsed")
        return _ok(services, rid, {"key": key, "value": nv})

    if key == "density":
        raw = str(value or "").strip().lower()
        d0 = services.load_cfg().get("display") if isinstance(services.load_cfg().get("display"), dict) else {}
        cur_b = bool(d0.get("tui_compact", False))
        if raw in {"", "toggle"}:
            nv_b = not cur_b
        elif raw == "on":
            nv_b = True
        elif raw == "off":
            nv_b = False
        else:
            return _err(services, rid, 4002, f"unknown density value: {value}")
        services.write_config_key("display.tui_compact", nv_b)
        return _ok(services, rid, {"key": key, "value": "on" if nv_b else "off"})

    if key == "battery":
        raw = str(value or "").strip().lower()
        d0 = services.load_cfg().get("display") if isinstance(services.load_cfg().get("display"), dict) else {}
        cur_b = bool(d0.get("battery", False))
        if raw in {"", "toggle"}:
            nv_b = not cur_b
        elif raw in {"on", "true", "yes"}:
            nv_b = True
        elif raw in {"off", "false", "no"}:
            nv_b = False
        else:
            return _err(services, rid, 4002, f"unknown battery value: {value}")
        services.write_config_key("display.battery", nv_b)
        return _ok(services, rid, {"key": key, "value": "on" if nv_b else "off"})

    if key == "theme":
        raw = str(value or "").strip().lower()
        if raw not in {"auto", "light", "dark"}:
            return _err(services, rid, 4002, f"unknown theme value: {value} (use auto|light|dark)")
        services.write_config_key("display.tui_theme", raw)
        return _ok(services, rid, {"key": key, "value": raw})

    if key == "statusbar":
        raw = str(value or "").strip().lower()
        display = services.load_cfg().get("display")
        d0 = display if isinstance(display, dict) else {}
        current = services.coerce_statusbar(d0.get("tui_statusbar", "top"))
        if raw in {"", "toggle"}:
            nv = "top" if current == "off" else "off"
        elif raw == "on":
            nv = "top"
        elif raw in {"top", "bottom", "off"}:
            nv = raw
        else:
            return _err(services, rid, 4002, f"unknown statusbar value: {value}")
        services.write_config_key("display.tui_statusbar", nv)
        return _ok(services, rid, {"key": key, "value": nv})

    if key == "mouse":
        aliases = {
            "0": "off",
            "1": "all",
            "all": "all",
            "any": "all",
            "button": "buttons",
            "buttons": "buttons",
            "click": "buttons",
            "false": "off",
            "full": "all",
            "no": "off",
            "off": "off",
            "on": "all",
            "scroll": "wheel",
            "true": "all",
            "wheel": "wheel",
            "yes": "all",
        }
        raw = ("" if value is None else str(value)).strip().lower()
        cfg = services.load_cfg()
        display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
        current = services.display_mouse_tracking(display)
        if raw in {"", "toggle"}:
            nv = "all" if current == "off" else "off"
        elif raw in aliases:
            nv = aliases[raw]
        else:
            return _err(services, rid, 4002, f"unknown mouse value: {value}")
        services.write_config_key("display.mouse_tracking", nv)
        return _ok(services, rid, {"key": key, "value": nv})

    if key == "indicator":
        raw = ("" if value is None else str(value)).strip().lower()
        if raw not in INDICATOR_STYLES:
            return _err(services, rid, 4002, f"unknown indicator: {raw!r}; pick one of {'|'.join(INDICATOR_STYLES)}")
        services.write_config_key("display.tui_status_indicator", raw)
        return _ok(services, rid, {"key": key, "value": raw})

    if key in {"cwd", "terminal.cwd", "workdir"}:
        raw = str(value or "").strip()
        if not raw:
            return _err(services, rid, 4002, "cwd required")
        cwd = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isdir(cwd):
            return _err(services, rid, 4002, f"working directory does not exist: {raw}")
        services.write_config_key("terminal.cwd", cwd)
        os.environ["TERMINAL_CWD"] = cwd
        return _ok(services, rid, {"key": "terminal.cwd", "value": cwd, "cwd": cwd, "branch": services.git_branch_for_cwd(cwd)})

    if key in {"prompt", "personality", "skin"}:
        try:
            cfg = services.load_cfg_raw()
            if key == "prompt":
                if value == "clear":
                    cfg.pop("custom_prompt", None)
                    nv = ""
                else:
                    cfg["custom_prompt"] = value
                    nv = value
                services.save_cfg(cfg)
            elif key == "personality":
                sid_key = params.get("session_id", "")
                pname, new_prompt = services.validate_personality(str(value or ""), cfg)
                services.write_config_key("display.personality", pname)
                services.write_config_key("agent.system_prompt", new_prompt)
                nv = str(value or "none")
                history_reset, info = services.apply_personality_to_session(sid_key, session, new_prompt, pname)
            else:
                services.write_config_key(f"display.{key}", value)
                nv = value
                services.broadcast_global_event("skin.changed", services.resolve_skin())
                services.note_skin_broadcast()
            resp = {"key": key, "value": nv}
            if key == "personality":
                resp["history_reset"] = history_reset
                if info is not None:
                    resp["info"] = info
            return _ok(services, rid, resp)
        except Exception as e:
            return _err(services, rid, 5001, str(e))

    return _err(services, rid, 4002, f"unknown config key: {key}")


def _set_yolo(rid: Any, params: dict, value: Any, session: dict | None, services: ConfigServices) -> dict:
    try:
        from tools.approval import disable_session_yolo, enable_session_yolo, is_session_yolo_enabled
        from utils import is_truthy_value

        scope = str(params.get("scope") or "session").strip().lower()
        raw = str(value or "").strip().lower()

        def resolve_toggle(current: bool) -> bool:
            if raw in {"1", "on", "true", "yes"}:
                return True
            if raw in {"0", "off", "false", "no"}:
                return False
            return not current

        if scope == "global":
            from tools.approval import _normalize_approval_mode

            cfg = services.load_cfg()
            appr = cfg.get("approvals") if isinstance(cfg, dict) else None
            current = _normalize_approval_mode((appr if isinstance(appr, dict) else {}).get("mode", "manual")) == "off"
            enable = resolve_toggle(current)
            services.write_config_key("approvals.mode", "off" if enable else "manual")
            for sid, sess in _session_items(services):
                agent = sess.get("agent")
                if agent is not None:
                    services.emit("session.info", sid, services.session_info(agent, sess))
            return _ok(services, rid, {"key": "yolo", "value": "1" if enable else "0", "scope": "global"})

        if session:
            current = is_session_yolo_enabled(session["session_key"])
            enable = resolve_toggle(current)
            if enable:
                enable_session_yolo(session["session_key"])
                nv = "1"
            else:
                disable_session_yolo(session["session_key"])
                nv = "0"
            agent = session.get("agent")
            if agent is not None:
                services.emit("session.info", params.get("session_id", ""), services.session_info(agent, session))
        else:
            current = is_truthy_value(os.environ.get("HERMES_YOLO_MODE"))
            enable = resolve_toggle(current)
            if enable:
                os.environ["HERMES_YOLO_MODE"] = "1"
                nv = "1"
            else:
                os.environ.pop("HERMES_YOLO_MODE", None)
                nv = "0"
        return _ok(services, rid, {"key": "yolo", "value": nv, "scope": "session"})
    except Exception as e:
        return _err(services, rid, 5001, str(e))


def _set_reasoning(rid: Any, params: dict, value: Any, session: dict | None, services: ConfigServices) -> dict:
    try:
        from hermes_constants import parse_reasoning_effort

        arg = str(value or "").strip().lower()
        scope = str(params.get("scope") or "").strip().lower()
        if arg in {"show", "on", "hide", "off", "full", "all", "clamp", "collapse", "short"}:
            cfg = services.load_cfg_raw()
            display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
            sections = display.get("sections") if isinstance(display.get("sections"), dict) else {}
            if arg in {"show", "on"}:
                display["show_reasoning"] = True
                sections["thinking"] = "expanded"
                value_out = "show"
                if session:
                    session["show_reasoning"] = True
            elif arg in {"hide", "off"}:
                display["show_reasoning"] = False
                sections["thinking"] = "hidden"
                value_out = "hide"
                if session:
                    session["show_reasoning"] = False
            elif arg in {"full", "all"}:
                display["reasoning_full"] = True
                sections["thinking"] = "expanded"
                value_out = "full"
            else:
                display["reasoning_full"] = False
                sections["thinking"] = "collapsed"
                value_out = "clamp"
            display["sections"] = sections
            cfg["display"] = display
            services.save_cfg(cfg)
            return _ok(services, rid, {"key": "reasoning", "value": value_out})

        parsed = parse_reasoning_effort(arg)
        if parsed is None:
            return _err(services, rid, 4002, f"unknown reasoning value: {value}")
        if scope == "global" or session is None:
            services.write_config_key("agent.reasoning_effort", arg)
            if session is not None:
                session.pop("create_reasoning_override", None)
        else:
            session["create_reasoning_override"] = parsed
        if session and session.get("agent") is not None:
            session["agent"].reasoning_config = parsed
            services.persist_live_session_runtime(session)
            services.emit("session.info", params.get("session_id", ""), services.session_info(session["agent"], session))
        return _ok(services, rid, {"key": "reasoning", "value": arg})
    except Exception as e:
        return _err(services, rid, 5001, str(e))


_E_PROJECTS = 5061
_E_NO_PROJECT = 5062
_E_PROJECT_ARG = 5063


class _NoProject(Exception):
    pass


def _projects_payload(conn) -> dict:
    from hermes_cli import projects_db as pdb

    return {"projects": [p.to_dict() for p in pdb.list_projects(conn, include_archived=True)], "active_id": pdb.get_active_id(conn)}


def _require_project(pdb, conn, params: dict):
    proj = pdb.get_project(conn, str(params.get("id") or ""))
    if proj is None:
        raise _NoProject
    return proj


def _project_handler(fn: Callable[[Any, dict, Any, Any, ConfigServices], dict]) -> Callable[[Any, dict, ConfigServices], dict]:
    def handler(rid: Any, params: dict, services: ConfigServices) -> dict:
        try:
            from hermes_cli import projects_db as pdb

            with pdb.connect_closing() as conn:
                return fn(rid, params, pdb, conn, services)
        except _NoProject:
            return _err(services, rid, _E_NO_PROJECT, "no such project")
        except ValueError as e:
            return _err(services, rid, _E_PROJECT_ARG, str(e))
        except Exception as e:
            return _err(services, rid, _E_PROJECTS, str(e))

    return handler


@_project_handler
def _projects_list(rid: Any, params: dict, pdb, conn, services: ConfigServices) -> dict:
    return _ok(services, rid, _projects_payload(conn))


@_project_handler
def _projects_get(rid: Any, params: dict, pdb, conn, services: ConfigServices) -> dict:
    return _ok(services, rid, {"project": _require_project(pdb, conn, params).to_dict()})


@_project_handler
def _projects_create(rid: Any, params: dict, pdb, conn, services: ConfigServices) -> dict:
    pid = pdb.create_project(
        conn,
        name=str(params.get("name") or ""),
        slug=params.get("slug"),
        folders=params.get("folders") or [],
        primary_path=params.get("primary_path"),
        description=params.get("description"),
        icon=params.get("icon"),
        color=params.get("color"),
        board_slug=params.get("board_slug"),
    )
    if params.get("use"):
        pdb.set_active(conn, pid)
    proj = pdb.get_project(conn, pid)
    return _ok(services, rid, {"project": proj.to_dict() if proj else None})


@_project_handler
def _projects_update(rid: Any, params: dict, pdb, conn, services: ConfigServices) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.update_project(conn, proj.id, name=params.get("name"), description=params.get("description"), icon=params.get("icon"), color=params.get("color"), board_slug=params.get("board_slug"))
    return _ok(services, rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_project_handler
def _projects_add_folder(rid: Any, params: dict, pdb, conn, services: ConfigServices) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.add_folder(conn, proj.id, str(params.get("path") or ""), label=params.get("label"), is_primary=bool(params.get("is_primary")))
    return _ok(services, rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_project_handler
def _projects_remove_folder(rid: Any, params: dict, pdb, conn, services: ConfigServices) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.remove_folder(conn, proj.id, str(params.get("path") or ""))
    return _ok(services, rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_project_handler
def _projects_set_primary(rid: Any, params: dict, pdb, conn, services: ConfigServices) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.set_primary(conn, proj.id, str(params.get("path") or ""))
    return _ok(services, rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_project_handler
def _projects_archive(rid: Any, params: dict, pdb, conn, services: ConfigServices) -> dict:
    proj = _require_project(pdb, conn, params)
    (pdb.restore_project if params.get("restore") else pdb.archive_project)(conn, proj.id)
    return _ok(services, rid, _projects_payload(conn))


@_project_handler
def _projects_delete(rid: Any, params: dict, pdb, conn, services: ConfigServices) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.delete_project(conn, proj.id)
    return _ok(services, rid, _projects_payload(conn))


@_project_handler
def _projects_set_active(rid: Any, params: dict, pdb, conn, services: ConfigServices) -> dict:
    pdb.set_active(conn, _require_project(pdb, conn, params).id if params.get("id") else None)
    return _ok(services, rid, {"active_id": pdb.get_active_id(conn)})


@_project_handler
def _projects_for_cwd(rid: Any, params: dict, pdb, conn, services: ConfigServices) -> dict:
    cwd = services.completion_cwd({"cwd": str(params.get("cwd") or "").strip()} if params.get("cwd") else {})
    proj = pdb.project_for_path(conn, cwd)
    return _ok(services, rid, {"project": proj.to_dict() if proj else None, "cwd": cwd, "branch": services.git_branch_for_cwd(cwd)})


def _is_repo_junk(root: str) -> bool:
    if not root:
        return True
    from hermes_constants import get_hermes_home

    real = os.path.realpath(root)
    home = os.path.realpath(os.path.expanduser("~"))
    hermes_home = os.path.realpath(str(get_hermes_home()))
    return real == home or real == hermes_home or real.startswith(hermes_home + os.sep)


def _is_session_cwd_junk(cwd: str) -> bool:
    if not cwd:
        return True
    from hermes_constants import get_hermes_home

    real = os.path.normcase(os.path.realpath(cwd))
    home = os.path.normcase(os.path.realpath(os.path.expanduser("~")))
    hermes_home = os.path.normcase(os.path.realpath(str(get_hermes_home())))
    return real == home or real == hermes_home


def _repo_discovery_policy(services: ConfigServices, raw: dict | None = None) -> dict:
    from hermes_cli.config import DEFAULT_CONFIG

    defaults = DEFAULT_CONFIG["desktop"]
    source = raw if isinstance(raw, dict) else (services.load_cfg().get("desktop") or {})
    if not isinstance(source, dict):
        source = {}
    enabled = source.get("enabled", source.get("repo_scan_enabled", defaults["repo_scan_enabled"]))
    roots = source.get("roots", source.get("repo_scan_roots", defaults["repo_scan_roots"]))
    excludes = source.get("exclude_paths", source.get("repo_scan_exclude_paths", defaults["repo_scan_exclude_paths"]))
    return {
        "enabled": enabled if isinstance(enabled, bool) else defaults["repo_scan_enabled"],
        "roots": [v.strip() for v in roots if isinstance(v, str) and v.strip()] if isinstance(roots, list) else list(defaults["repo_scan_roots"]),
        "exclude_paths": [v.strip() for v in excludes if isinstance(v, str) and v.strip()] if isinstance(excludes, list) else list(defaults["repo_scan_exclude_paths"]),
    }


def _repo_discovery_policy_key(policy: dict) -> str:
    def paths(values: list[str]) -> list[str]:
        normalized = set()
        home = os.path.expanduser("~")
        for value in values:
            expanded = os.path.expanduser(value)
            if not os.path.isabs(expanded):
                expanded = os.path.join(home, expanded)
            normalized.add(os.path.normcase(os.path.abspath(expanded)))
        return sorted(normalized)

    canonical = {"enabled": bool(policy["enabled"]), "roots": paths(policy["roots"]), "exclude_paths": paths(policy["exclude_paths"])}
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _repo_discovery_policy_is_default(services: ConfigServices, policy: dict) -> bool:
    from hermes_cli.config import DEFAULT_CONFIG

    return _repo_discovery_policy_key(policy) == _repo_discovery_policy_key(_repo_discovery_policy(services, DEFAULT_CONFIG["desktop"]))


def _discover_repos_payload(services: ConfigServices, db, *, conn=None, backfill: bool = True, include_cached: bool = True) -> list[dict]:
    repos: dict[str, dict] = {}

    def agg(root: str) -> dict:
        return repos.setdefault(root, {"root": root, "label": "", "sessions": 0, "last_active": 0.0})

    cwd_rows = list(db.distinct_session_cwds())
    git_probe.warm_roots(str(r.get("cwd") or "") for r in cwd_rows)
    cwd_to_root: dict[str, str] = {}
    for row in cwd_rows:
        cwd = str(row.get("cwd") or "")
        root = services.git_common_repo_root_for_cwd(cwd)
        if not root:
            continue
        cwd_to_root[cwd] = root
        if _is_repo_junk(root):
            continue
        entry = agg(root)
        entry["sessions"] += int(row.get("sessions") or 0)
        entry["last_active"] = max(entry["last_active"], float(row.get("last_active") or 0))
    if backfill:
        try:
            db.backfill_repo_roots(cwd_to_root)
        except Exception:
            pass
    if include_cached:
        try:
            from hermes_cli import projects_db as pdb

            def read(c) -> None:
                for entry in pdb.list_discovered_repos(c):
                    root = str(entry.get("root") or "")
                    if not root or _is_repo_junk(root):
                        continue
                    repo = agg(root)
                    if entry.get("label"):
                        repo["label"] = entry["label"]

            if conn is not None:
                read(conn)
            else:
                with pdb.connect_closing() as own:
                    read(own)
        except Exception:
            pass
    out = sorted(repos.values(), key=lambda repo: repo["last_active"], reverse=True)
    for repo in out:
        repo["label"] = repo["label"] or os.path.basename(repo["root"].rstrip("/\\")) or repo["root"]
    return out


def _projects_discover_repos(rid: Any, params: dict, services: ConfigServices) -> dict:
    try:
        db = services.get_db()
        if db is None:
            return _ok(services, rid, {"repos": []})
        from hermes_cli import projects_db as pdb

        policy = _repo_discovery_policy(services)
        policy_key = _repo_discovery_policy_key(policy)
        with pdb.connect_closing() as conn:
            pdb.reconcile_discovered_repos_policy(conn, policy_key, preserve_unversioned=_repo_discovery_policy_is_default(services, policy))
            repos = _discover_repos_payload(services, db, conn=conn, include_cached=policy["enabled"])
        return _ok(services, rid, {"repos": repos, "discovery_policy": policy})
    except Exception as e:
        return _err(services, rid, 5061, str(e))


def _projects_record_repos(rid: Any, params: dict, services: ConfigServices) -> dict:
    try:
        from hermes_cli import projects_db as pdb

        policy = _repo_discovery_policy(services)
        policy_key = _repo_discovery_policy_key(policy)
        incoming_raw = params.get("discovery_policy")
        incoming_policy = _repo_discovery_policy(services, incoming_raw) if isinstance(incoming_raw, dict) else None
        incoming_matches = incoming_policy is not None and _repo_discovery_policy_key(incoming_policy) == policy_key
        accept_legacy_default = incoming_policy is None and _repo_discovery_policy_is_default(services, policy)
        pairs: list[tuple[str, str | None]] = []
        for item in params.get("repos") or []:
            if isinstance(item, str):
                pairs.append((item, None))
            elif isinstance(item, dict) and item.get("root"):
                pairs.append((str(item["root"]), item.get("label")))
        with pdb.connect_closing() as conn:
            pdb.reconcile_discovered_repos_policy(conn, policy_key, preserve_unversioned=_repo_discovery_policy_is_default(services, policy))
            accepted = bool(policy["enabled"] and (incoming_matches or accept_legacy_default))
            if accepted:
                pdb.record_discovered_repos(conn, pairs, replace=True, policy_key=policy_key)
            elif not policy["enabled"]:
                pdb.clear_discovered_repos(conn, policy_key=policy_key)
        db = services.get_db()
        repos = _discover_repos_payload(services, db, include_cached=policy["enabled"]) if db is not None else []
        return _ok(services, rid, {"repos": repos, "accepted": accepted, "discovery_policy": policy})
    except Exception as e:
        return _err(services, rid, 5061, str(e))


_PROJECT_TREE_EXCLUDED_SOURCES = ["cron", "kanban"]
_DIR_EXISTS_CACHE: dict[str, bool] = {}


def _project_tree_row(r: Mapping[str, Any]) -> dict:
    return {
        "id": r.get("id"),
        "_lineage_root_id": r.get("_lineage_root_id"),
        "parent_session_id": r.get("parent_session_id"),
        "title": r.get("title"),
        "preview": r.get("preview"),
        "started_at": r.get("started_at") or 0,
        "ended_at": r.get("ended_at"),
        "last_active": r.get("last_active") or r.get("started_at") or 0,
        "source": r.get("source"),
        "archived": bool(r.get("archived")),
        "message_count": r.get("message_count") or 0,
        "tool_call_count": r.get("tool_call_count") or 0,
        "input_tokens": r.get("input_tokens") or 0,
        "output_tokens": r.get("output_tokens") or 0,
        "model": r.get("model"),
        "is_active": False,
        "cwd": r.get("cwd"),
        "git_branch": r.get("git_branch"),
        "git_repo_root": r.get("git_repo_root"),
    }


def _project_tree_inputs(services: ConfigServices, db, session_limit: int, *, include_discovered: bool) -> tuple[list[dict], list[dict], list[dict], str | None]:
    rows = db.list_sessions_rich(limit=session_limit, offset=0, order_by_last_active=True, min_message_count=1, include_children=False, exclude_sources=_PROJECT_TREE_EXCLUDED_SOURCES, include_archived=False)
    sessions = [_project_tree_row(r) for r in rows]
    git_probe.warm_roots(s["cwd"] for s in sessions if s.get("cwd"))
    from hermes_cli import projects_db as pdb

    policy = _repo_discovery_policy(services)
    policy_key = _repo_discovery_policy_key(policy)
    with pdb.connect_closing() as conn:
        if include_discovered:
            pdb.reconcile_discovered_repos_policy(conn, policy_key, preserve_unversioned=_repo_discovery_policy_is_default(services, policy))
        projects = [p.to_dict() for p in pdb.list_projects(conn)]
        active_id = pdb.get_active_id(conn)
        discovered = _discover_repos_payload(services, db, conn=conn, backfill=False, include_cached=policy["enabled"]) if include_discovered else []
    return sessions, projects, discovered, active_id


def _dir_exists_cached(path: str) -> bool:
    hit = _DIR_EXISTS_CACHE.get(path)
    if hit is None:
        hit = os.path.isdir(path)
        _DIR_EXISTS_CACHE[path] = hit
    return hit


def _build_project_tree(services: ConfigServices, db, *, preview_limit: int, hydrate: bool, session_limit: int, include_discovered: bool) -> tuple[dict, str | None]:
    from tui_gateway import project_tree

    _DIR_EXISTS_CACHE.clear()
    sessions, projects, discovered, active_id = _project_tree_inputs(services, db, session_limit, include_discovered=include_discovered)
    tree = project_tree.build_tree(projects, sessions, discovered, services.resolve_cwd_git, preview_limit=preview_limit, hydrate=hydrate, is_junk_root=_is_repo_junk, is_junk_cwd=_is_session_cwd_junk, exists=_dir_exists_cached)
    return tree, active_id


def _projects_tree(rid: Any, params: dict, services: ConfigServices) -> dict:
    try:
        db = services.get_db()
        if db is None:
            return _ok(services, rid, {"projects": [], "active_id": None, "scoped_session_ids": []})
        tree, active_id = _build_project_tree(services, db, preview_limit=int(params.get("preview_limit") or 3), hydrate=False, session_limit=int(params.get("session_limit") or 2000), include_discovered=True)
        return _ok(services, rid, {"projects": tree["projects"], "active_id": active_id, "scoped_session_ids": tree["scoped_session_ids"]})
    except Exception as e:
        return _err(services, rid, 5061, str(e))


def _projects_project_sessions(rid: Any, params: dict, services: ConfigServices) -> dict:
    try:
        project_id = str(params.get("project_id") or "")
        if not project_id:
            return _err(services, rid, 5063, "project_id required")
        db = services.get_db()
        if db is None:
            return _ok(services, rid, {"project": None})
        tree, _active = _build_project_tree(services, db, preview_limit=0, hydrate=True, session_limit=int(params.get("session_limit") or 5000), include_discovered=False)
        proj = next((p for p in tree["projects"] if p["id"] == project_id), None)
        return _ok(services, rid, {"project": proj})
    except Exception as e:
        return _err(services, rid, 5061, str(e))


def compute_mcp_rev(load_cfg: Callable[[], dict]) -> str:
    try:
        cfg = load_cfg()
        rev_src = json.dumps({"mcp": cfg.get("mcp"), "mcp_servers": cfg.get("mcp_servers"), "tools": cfg.get("tools")}, sort_keys=True, default=str)
        return hashlib.sha1(rev_src.encode()).hexdigest()[:12]
    except Exception:
        return ""


def _setup_status(rid: Any, params: dict, services: ConfigServices) -> dict:
    try:
        from hermes_cli.main import _has_any_provider_configured

        return _ok(services, rid, {"provider_configured": bool(_has_any_provider_configured())})
    except Exception as e:
        return _err(services, rid, 5016, str(e))


def _setup_runtime_check(rid: Any, params: dict, services: ConfigServices) -> dict:
    try:
        from hermes_cli.auth import has_usable_secret
        from hermes_cli.main import _has_any_provider_configured
        from hermes_cli.runtime_provider import resolve_runtime_provider

        requested = str(params.get("provider") or "").strip() or None
        runtime = resolve_runtime_provider(requested=requested)
        provider_configured = bool(_has_any_provider_configured())
        provider = runtime.get("provider") or "provider"
        source = str(runtime.get("source") or "")
        if not provider_configured and provider == "bedrock" and source in {"iam-role", "aws-sdk-default-chain"}:
            return _ok(services, rid, {"ok": False, "provider": provider, "model": runtime.get("model"), "source": source, "error": "No Hermes provider is configured."})
        api_key = runtime.get("api_key")
        api_key_text = "" if callable(api_key) else str(api_key or "").strip()
        credential_ok = callable(api_key) or api_key_text in {"aws-sdk", "no-key-required"} or has_usable_secret(api_key_text) or bool(runtime.get("command"))
        if not credential_ok:
            return _ok(services, rid, {"ok": False, "provider": provider, "model": runtime.get("model"), "source": runtime.get("source"), "error": f"No usable credentials found for {provider}."})
        return _ok(services, rid, {"ok": True, "provider": runtime.get("provider"), "model": runtime.get("model"), "source": runtime.get("source")})
    except Exception as e:
        return _ok(services, rid, {"ok": False, "error": str(e)})


_METHOD_IMPLS: dict[str, Callable[[Any, dict, ConfigServices], dict]] = {
    "config.get": _config_get,
    "config.set": _config_set,
    "projects.discover_repos": _projects_discover_repos,
    "projects.record_repos": _projects_record_repos,
    "projects.tree": _projects_tree,
    "projects.project_sessions": _projects_project_sessions,
    "projects.list": _projects_list,
    "projects.get": _projects_get,
    "projects.create": _projects_create,
    "projects.update": _projects_update,
    "projects.add_folder": _projects_add_folder,
    "projects.remove_folder": _projects_remove_folder,
    "projects.set_primary": _projects_set_primary,
    "projects.archive": _projects_archive,
    "projects.delete": _projects_delete,
    "projects.set_active": _projects_set_active,
    "projects.for_cwd": _projects_for_cwd,
    "setup.status": _setup_status,
    "setup.runtime_check": _setup_runtime_check,
}


def _bind(name: str, fn: Callable[[Any, dict, ConfigServices], dict], services: ConfigServices) -> Callable[[Any, dict], dict]:
    def handler(rid: Any, params: dict) -> dict:
        return fn(rid, params, services)

    handler.__name__ = name.replace(".", "_")
    handler.__qualname__ = handler.__name__
    handler.__module__ = __name__
    return handler


def register(methods: dict[str, Callable[[Any, dict], dict]], *, services: ConfigServices) -> None:
    """Register direct config-domain callables under the existing RPC names."""
    REGISTERED_METHODS.clear()
    for name in CONFIG_METHODS:
        handler = _bind(name, _METHOD_IMPLS[name], services)
        REGISTERED_METHODS[name] = handler
        methods[name] = handler
