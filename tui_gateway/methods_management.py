"""Management JSON-RPC handlers for the TUI gateway."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import RLock
from typing import Any, Callable, MutableMapping


@dataclass(frozen=True)
class RpcServices:
    ok: Callable[[Any, dict], dict]
    err: Callable[[Any, int, str], dict]


@dataclass(frozen=True)
class BrowserManagementServices:
    resolve_cdp_url: Callable[[], str]
    connect: Callable[[Any, dict], dict]
    disconnect: Callable[[Any], dict]


@dataclass(frozen=True)
class ConfigManagementServices:
    load_cfg: Callable[[], dict]
    resolve_model: Callable[[], str]
    cfg_max_turns: Callable[[dict, int], int]
    cwd: Callable[[], str]
    hermes_home: Callable[[], Path]
    getenv: Callable[[str, str], str]


@dataclass(frozen=True)
class SessionManagementServices:
    session: Callable[[str], dict | None]
    reset_session_agent: Callable[[str, dict], dict]


@dataclass(frozen=True)
class ManagementServices:
    rpc: RpcServices
    browser: BrowserManagementServices
    config: ConfigManagementServices
    sessions: SessionManagementServices
    load_enabled_toolsets: Callable[[], list[str] | None]


def default_management_services(
    *,
    ok: Callable[[Any, dict], dict],
    err: Callable[[Any, int, str], dict],
    sessions: MutableMapping[str, dict],
    sessions_lock: RLock,
    resolve_browser_cdp_url: Callable[[], str],
    browser_connect: Callable[[Any, dict], dict],
    browser_disconnect: Callable[[Any], dict],
    load_cfg: Callable[[], dict],
    resolve_model: Callable[[], str],
    cfg_max_turns: Callable[[dict, int], int],
    hermes_home: Path,
    load_enabled_toolsets: Callable[[], list[str] | None],
    reset_session_agent: Callable[[str, dict], dict],
) -> ManagementServices:
    def session(sid: str) -> dict | None:
        with sessions_lock:
            return sessions.get(sid)

    return ManagementServices(
        rpc=RpcServices(ok=ok, err=err),
        browser=BrowserManagementServices(
            resolve_cdp_url=resolve_browser_cdp_url,
            connect=browser_connect,
            disconnect=browser_disconnect,
        ),
        config=ConfigManagementServices(
            load_cfg=load_cfg,
            resolve_model=resolve_model,
            cfg_max_turns=cfg_max_turns,
            cwd=os.getcwd,
            hermes_home=lambda: Path(hermes_home),
            getenv=os.environ.get,
        ),
        sessions=SessionManagementServices(
            session=session,
            reset_session_agent=reset_session_agent,
        ),
        load_enabled_toolsets=load_enabled_toolsets,
    )


def _browser_manage(rid, params: dict, services: ManagementServices) -> dict:
    action = params.get("action", "status")

    if action == "status":
        url = services.browser.resolve_cdp_url()
        return services.rpc.ok(rid, {"connected": bool(url), "url": url})

    if action == "disconnect":
        return services.browser.disconnect(rid)

    if action != "connect":
        return services.rpc.err(rid, 4015, f"unknown action: {action}")

    return services.browser.connect(rid, params)


def _plugins_list(rid, params: dict, services: ManagementServices) -> dict:
    try:
        from hermes_cli.plugins import get_plugin_manager

        return services.rpc.ok(
            rid,
            {
                "plugins": [
                    {
                        "name": name,
                        "version": getattr(info, "version", "?"),
                        "enabled": getattr(info, "enabled", True),
                    }
                    for name, info in get_plugin_manager()._plugins.items()
                ]
            },
        )
    except Exception as exc:
        return services.rpc.err(rid, 5032, str(exc))


def _config_show(rid, params: dict, services: ManagementServices) -> dict:
    try:
        cfg = services.config.load_cfg()
        model = services.config.resolve_model()
        api_key = services.config.getenv("HERMES_API_KEY", "") or cfg.get("api_key", "")
        masked = f"****{api_key[-4:]}" if len(api_key) > 4 else "(not set)"
        base_url = services.config.getenv("HERMES_BASE_URL", "") or cfg.get("base_url", "")

        sections = [
            {
                "title": "Model",
                "rows": [
                    ["Model", model],
                    ["Base URL", base_url or "(default)"],
                    ["API Key", masked],
                ],
            },
            {
                "title": "Agent",
                "rows": [
                    ["Max Turns", str(services.config.cfg_max_turns(cfg, 500))],
                    ["Toolsets", ", ".join(cfg.get("enabled_toolsets", [])) or "all"],
                    ["Verbose", str(cfg.get("verbose", False))],
                ],
            },
            {
                "title": "Environment",
                "rows": [
                    ["Working Dir", services.config.cwd()],
                    ["Config File", str(services.config.hermes_home() / "config.yaml")],
                ],
            },
        ]
        return services.rpc.ok(rid, {"sections": sections})
    except Exception as exc:
        return services.rpc.err(rid, 5030, str(exc))


def _tools_list(rid, params: dict, services: ManagementServices) -> dict:
    try:
        from toolsets import get_all_toolsets, get_toolset_info

        session = services.sessions.session(params.get("session_id", ""))
        enabled = (
            set(getattr(session["agent"], "enabled_toolsets", []) or [])
            if session
            else set(services.load_enabled_toolsets() or [])
        )

        items = []
        for name in sorted(get_all_toolsets().keys()):
            info = get_toolset_info(name)
            if not info:
                continue
            items.append(
                {
                    "name": name,
                    "description": info["description"],
                    "tool_count": info["tool_count"],
                    "enabled": name in enabled if enabled else True,
                    "tools": info["resolved_tools"],
                }
            )
        return services.rpc.ok(rid, {"toolsets": items})
    except Exception as exc:
        return services.rpc.err(rid, 5031, str(exc))


def _tools_show(rid, params: dict, services: ManagementServices) -> dict:
    try:
        from model_tools import get_tool_definitions, get_toolset_for_tool

        session = services.sessions.session(params.get("session_id", ""))
        enabled = (
            getattr(session["agent"], "enabled_toolsets", None)
            if session
            else services.load_enabled_toolsets()
        )
        tools = get_tool_definitions(
            enabled_toolsets=enabled,
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        sections = {}

        for tool in sorted(tools, key=lambda item: item["function"]["name"]):
            name = tool["function"]["name"]
            desc = str(tool["function"].get("description", "") or "").split("\n")[0]
            if ". " in desc:
                desc = desc[: desc.index(". ") + 1]
            sections.setdefault(get_toolset_for_tool(name) or "unknown", []).append(
                {"name": name, "description": desc}
            )

        return services.rpc.ok(
            rid,
            {
                "sections": [
                    {"name": name, "tools": rows}
                    for name, rows in sorted(sections.items())
                ],
                "total": len(tools),
            },
        )
    except Exception as exc:
        return services.rpc.err(rid, 5034, str(exc))


def _tools_configure(rid, params: dict, services: ManagementServices) -> dict:
    action = str(params.get("action", "") or "").strip().lower()
    targets = [
        str(name).strip() for name in params.get("names", []) or [] if str(name).strip()
    ]
    if action not in {"disable", "enable"}:
        return services.rpc.err(rid, 4017, f"unknown tools action: {action}")
    if not targets:
        return services.rpc.err(rid, 4018, "names required")

    try:
        from hermes_cli.config import load_config, save_config
        from hermes_cli.tools_config import (
            CONFIGURABLE_TOOLSETS,
            _apply_mcp_change,
            _apply_toolset_change,
            _get_platform_tools,
            _get_plugin_toolset_keys,
        )

        cfg = load_config()
        valid_toolsets = {
            ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS
        } | _get_plugin_toolset_keys()
        toolset_targets = [name for name in targets if ":" not in name]
        mcp_targets = [name for name in targets if ":" in name]
        unknown = [name for name in toolset_targets if name not in valid_toolsets]
        toolset_targets = [name for name in toolset_targets if name in valid_toolsets]

        if toolset_targets:
            _apply_toolset_change(cfg, "cli", toolset_targets, action)

        missing_servers = (
            _apply_mcp_change(cfg, mcp_targets, action) if mcp_targets else set()
        )
        save_config(cfg)

        sid = params.get("session_id", "")
        session = services.sessions.session(sid)
        info = services.sessions.reset_session_agent(sid, session) if session else None
        enabled = sorted(
            _get_platform_tools(load_config(), "cli", include_default_mcp_servers=False)
        )
        changed = [
            name
            for name in targets
            if name not in unknown
            and (":" not in name or name.split(":", 1)[0] not in missing_servers)
        ]

        return services.rpc.ok(
            rid,
            {
                "changed": changed,
                "enabled_toolsets": enabled,
                "info": info,
                "missing_servers": sorted(missing_servers),
                "reset": bool(session),
                "unknown": unknown,
            },
        )
    except Exception as exc:
        return services.rpc.err(rid, 5035, str(exc))


def _toolsets_list(rid, params: dict, services: ManagementServices) -> dict:
    try:
        from toolsets import get_all_toolsets, get_toolset_info

        session = services.sessions.session(params.get("session_id", ""))
        enabled = (
            set(getattr(session["agent"], "enabled_toolsets", []) or [])
            if session
            else set(services.load_enabled_toolsets() or [])
        )

        items = []
        for name in sorted(get_all_toolsets().keys()):
            info = get_toolset_info(name)
            if not info:
                continue
            items.append(
                {
                    "name": name,
                    "description": info["description"],
                    "tool_count": info["tool_count"],
                    "enabled": name in enabled if enabled else True,
                }
            )
        return services.rpc.ok(rid, {"toolsets": items})
    except Exception as exc:
        return services.rpc.err(rid, 5032, str(exc))


def _agents_list(rid, params: dict, services: ManagementServices) -> dict:
    try:
        from tools.process_registry import process_registry

        procs = process_registry.list_sessions()
        return services.rpc.ok(
            rid,
            {
                "processes": [
                    {
                        "session_id": proc["session_id"],
                        "command": proc["command"][:80],
                        "status": proc["status"],
                        "uptime": proc["uptime_seconds"],
                    }
                    for proc in procs
                ]
            },
        )
    except Exception as exc:
        return services.rpc.err(rid, 5033, str(exc))


def _cron_manage(rid, params: dict, services: ManagementServices) -> dict:
    action, jid = params.get("action", "list"), params.get("name", "")
    try:
        from tools.cronjob_tools import cronjob

        if action == "list":
            return services.rpc.ok(rid, json.loads(cronjob(action="list")))
        if action == "add":
            return services.rpc.ok(
                rid,
                json.loads(
                    cronjob(
                        action="create",
                        name=jid,
                        schedule=params.get("schedule", ""),
                        prompt=params.get("prompt", ""),
                    )
                ),
            )
        if action in {"remove", "pause", "resume"}:
            return services.rpc.ok(rid, json.loads(cronjob(action=action, job_id=jid)))
        return services.rpc.err(rid, 4016, f"unknown cron action: {action}")
    except Exception as exc:
        return services.rpc.err(rid, 5023, str(exc))


def _learning_frames(rid, params: dict, services: ManagementServices) -> dict:
    try:
        cols = int(params.get("cols", 80) or 80)
        rows = int(params.get("rows", 24) or 24)
        frames = int(params.get("frames", 48) or 48)
    except (TypeError, ValueError):
        cols, rows, frames = 80, 24, 48
    try:
        from agent.learning_graph import build_learning_graph
        from agent.learning_graph_render import render_frames

        payload = build_learning_graph()
        return services.rpc.ok(
            rid,
            render_frames(
                payload,
                cols=max(20, cols),
                rows=max(10, rows),
                frames=frames,
            ),
        )
    except Exception as exc:
        return services.rpc.err(rid, 5000, f"learning.frames failed: {exc}")


def _learning_detail(rid, params: dict, services: ManagementServices) -> dict:
    try:
        from agent.learning_mutations import node_detail

        return services.rpc.ok(rid, node_detail(str(params.get("id", ""))))
    except Exception as exc:
        return services.rpc.err(rid, 5000, f"learning.detail failed: {exc}")


def _learning_delete(rid, params: dict, services: ManagementServices) -> dict:
    try:
        from agent.learning_mutations import delete_node

        return services.rpc.ok(rid, delete_node(str(params.get("id", ""))))
    except Exception as exc:
        return services.rpc.err(rid, 5000, f"learning.delete failed: {exc}")


def _learning_edit(rid, params: dict, services: ManagementServices) -> dict:
    try:
        from agent.learning_mutations import edit_node

        return services.rpc.ok(
            rid,
            edit_node(str(params.get("id", "")), str(params.get("content", ""))),
        )
    except Exception as exc:
        return services.rpc.err(rid, 5000, f"learning.edit failed: {exc}")


def _skills_manage(rid, params: dict, services: ManagementServices) -> dict:
    action, query = params.get("action", "list"), params.get("query", "")
    try:
        if action == "list":
            from hermes_cli.banner import get_available_skills

            return services.rpc.ok(rid, {"skills": get_available_skills()})
        if action == "search":
            from tools.skills_hub import GitHubAuth, create_source_router, unified_search

            raw = (
                unified_search(
                    query,
                    create_source_router(GitHubAuth()),
                    source_filter="all",
                    limit=20,
                )
                or []
            )
            return services.rpc.ok(
                rid,
                {
                    "results": [
                        {"name": result.name, "description": result.description}
                        for result in raw
                    ]
                },
            )
        if action == "install":
            from hermes_cli.skills_hub import do_install

            class QuietConsole:
                def print(self, *args, **kwargs):
                    pass

            do_install(query, skip_confirm=True, console=QuietConsole())
            return services.rpc.ok(rid, {"installed": True, "name": query})
        if action == "browse":
            from hermes_cli.skills_hub import browse_skills

            page = int(params.get("page", 0) or 0) or (
                int(query) if str(query).isdigit() else 1
            )
            return services.rpc.ok(
                rid,
                browse_skills(page=page, page_size=int(params.get("page_size", 20))),
            )
        if action == "inspect":
            from hermes_cli.skills_hub import inspect_skill

            return services.rpc.ok(rid, {"info": inspect_skill(query) or {}})
        return services.rpc.err(rid, 4017, f"unknown skills action: {action}")
    except Exception as exc:
        return services.rpc.err(rid, 5024, str(exc))


def _skills_reload(rid, params: dict, services: ManagementServices) -> dict:
    try:
        from agent.skill_commands import reload_skills

        result = reload_skills()
        added = result.get("added") or []
        removed = result.get("removed") or []
        total = int(result.get("total") or 0)

        lines = ["Reloading skills..."]
        if not added and not removed:
            lines.append("No new skills detected.")
        if added:
            lines.append("Added skills:")
            lines.extend(f"  - {item.get('name', '')}" for item in added)
        if removed:
            lines.append("Removed skills:")
            lines.extend(f"  - {item.get('name', '')}" for item in removed)
        lines.append(f"{total} skill(s) available")
        return services.rpc.ok(rid, {"output": "\n".join(lines), "result": result})
    except Exception as exc:
        return services.rpc.err(rid, 5025, str(exc))


def _plugins_manage(rid, params: dict, services: ManagementServices) -> dict:
    action = params.get("action", "list")
    try:
        from hermes_cli.plugins_cmd import (
            _discover_all_plugins,
            _get_disabled_set,
            _get_enabled_set,
            _plugin_status,
        )

        def rows():
            enabled = _get_enabled_set()
            disabled = _get_disabled_set()
            out = []
            for name, version, desc, source, _directory, key in sorted(
                _discover_all_plugins()
            ):
                out.append(
                    {
                        "name": name,
                        "version": str(version or ""),
                        "description": desc or "",
                        "source": source,
                        "status": _plugin_status(name, enabled, disabled, key=key),
                    }
                )
            return out

        if action == "list":
            plugin_rows = rows()
            user_count = sum(1 for row in plugin_rows if row["source"] != "bundled")
            return services.rpc.ok(
                rid,
                {
                    "plugins": plugin_rows,
                    "user_count": user_count,
                    "bundled_count": len(plugin_rows) - user_count,
                },
            )

        if action == "toggle":
            from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

            name = (params.get("name") or "").strip()
            if not name:
                return services.rpc.err(rid, 4019, "plugins.toggle requires a 'name'")
            enable = bool(params.get("enable"))
            result = dashboard_set_agent_plugin_enabled(name, enabled=enable)
            if not result.get("ok"):
                return services.rpc.err(
                    rid,
                    5026,
                    result.get("error") or "toggle failed",
                )
            row = next((item for item in rows() if item["name"] == name), None)
            return services.rpc.ok(
                rid,
                {
                    "ok": True,
                    "unchanged": bool(result.get("unchanged")),
                    "name": name,
                    "plugin": row,
                },
            )

        return services.rpc.err(rid, 4017, f"unknown plugins action: {action}")
    except Exception as exc:
        return services.rpc.err(rid, 5026, str(exc))


_HANDLERS = {
    "browser.manage": _browser_manage,
    "plugins.list": _plugins_list,
    "plugins.manage": _plugins_manage,
    "config.show": _config_show,
    "tools.list": _tools_list,
    "tools.show": _tools_show,
    "tools.configure": _tools_configure,
    "toolsets.list": _toolsets_list,
    "agents.list": _agents_list,
    "cron.manage": _cron_manage,
    "learning.frames": _learning_frames,
    "learning.detail": _learning_detail,
    "learning.delete": _learning_delete,
    "learning.edit": _learning_edit,
    "skills.manage": _skills_manage,
    "skills.reload": _skills_reload,
}


def _bind(handler: Callable[[Any, dict, ManagementServices], dict], services: ManagementServices):
    bound = partial(handler, services=services)
    bound.__module__ = __name__
    return bound


def register(methods: MutableMapping[str, Callable], *, services: ManagementServices) -> None:
    for name, handler in _HANDLERS.items():
        methods[name] = _bind(handler, services)
