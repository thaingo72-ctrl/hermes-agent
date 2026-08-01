"""Delegation-control JSON-RPC handlers for the TUI gateway."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class DelegationServices:
    sess_nowait: Callable[[dict, Any], tuple[dict | None, dict | None]]
    current_transport: Callable[[], Any]
    stdio_transport: Any
    enqueue_prompt: Callable[[dict, Any, Any], None]
    record_inflight_correction: Callable[[dict, Any], None]
    spawn_trees_root: Callable[[], Path]
    spawn_tree_session_dir: Callable[[str], Path]
    append_spawn_tree_index: Callable[[Path, dict], None]
    read_spawn_tree_index: Callable[[Path], list[dict]]
    list_active_subagents: Callable[[], list]
    is_spawn_paused: Callable[[], bool]
    get_max_spawn_depth: Callable[[], int]
    get_max_concurrent_children: Callable[[], int]
    set_spawn_paused: Callable[[bool], bool]
    interrupt_subagent: Callable[[str], bool]
    now: Callable[[], float] = time.time


def default_delegation_tool_services(
    *,
    sess_nowait: Callable[[dict, Any], tuple[dict | None, dict | None]],
    current_transport: Callable[[], Any],
    stdio_transport: Any,
    enqueue_prompt: Callable[[dict, Any, Any], None],
    record_inflight_correction: Callable[[dict, Any], None],
    spawn_trees_root: Callable[[], Path],
    spawn_tree_session_dir: Callable[[str], Path],
    append_spawn_tree_index: Callable[[Path, dict], None],
    read_spawn_tree_index: Callable[[Path], list[dict]],
) -> DelegationServices:
    """Build lazy service wrappers so tests can patch delegate_tool imports."""

    def list_active_subagents() -> list:
        from tools.delegate_tool import list_active_subagents as service

        return service()

    def is_spawn_paused() -> bool:
        from tools.delegate_tool import is_spawn_paused as service

        return service()

    def get_max_spawn_depth() -> int:
        from tools.delegate_tool import _get_max_spawn_depth as service

        return service()

    def get_max_concurrent_children() -> int:
        from tools.delegate_tool import _get_max_concurrent_children as service

        return service()

    def set_spawn_paused(paused: bool) -> bool:
        from tools.delegate_tool import set_spawn_paused as service

        return service(paused)

    def interrupt_subagent(subagent_id: str) -> bool:
        from tools.delegate_tool import interrupt_subagent as service

        return service(subagent_id)

    return DelegationServices(
        sess_nowait=sess_nowait,
        current_transport=current_transport,
        stdio_transport=stdio_transport,
        enqueue_prompt=enqueue_prompt,
        record_inflight_correction=record_inflight_correction,
        spawn_trees_root=spawn_trees_root,
        spawn_tree_session_dir=spawn_tree_session_dir,
        append_spawn_tree_index=append_spawn_tree_index,
        read_spawn_tree_index=read_spawn_tree_index,
        list_active_subagents=list_active_subagents,
        is_spawn_paused=is_spawn_paused,
        get_max_spawn_depth=get_max_spawn_depth,
        get_max_concurrent_children=get_max_concurrent_children,
        set_spawn_paused=set_spawn_paused,
        interrupt_subagent=interrupt_subagent,
    )


def _ok(rid: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid: Any, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}


def delegation_status(rid: Any, params: dict, services: DelegationServices) -> dict:
    return _ok(
        rid,
        {
            "active": services.list_active_subagents(),
            "paused": services.is_spawn_paused(),
            "max_spawn_depth": services.get_max_spawn_depth(),
            "max_concurrent_children": services.get_max_concurrent_children(),
        },
    )


def delegation_pause(rid: Any, params: dict, services: DelegationServices) -> dict:
    paused = bool(params.get("paused", True))
    return _ok(rid, {"paused": services.set_spawn_paused(paused)})


def subagent_interrupt(rid: Any, params: dict, services: DelegationServices) -> dict:
    subagent_id = str(params.get("subagent_id") or "").strip()
    if not subagent_id:
        return _err(rid, 4000, "subagent_id required")
    ok = services.interrupt_subagent(subagent_id)
    return _ok(rid, {"found": ok, "subagent_id": subagent_id})


def spawn_tree_save(rid: Any, params: dict, services: DelegationServices) -> dict:
    session_id = str(params.get("session_id") or "").strip()
    subagents = params.get("subagents") or []
    if not isinstance(subagents, list) or not subagents:
        return _err(rid, 4000, "subagents list required")

    started_at = params.get("started_at")
    finished_at = params.get("finished_at") or services.now()
    label = str(params.get("label") or "")
    ts = datetime.utcfromtimestamp(float(finished_at)).strftime("%Y%m%dT%H%M%S")
    path = services.spawn_tree_session_dir(session_id or "default") / f"{ts}.json"
    try:
        payload = {
            "session_id": session_id,
            "started_at": float(started_at) if started_at else None,
            "finished_at": float(finished_at),
            "label": label,
            "subagents": subagents,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        return _err(rid, 5000, f"spawn_tree.save failed: {exc}")

    services.append_spawn_tree_index(
        path.parent,
        {
            "path": str(path),
            "session_id": session_id,
            "started_at": payload["started_at"],
            "finished_at": payload["finished_at"],
            "label": label,
            "count": len(subagents),
        },
    )
    return _ok(rid, {"path": str(path), "session_id": session_id})


def spawn_tree_list(rid: Any, params: dict, services: DelegationServices) -> dict:
    session_id = str(params.get("session_id") or "").strip()
    limit = int(params.get("limit") or 50)
    cross_session = bool(params.get("cross_session"))

    if cross_session:
        roots = [p for p in services.spawn_trees_root().iterdir() if p.is_dir()]
    else:
        roots = [services.spawn_tree_session_dir(session_id or "default")]

    entries: list[dict] = []
    for directory in roots:
        indexed = services.read_spawn_tree_index(directory)
        if indexed:
            entries.extend(
                entry
                for entry in indexed
                if (path := entry.get("path")) and Path(path).exists()
            )
            continue

        for path in directory.glob("*.json"):
            if path.name == "_index.jsonl":
                continue
            try:
                stat = path.stat()
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    raw = {}
                subagents = raw.get("subagents") or []
                entries.append(
                    {
                        "path": str(path),
                        "session_id": raw.get("session_id") or directory.name,
                        "finished_at": raw.get("finished_at") or stat.st_mtime,
                        "started_at": raw.get("started_at"),
                        "label": raw.get("label") or "",
                        "count": len(subagents) if isinstance(subagents, list) else 0,
                    }
                )
            except OSError:
                continue

    entries.sort(key=lambda entry: entry.get("finished_at") or 0, reverse=True)
    return _ok(rid, {"entries": entries[:limit]})


def spawn_tree_load(rid: Any, params: dict, services: DelegationServices) -> dict:
    raw_path = str(params.get("path") or "").strip()
    if not raw_path:
        return _err(rid, 4000, "path required")

    root = services.spawn_trees_root().resolve()
    try:
        resolved = Path(raw_path).resolve()
        resolved.relative_to(root)
    except (ValueError, OSError) as exc:
        return _err(rid, 4030, f"path outside spawn-trees root: {exc}")

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _err(rid, 5000, f"spawn_tree.load failed: {exc}")

    return _ok(rid, payload)


def session_steer(rid: Any, params: dict, services: DelegationServices) -> dict:
    text = (params.get("text") or "").strip()
    if not text:
        return _err(rid, 4002, "text is required")
    session, error = services.sess_nowait(params, rid)
    if error:
        return error
    agent = session.get("agent") if session is not None else None
    if agent is None or not hasattr(agent, "steer"):
        return _err(rid, 4010, "agent does not support steer")
    try:
        accepted = agent.steer(text)
    except Exception as exc:
        return _err(rid, 5000, f"steer failed: {exc}")
    if accepted:
        with session["history_lock"]:
            services.record_inflight_correction(session, text)
            session["last_active"] = services.now()
    return _ok(rid, {"status": "queued" if accepted else "rejected", "text": text})


def session_redirect(rid: Any, params: dict, services: DelegationServices) -> dict:
    text = (params.get("text") or "").strip()
    if not text:
        return _err(rid, 4002, "text is required")
    session, error = services.sess_nowait(params, rid)
    if error:
        return error
    agent = session.get("agent") if session is not None else None
    if agent is None and session.get("running"):
        services.enqueue_prompt(
            session,
            text,
            services.current_transport() or services.stdio_transport,
        )
        session["last_active"] = services.now()
        return _ok(rid, {"status": "queued", "text": text})
    if (
        agent is None
        or getattr(agent, "_supports_active_turn_redirect", False) is not True
        or not hasattr(agent, "redirect")
    ):
        return _err(rid, 4010, "agent does not support active-turn redirect")
    try:
        accepted = agent.redirect(text)
    except Exception as exc:
        return _err(rid, 5000, f"redirect failed: {exc}")
    if accepted:
        with session["history_lock"]:
            services.record_inflight_correction(session, text)
            session["last_active"] = services.now()
    return _ok(
        rid,
        {"status": "redirected" if accepted else "rejected", "text": text},
    )


def terminal_resize(rid: Any, params: dict, services: DelegationServices) -> dict:
    session, error = services.sess_nowait(params, rid)
    if error:
        return error
    session["cols"] = int(params.get("cols", 80))
    return _ok(rid, {"cols": session["cols"]})


def register(
    methods: dict[str, Callable[[Any, dict], dict]],
    *,
    services: DelegationServices,
) -> None:
    """Register delegation-control callables under their JSON-RPC names."""

    methods["delegation.status"] = partial(delegation_status, services=services)
    methods["delegation.pause"] = partial(delegation_pause, services=services)
    methods["subagent.interrupt"] = partial(subagent_interrupt, services=services)
    methods["spawn_tree.save"] = partial(spawn_tree_save, services=services)
    methods["spawn_tree.list"] = partial(spawn_tree_list, services=services)
    methods["spawn_tree.load"] = partial(spawn_tree_load, services=services)
    methods["session.steer"] = partial(session_steer, services=services)
    methods["session.redirect"] = partial(session_redirect, services=services)
    methods["terminal.resize"] = partial(terminal_resize, services=services)
