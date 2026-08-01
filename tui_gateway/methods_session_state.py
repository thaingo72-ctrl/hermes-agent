"""Canonical TUI session-state JSON-RPC handlers."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home


@dataclass(frozen=True)
class RpcServices:
    ok: Callable[[Any, dict], dict]
    err: Callable[[Any, int, str], dict]
    db_unavailable_error: Callable[..., dict]
    emit: Callable[[str, str, dict], None]
    status_update: Callable[[str, str, str | None], None]


@dataclass(frozen=True)
class SessionAccessServices:
    sess: Callable[[dict, Any], tuple[dict | None, dict | None]]
    sess_nowait: Callable[[dict, Any], tuple[dict | None, dict | None]]
    session_db: Callable[[dict], Any]
    profile_db: Callable[[dict | None], Any]
    session_resume_lock: Any
    pop_session_by_id: Callable[[str], dict | None]
    teardown_popped_session: Callable[[dict | None], bool]


@dataclass(frozen=True)
class SessionViewServices:
    history_to_messages: Callable[[list[dict] | None], list[dict]]
    metadata_mirror: Callable[[dict | None], dict]
    session_usage_snapshot: Callable[[dict | None], dict]
    project_info_for_cwd: Callable[[str], dict | None]
    display_session_cwd: Callable[[dict | None], str]
    session_info: Callable[[Any, dict | None], dict]


@dataclass(frozen=True)
class ComputeHostServices:
    session_uses_compute_host: Callable[[dict], bool]
    send_compute_host_control: Callable[..., dict]
    apply_metadata_mirror: Callable[[dict, dict | None], None]
    get_supervisor: Callable[[], Any]


@dataclass(frozen=True)
class CompressionServices:
    compress_session_history: Callable[..., tuple[int, dict]]
    sync_session_key_after_compress: Callable[[str, dict], None]
    lock_held_error: type[Exception]


@dataclass(frozen=True)
class BranchServices:
    new_session_key: Callable[[], str]
    session_source: Callable[[dict | None], str]
    resolve_model: Callable[[], str]
    session_cwd: Callable[[dict | None], str]
    make_agent: Callable[..., Any]
    init_session: Callable[..., Any]
    sessions: dict
    set_session_context: Callable[..., Any]
    clear_session_context: Callable[[list], None]
    set_hermes_home_override: Callable[[str], Any]
    reset_hermes_home_override: Callable[[Any], None]


@dataclass(frozen=True)
class InterruptServices:
    tts_stream_stop: Callable[[], None]
    clear_pending: Callable[[str | None], None]
    clear_inflight_turn: Callable[[dict], None]
    resolve_gateway_approval: Callable[..., Any]


@dataclass(frozen=True)
class SessionStateServices:
    rpc: RpcServices
    access: SessionAccessServices
    view: SessionViewServices
    compute: ComputeHostServices
    compression: CompressionServices
    branch: BranchServices
    interrupt: InterruptServices


def session_status(rid, params: dict, *, services: SessionStateServices) -> dict:
    session, err = services.access.sess_nowait(params, rid)
    if err:
        return err

    from hermes_constants import display_hermes_home

    key = session.get("session_key") or params.get("session_id") or ""
    agent = session.get("agent")
    meta = {}
    with services.access.session_db(session) as db:
        if db is None:
            with services.access.profile_db(params) as db2:
                db = db2
                if db and key:
                    try:
                        meta = db.get_session(key) or {}
                    except Exception:
                        meta = {}
                db = None
        if db is not None and key:
            try:
                meta = db.get_session(key) or {}
            except Exception:
                meta = {}

    def _dt(value, fallback: datetime | None = None) -> datetime:
        if value:
            try:
                return datetime.fromtimestamp(float(value))
            except Exception:
                pass
        return fallback or datetime.now()

    created = _dt(meta.get("started_at"))
    updated = created
    for field in ("updated_at", "last_updated_at", "last_activity_at"):
        if meta.get(field):
            updated = _dt(meta.get(field), created)
            break

    mirror = services.view.metadata_mirror(session)
    usage = services.view.session_usage_snapshot(session)
    provider = getattr(agent, "provider", None) or mirror.get("provider") or "unknown"
    model = getattr(agent, "model", None) or mirror.get("model") or "(unknown)"
    project = services.view.project_info_for_cwd(
        services.view.display_session_cwd(session)
    )
    lines = [
        "Hermes TUI Status",
        "",
        f"Session ID: {key}",
        f"Path: {display_hermes_home()}",
    ]
    if project:
        lines.append(f"Project: {project['name']}")
    title = (meta.get("title") or "").strip()
    if title:
        lines.append(f"Title: {title}")
    lines.extend(
        [
            f"Model: {model} ({provider})",
            f"Created: {created.strftime('%Y-%m-%d %H:%M')}",
            f"Last Activity: {updated.strftime('%Y-%m-%d %H:%M')}",
            f"Tokens: {int(usage.get('total') or 0):,}",
            f"Agent Running: {'Yes' if session.get('running') else 'No'}",
        ]
    )
    return services.rpc.ok(rid, {"output": "\n".join(lines)})


def session_history(rid, params: dict, *, services: SessionStateServices) -> dict:
    session, err = services.access.sess_nowait(params, rid)
    if err:
        return err
    history = list(session.get("history", []))
    if session.get("session_key"):
        with services.access.session_db(session) as db:
            if db is not None:
                try:
                    history = db.get_messages_as_conversation(
                        session["session_key"], include_ancestors=True
                    )
                except Exception:
                    pass
    return services.rpc.ok(
        rid,
        {
            "count": len(history),
            "messages": services.view.history_to_messages(history),
        },
    )


def session_undo(rid, params: dict, *, services: SessionStateServices) -> dict:
    session, err = services.access.sess(params, rid)
    if err:
        return err
    if session.get("running"):
        return services.rpc.err(
            rid, 4009, "session busy — /interrupt the current turn before /undo"
        )
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
            session["history_version"] = int(session.get("history_version", 0)) + 1
    return services.rpc.ok(rid, {"removed": removed})


def session_compress(rid, params: dict, *, services: SessionStateServices) -> dict:
    session, err = services.access.sess_nowait(params, rid)
    if err:
        return err
    assert session is not None
    if services.compute.session_uses_compute_host(session):
        sid = str(params.get("session_id") or "")
        focus_topic = str(params.get("focus_topic", "") or "").strip()
        command = "/compress" + (f" {focus_topic}" if focus_topic else "")
        try:
            ack = services.compute.send_compute_host_control(
                sid,
                route_name="session.compress",
                command=command,
                wait=True,
                timeout=120.0,
            )
        except Exception as exc:
            return services.rpc.err(rid, 5019, f"compute-host compress failed: {exc}")
        if ack.get("type") in {"control.error", "error"}:
            return services.rpc.err(
                rid, 4009, str(ack.get("message") or "compute-host compress failed")
            )
        services.compute.apply_metadata_mirror(session, ack)
        host_result = ack.get("result")
        if isinstance(host_result, dict):
            return services.rpc.ok(rid, {**host_result, "turn_isolation": True})
        host_info = (
            ack.get("session_info") if isinstance(ack.get("session_info"), dict) else {}
        )
        host_messages = (
            services.view.history_to_messages(ack.get("messages"))
            if isinstance(ack.get("messages"), list)
            else []
        )
        host_ack = {key: value for key, value in ack.items() if key != "messages"}
        return services.rpc.ok(
            rid,
            {
                "status": "compressed",
                "turn_isolation": True,
                "host_ack": host_ack,
                "info": host_info,
                "messages": host_messages,
                "usage": (
                    host_info.get("usage")
                    if isinstance(host_info.get("usage"), dict)
                    else {}
                ),
            },
        )
    session, err = services.access.sess(params, rid)
    if err:
        return err
    if session.get("running"):
        return services.rpc.err(
            rid, 4009, "session busy — /interrupt the current turn before /compress"
        )
    from agent.conversation_compression import (
        finalize_context_engine_compression_notification,
    )

    sid = params.get("session_id", "")
    focus_topic = str(params.get("focus_topic", "") or "").strip()
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

        if before_count >= 4:
            focus_suffix = f', focus: "{focus_topic}"' if focus_topic else ""
            services.rpc.status_update(
                sid,
                "compressing",
                f"⠋ compressing {before_count} messages "
                f"(~{before_tokens:,} tok){focus_suffix}…",
            )

        try:
            removed, usage = services.compression.compress_session_history(
                session,
                focus_topic,
                approx_tokens=before_tokens,
                before_messages=before_messages,
                history_version=history_version,
            )
            with session["history_lock"]:
                messages = list(session.get("history", []))
            after_count = len(messages)
            sys_prompt_after = getattr(agent, "_cached_system_prompt", "") or sys_prompt
            tools_after = getattr(agent, "tools", None) or tools
            after_tokens = (
                estimate_request_tokens_rough(
                    messages,
                    system_prompt=sys_prompt_after,
                    tools=tools_after,
                )
                if after_count
                else 0
            )
            agent = session["agent"]
            services.compression.sync_session_key_after_compress(sid, session)
            summary = summarize_manual_compression(
                before_messages,
                messages,
                before_tokens,
                after_tokens,
                compression_state=getattr(agent, "context_compressor", None),
            )
            info = services.view.session_info(agent, session)
            services.rpc.emit("session.info", sid, info)
            finalize_context_engine_compression_notification(
                agent,
                committed=True,
            )
            return services.rpc.ok(
                rid,
                {
                    "status": "aborted" if summary["aborted"] else "compressed",
                    "removed": removed,
                    "before_messages": before_count,
                    "after_messages": after_count,
                    "before_tokens": before_tokens,
                    "after_tokens": after_tokens,
                    "summary": summary,
                    "usage": usage,
                    "info": info,
                    "messages": services.view.history_to_messages(messages),
                },
            )
        finally:
            services.rpc.status_update(sid, "ready")
    except services.compression.lock_held_error as exc:
        services.rpc.status_update(sid, "ready")
        from agent.manual_compression_feedback import describe_compression_lock_skip

        return services.rpc.ok(
            rid,
            {
                "compressed": False,
                "lock_held": True,
                "message": describe_compression_lock_skip(exc.holder),
            },
        )
    except Exception as exc:
        finalize_context_engine_compression_notification(
            session["agent"],
            committed=False,
        )
        return services.rpc.err(rid, 5005, str(exc))


def session_save(rid, params: dict, *, services: SessionStateServices) -> dict:
    session, err = services.access.sess(params, rid)
    if err:
        return err

    if services.compute.session_uses_compute_host(session):
        sid = str(params.get("session_id") or "")
        try:
            ack = services.compute.send_compute_host_control(
                sid,
                route_name="session.save",
                wait=True,
            )
        except Exception as exc:
            return services.rpc.err(
                rid, 5011, f"compute-host session save failed: {exc}"
            )
        if ack.get("type") in {"control.error", "error"}:
            return services.rpc.err(
                rid,
                5011,
                str(ack.get("message") or "compute-host session save failed"),
            )
        result = ack.get("result")
        if not isinstance(result, dict):
            return services.rpc.err(
                rid, 5011, "compute-host session save returned an invalid response"
            )
        return services.rpc.ok(rid, result)

    agent = session["agent"]
    saved_dir = get_hermes_home() / "sessions" / "saved"
    try:
        saved_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return services.rpc.err(
            rid, 5011, f"failed to create save directory {saved_dir}: {exc}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = saved_dir / f"hermes_conversation_{timestamp}.json"

    with session["history_lock"]:
        messages = list(session.get("history", []))

    session_id = getattr(agent, "session_id", None) or session.get("session_key") or ""
    agent_start = getattr(agent, "session_start", None)
    if isinstance(agent_start, datetime):
        session_start = agent_start.isoformat()
    else:
        created_at = session.get("created_at")
        session_start = (
            datetime.fromtimestamp(created_at).isoformat()
            if isinstance(created_at, (int, float))
            else ""
        )

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": getattr(agent, "model", ""),
                    "session_id": session_id,
                    "session_start": session_start,
                    "system_prompt": getattr(agent, "_cached_system_prompt", "") or "",
                    "messages": messages,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return services.rpc.ok(rid, {"file": str(path)})
    except Exception as exc:
        return services.rpc.err(rid, 5011, str(exc))


def session_close(rid, params: dict, *, services: SessionStateServices) -> dict:
    sid = params.get("session_id", "")
    with services.access.session_resume_lock:
        session = services.access.pop_session_by_id(sid)
    closed = services.access.teardown_popped_session(session)
    return services.rpc.ok(rid, {"closed": closed})


def session_branch(rid, params: dict, *, services: SessionStateServices) -> dict:
    session, err = services.access.sess(params, rid)
    if err:
        return err
    with services.access.session_db(session) as db:
        if db is None:
            return services.rpc.db_unavailable_error(rid, code=5008)
        old_key = session["session_key"]
        with session["history_lock"]:
            history = [dict(msg) for msg in session.get("history", [])]
        if not history:
            return services.rpc.err(rid, 4008, "nothing to branch — send a message first")
        count = params.get("count")
        if isinstance(count, int) and count > 0:
            history = history[:count]
        new_key = services.branch.new_session_key()
        new_sid = uuid.uuid4().hex[:8]
        source = services.branch.session_source(session)
        lease = None
        branch_name = params.get("name", "")
        try:
            if branch_name:
                title = branch_name
            else:
                current = db.get_session_title(old_key) or "branch"
                title = (
                    db.get_next_title_in_lineage(current)
                    if hasattr(db, "get_next_title_in_lineage")
                    else f"{current} (branch)"
                )
            db.create_session(
                new_key,
                source=source,
                model=services.branch.resolve_model(),
                model_config={"_branched_from": old_key},
                parent_session_id=old_key,
                cwd=services.branch.session_cwd(session),
                profile_name=(
                    Path(session["profile_home"]).name
                    if session.get("profile_home")
                    else None
                ),
            )
            for msg in history:
                db.append_message(
                    session_id=new_key,
                    role=msg.get("role", "user"),
                    content=msg.get("content"),
                    timestamp=msg.get("timestamp"),
                )
            db.set_session_title(new_key, title)
        except Exception as exc:
            if lease is not None:
                lease.release()
            return services.rpc.err(rid, 5008, f"branch failed: {exc}")
    try:
        parent_home = session.get("profile_home")
        branch_db = None
        if parent_home:
            from hermes_state import SessionDB

            branch_db = SessionDB(db_path=Path(parent_home) / "state.db")
        home_token = (
            services.branch.set_hermes_home_override(parent_home)
            if parent_home
            else None
        )
        try:
            tokens = services.branch.set_session_context(new_key)
            try:
                agent = services.branch.make_agent(
                    new_sid,
                    new_key,
                    session_id=new_key,
                    session_db=branch_db,
                    platform_override=source,
                )
            finally:
                services.branch.clear_session_context(tokens)
            services.branch.init_session(
                new_sid,
                new_key,
                agent,
                list(history),
                cols=session.get("cols", 80),
                cwd=services.branch.session_cwd(session),
                session_db=branch_db,
                source=source,
                profile_home=parent_home,
            )
        finally:
            if home_token is not None:
                services.branch.reset_hermes_home_override(home_token)
        if new_sid in services.branch.sessions:
            services.branch.sessions[new_sid]["active_session_lease"] = lease
    except Exception as exc:
        if lease is not None:
            lease.release()
        return services.rpc.err(rid, 5000, f"agent init failed on branch: {exc}")
    branched_session = services.branch.sessions.get(new_sid)
    return services.rpc.ok(
        rid,
        {
            "session_id": new_sid,
            "stored_session_id": new_key,
            "title": title,
            "parent": old_key,
            "message_count": len(history),
            "messages": services.view.history_to_messages(history),
            "info": services.view.session_info(agent, branched_session),
        },
    )


def session_interrupt(rid, params: dict, *, services: SessionStateServices) -> dict:
    services.interrupt.tts_stream_stop()
    session, err = services.access.sess_nowait(params, rid)
    if err:
        return err
    if services.compute.session_uses_compute_host(session):
        sid = str(params.get("session_id") or "")
        if session.get("running"):
            try:
                services.compute.get_supervisor().interrupt(
                    sid, request_id=f"interrupt-{rid}"
                )
            except Exception as exc:
                return services.rpc.err(
                    rid, 5019, f"compute-host interrupt failed: {exc}"
                )
        with session["history_lock"]:
            session["_turn_cancel_requested"] = True
            session["queued_prompt"] = None
        services.interrupt.clear_pending(sid)
        try:
            services.interrupt.resolve_gateway_approval(
                session["session_key"], "deny", resolve_all=True
            )
        except Exception:
            pass
        return services.rpc.ok(rid, {"status": "interrupted", "turn_isolation": True})
    session, err = services.access.sess(params, rid)
    if err:
        return err
    run_thread = session.get("_run_thread")
    run_thread_alive = run_thread is not None and run_thread.is_alive()
    should_interrupt = bool(session.get("running"))
    if should_interrupt and hasattr(session["agent"], "interrupt"):
        session["agent"].interrupt()
    with session["history_lock"]:
        session["_turn_cancel_requested"] = True
        session["queued_prompt"] = None
    if not run_thread_alive:
        with session["history_lock"]:
            if session.get("running"):
                session["running"] = False
                services.interrupt.clear_inflight_turn(session)

    services.interrupt.clear_pending(params.get("session_id", ""))
    try:
        services.interrupt.resolve_gateway_approval(
            session["session_key"], "deny", resolve_all=True
        )
    except Exception:
        pass
    return services.rpc.ok(rid, {"status": "interrupted"})


def register(methods: dict[str, Callable], *, services: SessionStateServices) -> None:
    methods["session.status"] = partial(session_status, services=services)
    methods["session.history"] = partial(session_history, services=services)
    methods["session.undo"] = partial(session_undo, services=services)
    methods["session.compress"] = partial(session_compress, services=services)
    methods["session.save"] = partial(session_save, services=services)
    methods["session.close"] = partial(session_close, services=services)
    methods["session.branch"] = partial(session_branch, services=services)
    methods["session.interrupt"] = partial(session_interrupt, services=services)
