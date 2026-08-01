"""Session lifecycle JSON-RPC handlers for the TUI gateway.

This module owns the lifecycle RPCs that create, list, resume, activate,
delete, and retitle sessions. Runtime state is supplied explicitly through
``SessionLifecycleServices``; the handlers do not rebind or mutate module
globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import logging
import os
import threading
import time
import uuid

from agent.coding_context import project_facts_for as build_project_facts
from agent.replay_cleanup import sanitize_replay_history
from agent.secret_scope import (
    build_profile_secret_scope,
    reset_secret_scope,
    set_secret_scope,
)
from agent.verification_evidence import verification_status as build_verification_status
from hermes_constants import (
    get_hermes_home,
    parse_reasoning_effort,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from utils import is_truthy_value
from .transport import current_transport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionLifecycleServices:
    child_run_active: Callable[[str], bool]
    claim_or_reuse_live: Callable[..., tuple[str, dict] | None]
    clear_session_context: Callable[[Any], None]
    coerce_seed_history: Callable[[Any], list]
    completion_cwd: Callable[[dict], str]
    db_unavailable_error: Callable[..., dict]
    default_session_cwd: Callable[[], str]
    deferred_session_record: Callable[..., dict]
    desktop_backend_contract: int
    emit: Callable[[str, str, dict], None]
    emit_session_info_for_session: Callable[[str, dict], None]
    enable_gateway_prompts: Callable[[], None]
    ensure_session_db_row: Callable[[dict], None]
    err: Callable[[Any, int, str], dict]
    find_live_session_by_key: Callable[[str], tuple[str, dict] | None]
    get_db: Callable[[], Any]
    git_branch_for_cwd: Callable[[str], str | None]
    history_to_messages: Callable[[list], list]
    init_session: Callable[..., None]
    lazy_resume_info: Callable[..., dict]
    live_session_payload: Callable[..., dict]
    load_show_reasoning: Callable[[], bool]
    load_tool_progress_mode: Callable[[], str]
    make_agent: Callable[..., Any]
    maybe_schedule_auto_continue: Callable[[str, dict, str], dict | None]
    new_session_key: Callable[[], str]
    ok: Callable[[Any, dict], dict]
    profile_configured_cwd: Callable[[Path | None], str]
    profile_db: Callable[[dict | None], Any]
    profile_home: Callable[[str | None], Path | None]
    profile_scoped: Callable[[Callable[..., dict]], Callable[..., dict]]
    project_info_for_cwd: Callable[[str], dict]
    register_session_cwd: Callable[[dict], None]
    resolve_model: Callable[[], str]
    resolve_session_source: Callable[[str | None], str]
    response_profile_name: Callable[[str | None], str]
    schedule_agent_build: Callable[[str], None]
    schedule_session_cap_enforcement: Callable[[], None]
    sess_nowait: Callable[[dict, Any], tuple[dict | None, dict | None]]
    session_db: Callable[[dict], Any]
    session_info: Callable[[Any, dict], dict]
    session_live_item: Callable[[str, dict, str], dict]
    session_resume_lock: threading.Lock
    sessions: Callable[[], dict[str, dict]]
    sessions_lock: threading.Lock
    set_session_context: Callable[[str], Any]
    set_session_cwd: Callable[[dict, str], str]
    stdio_transport: Any
    stored_session_runtime_overrides: Callable[[dict], dict]


def session_create(services: SessionLifecycleServices, rid, params: dict) -> dict:
    sid = uuid.uuid4().hex[:8]
    key = services.new_session_key()
    cols = int(params.get("cols", 80))
    history = services.coerce_seed_history(params.get("messages"))
    title = str(params.get("title") or "").strip()
    # When set, this is a branch: the new chat copies an existing conversation's
    # history and links back to it so list_sessions_rich keeps it visible and the
    # sidebar can nest it under its parent. Mirrors the TUI /branch marker.
    parent_session_id = str(params.get("parent_session_id") or "").strip() or None
    # Did the client pick a workspace, or are we falling back to the gateway's
    # launch directory? Only an explicit choice is persisted as the session's
    # workspace (see services.ensure_session_db_row); otherwise it lands in "No
    # workspace" instead of whatever folder the desktop launched in.
    raw_cwd = str(params.get("cwd") or "").strip()
    try:
        explicit_cwd = bool(raw_cwd) and os.path.isdir(os.path.abspath(os.path.expanduser(raw_cwd)))
    except Exception:
        explicit_cwd = False
    resolved_cwd = services.completion_cwd(params)
    source = services.resolve_session_source(str(params.get("source") or "").strip() or None)
    services.enable_gateway_prompts()

    # ``profile`` (app-global remote mode): a new chat started under a non-launch
    # profile must build its agent + persist against THAT profile's home/state.db,
    # not the dashboard's launch profile. Stored on the session so _start_agent_build
    # and each turn re-bind HERMES_HOME. None/own profile → launch (unchanged).
    profile = (params.get("profile") or "").strip() or None
    profile_home = services.profile_home(profile)

    # The desktop composer owns its model/effort/fast as plain UI state and ships
    # it on every session.create. Honor each as a PER-SESSION override (built into
    # the agent below) — never a global config write, so picking a model/effort
    # for a new chat can't mutate the profile default. provider is optional
    # (resolved at build).
    create_model = str(params.get("model") or "").strip()
    session_model_override = (
        {"model": create_model, "provider": str(params.get("provider") or "").strip() or None}
        if create_model
        else None
    )
    create_reasoning_override = None
    if effort := str(params.get("reasoning_effort") or "").strip():
        try:
            from hermes_constants import parse_reasoning_effort

            create_reasoning_override = parse_reasoning_effort(effort)
        except Exception:
            create_reasoning_override = None
    # Presence is part of the contract: omitted means inherit the profile,
    # true pins priority, and false pins normal. Empty string is the internal
    # explicit-normal sentinel because services.make_agent uses None for inheritance.
    create_service_tier_override = None
    if "fast" in params:
        create_service_tier_override = (
            "priority" if is_truthy_value(params.get("fast")) else ""
        )

    ready = threading.Event()
    now = time.time()
    lease = None  # claimed lazily on the first turn (_ensure_active_session_slot)

    with services.sessions_lock:
        sessions = services.sessions()
        sessions[sid] = {
            "agent": None,
            "agent_error": None,
            "agent_ready": ready,
            "attached_images": [],
            "close_on_disconnect": is_truthy_value(params.get("close_on_disconnect", False)),
            "active_session_lease": lease,
            "cols": cols,
            "created_at": now,
            "edit_snapshots": {},
            "explicit_cwd": explicit_cwd,
            "history": history,
            "history_lock": threading.Lock(),
            "history_version": 0,
            "image_counter": 0,
            "cwd": resolved_cwd,
            "inflight_turn": None,
            "last_active": now,
            "model_override": session_model_override,
            "create_reasoning_override": create_reasoning_override,
            "create_service_tier_override": create_service_tier_override,
            "parent_session_id": parent_session_id,
            "pending_title": title or None,
            "profile_home": str(profile_home) if profile_home is not None else None,
            "running": False,
            "session_key": key,
            "show_reasoning": services.load_show_reasoning(),
            "source": source,
            "slash_worker": None,
            "tool_progress_mode": services.load_tool_progress_mode(),
            "tool_started_at": {},
            "transport": current_transport() or services.stdio_transport,
        }
        services.register_session_cwd(sessions[sid])

    # NOTE: we intentionally do NOT persist a DB row here. Every TUI/desktop
    # launch (and every "New agent" / draft) opens a session here just to paint
    # the composer, so eagerly creating a row left an "Untitled" empty session
    # behind for every launch the user never typed into. The row is now created
    # lazily on the first prompt (see services.ensure_session_db_row + prompt.submit),
    # and the AIAgent's own INSERT-OR-IGNORE persists it on the first turn too.

    # Return the lightweight session immediately so Ink can paint the composer
    # + skeleton panel, then build the real AIAgent just after this response is
    # flushed.  This keeps startup responsive while still hydrating tools/skills
    # without requiring the user to submit a first prompt.
    services.schedule_agent_build(sid)
    services.schedule_session_cap_enforcement()  # trim detached idle sessions over the cap

    return services.ok(
        rid,
        {
            "session_id": sid,
            "stored_session_id": key,
            "message_count": len(history),
            "messages": services.history_to_messages(history),
            "info": {
                # Reflect the per-session model override (desktop composer pick)
                # in the immediate response so the client doesn't briefly clobber
                # its sticky pick with the global default before the deferred
                # build's session.info lands.
                "model": (
                    session_model_override.get("model")
                    if session_model_override
                    else services.resolve_model()
                ),
                **(
                    {"provider": session_model_override["provider"]}
                    if session_model_override and session_model_override.get("provider")
                    else {}
                ),
                "tools": {},
                "skills": {},
                "cwd": services.sessions()[sid]["cwd"],
                "branch": services.git_branch_for_cwd(services.sessions()[sid]["cwd"]),
                "project": services.project_info_for_cwd(services.sessions()[sid]["cwd"]),
                "lazy": True,
                "desktop_contract": services.desktop_backend_contract,
                "profile_name": services.response_profile_name(profile),
            },
        },
    )


def session_list(services: SessionLifecycleServices, rid, params: dict) -> dict:
    with services.profile_db(params) as db:
        if db is None:
            return services.db_unavailable_error(rid, code=5006)
        try:
            # Resume picker should surface human conversation sessions from every
            # user-facing surface — CLI, TUI, all gateway platforms (including new
            # ones not enumerated here), ACP adapter clients, webhook sessions,
            # custom `HERMES_SESSION_SOURCE` values, and older installs with
            # different source labels. We deny-list only the noisy internal
            # sources (``tool`` sub-agent runs and ``kanban`` dispatcher
            # workers) rather than allow-listing a fixed set of platform names
            # that goes stale whenever a new platform is added or a user names
            # their own source.
            deny = frozenset({"kanban", "tool"})

            limit = int(params.get("limit", 200) or 200)
            # Over-fetch modestly so per-source filtering doesn't leave us
            # short; the compression-tip projection in ``list_sessions_rich``
            # can also merge rows.
            fetch_limit = max(limit * 2, 200)
            rows = [
                s
                for s in db.list_sessions_rich(
                    source=None,
                    limit=fetch_limit,
                    order_by_last_active=True,
                    compact_rows=True,
                )
                if (s.get("source") or "").strip().lower() not in deny
            ][:limit]
            return services.ok(
                rid,
                {
                    "sessions": [
                        {
                            "id": s["id"],
                            "title": s.get("title") or "",
                            "preview": s.get("preview") or "",
                            "started_at": s.get("started_at") or 0,
                            "message_count": s.get("message_count") or 0,
                            "source": s.get("source") or "",
                        }
                        for s in rows
                    ]
                },
            )
        except Exception as e:
            return services.err(rid, 5006, str(e))


def session_most_recent(services: SessionLifecycleServices, rid, params: dict) -> dict:
    """Return the most recent human-facing session id, or ``None``.

    Mirrors ``session.list``'s deny-list behaviour (drops ``tool``
    sub-agent rows and ``kanban`` worker rows).  Used by TUI auto-resume when
    ``display.tui_auto_resume_recent`` is on; the field is also handy
    for any CLI tooling that wants "latest session" without paginating
    the full list.

    Contract: a ``{"session_id": null}`` result means "no eligible
    session found right now".  Errors are also folded into that
    null-result shape (and logged) so callers don't have to special-
    case JSON-RPC error envelopes for what is a normal "no answer".

    Honors ``params.profile`` so app-global remote mode lists from the
    focused profile's ``state.db`` (mirrors ``session.resume``).
    """
    with services.profile_db(params) as db:
        if db is None:
            return services.ok(rid, {"session_id": None})
        try:
            deny = frozenset({"kanban", "tool"})
            # Over-fetch by a generous bounded amount so heavy sub-agent
            # users (lots of recent ``tool`` rows) don't get a false
            # "no eligible session" answer.  ``session.list`` uses a
            # similar over-fetch strategy.
            rows = db.list_sessions_rich(
                source=None, limit=200, order_by_last_active=True, compact_rows=True
            )
            for row in rows:
                src = (row.get("source") or "").strip().lower()
                if src in deny:
                    continue
                return services.ok(
                    rid,
                    {
                        "session_id": row.get("id"),
                        "title": row.get("title") or "",
                        "started_at": row.get("started_at") or 0,
                        "source": row.get("source") or "",
                    },
                )
            return services.ok(rid, {"session_id": None})
        except Exception:
            logger.exception("session.most_recent failed")
            return services.ok(rid, {"session_id": None})


def project_facts(services: SessionLifecycleServices, rid, params: dict) -> dict:
    """Structured project facts for a cwd — manifests, package manager, the
    exact verify commands, and context files.

    The same detection the coding-context posture (#43316) bakes into the system
    prompt, exposed so UIs (the desktop verify surface) consume it instead of
    re-sniffing. ``{"facts": null}`` means the cwd isn't a code workspace.
    """
    try:
        return services.ok(rid, {"facts": build_project_facts(params.get("cwd"))})
    except Exception:
        logger.exception("project.facts failed")
        return services.ok(rid, {"facts": None})


def verification_status_rpc(services: SessionLifecycleServices, rid, params: dict) -> dict:
    """Best known coding verification evidence for a cwd/session.

    Read-only consumer of the core ledger. It never runs checks and never
    upgrades targeted evidence into a repository-wide guarantee.
    """
    try:
        return services.ok(
            rid,
            {
                "verification": build_verification_status(
                    session_id=params.get("session_id") or params.get("session_key"),
                    cwd=params.get("cwd"),
                )
            },
        )
    except Exception:
        logger.exception("verification.status failed")
        return services.ok(rid, {"verification": {"status": "unknown", "evidence": None}})


def session_resume(services: SessionLifecycleServices, rid, params: dict) -> dict:
    target = params.get("session_id", "")
    if not target:
        return services.err(rid, 4006, "session_id required")
    try:
        cols = int(params.get("cols", 80))
    except (TypeError, ValueError):
        cols = 80
    # ``profile`` (app-global remote mode): resume a session that lives in another
    # local profile's state.db. None/own profile → the launch profile (unchanged).
    profile = (params.get("profile") or "").strip() or None
    profile_home = services.profile_home(profile)

    # In a profile scope, the agent OWNS a long-lived db handle bound to that
    # profile (do NOT auto-close it here). Otherwise reuse the shared launch db.
    if profile_home is not None:
        from hermes_state import SessionDB

        db = SessionDB(db_path=profile_home / "state.db")
    else:
        db = services.get_db()
    if db is None:
        return services.db_unavailable_error(rid, code=5000)

    found = db.get_session(target)
    if not found:
        found = db.get_session_by_title(target)
        if found:
            target = found["id"]
        elif is_truthy_value(params.get("lazy", False)) and services.child_run_active(target):
            # Race: a watch window opened on a freshly-spawned subagent. The
            # child relays `subagent.start` (which carries child_session_id and
            # triggers the window) BEFORE its first run_conversation() flushes
            # the DB row via _ensure_db_session, so db.get_session(target) is
            # momentarily empty. On slower hosts (notably WSL2, where SQLite +
            # process scheduling widen the gap) the window's resume consistently
            # lands inside this window and used to hard-fail "session not found"
            # — the frontend then 404'd on the REST messages fallback and the
            # window spun forever. The child is provably live (services.child_run_active),
            # so proceed into the lazy branch with empty history; the live mirror
            # streams the whole turn anyway and the row exists by upgrade time.
            found = {}
        else:
            return services.err(rid, 4007, "session not found")

    # Follow the compression-continuation chain to the live tip so a resume on
    # a rotated-out parent id binds to the descendant that actually holds the
    # post-compression turns. Auto-compression ends the session and forks a
    # continuation child; without this, resuming the original id (the desktop's
    # routed id when the chat was opened before it rotated) reloads the parent
    # transcript and the response generated after compression is missing — the
    # "I came back and the reply isn't there" bug on large sessions. Resolving
    # here also re-anchors the fast path below so a still-live rotated session
    # is reused (by its new key) instead of rebuilding a duplicate agent on the
    # stale parent. Skipped for lazy watch windows, which intentionally attach
    # to the exact child branch they were opened on.
    if found and not is_truthy_value(params.get("lazy", False)):
        try:
            tip = db.resolve_resume_session_id(target)
        except Exception:
            tip = target
        if tip and tip != target:
            target = tip
            found = db.get_session(target) or found

    profile_resume_cwd = str(found.get("cwd") or "").strip() or services.profile_configured_cwd(
        profile_home
    )

    def _reuse_live_payload(sid: str, session: dict) -> dict:
        payload = services.live_session_payload(
            sid,
            session,
            cols=cols,
            touch=True,
            transport=current_transport() or services.stdio_transport,
        )
        payload["resumed"] = target
        # A lazy watch session never owns a run loop, so its payload's running
        # flag is always False — overlay the child-run registry so a reconnecting
        # watch window keeps its busy indicator while the child is still mid-run.
        if session.get("agent") is None and services.child_run_active(target):
            payload["running"] = True
            payload["status"] = "streaming"
        return payload

    # Fast path: if the session is already live, reuse it under the lock.
    with services.session_resume_lock:
        live = services.find_live_session_by_key(target)
        if live is not None:
            return services.ok(rid, _reuse_live_payload(*live))

    # Lazy/watch resume: register the live session WITHOUT building an agent.
    # Used by the desktop's subagent windows — the child runs inside the
    # parent's turn, so its window only needs the stored history plus a
    # transport for the child-mirror's live events. Skipping services.make_agent here
    # is what keeps the window cheap while the backend is busy running the
    # delegation. A later prompt.submit upgrades it via _start_agent_build
    # (resume_session_id keeps the upgrade on the stored conversation).
    if is_truthy_value(params.get("lazy", False)):
        sid = uuid.uuid4().hex[:8]
        source = services.resolve_session_source(str(params.get("source") or "").strip() or None)
        lease = None  # claimed lazily on the first turn (_ensure_active_session_slot)
        try:
            db.reopen_session(target)
            # The child's OWN conversation only — include_ancestors would prepend
            # the parent's transcript onto the subagent's branch.
            # repair_alternation: this resume feeds LIVE REPLAY (the loaded
            # history becomes the resumed session record's working conversation),
            # so heal a durable ``user;user`` violation once here instead of
            # re-firing the pre-request repair on every subsequent turn.
            history = db.get_messages_as_conversation(target, repair_alternation=True)
        except Exception as e:
            if lease is not None:
                lease.release()
            return services.err(rid, 5000, f"resume failed: {e}")
        cwd = profile_resume_cwd or services.default_session_cwd()
        record = services.deferred_session_record(
            target,
            cols=cols,
            cwd=cwd,
            history=history,
            lease=lease,
            source=source,
            close_on_disconnect=is_truthy_value(params.get("close_on_disconnect", False)),
            profile_home=profile_home,
            lazy=True,
        )
        if (live := services.claim_or_reuse_live(sid, target, record, lease)) is not None:
            return services.ok(rid, _reuse_live_payload(*live))
        # A delegated child mid-run emits no session events of its own — report
        # its liveness from the relay registry so the window shows a busy turn.
        child_running = services.child_run_active(target)
        # User-visible messages use the VERBATIM display projection (child-only,
        # no ancestors — matching the repaired read above), so model-invisible
        # rows persisted by #65919 (verification candidates collapsed by
        # repair_message_sequence) survive in the watch window just as they do
        # on the eager resume + REST paths. The repaired ``history`` above still
        # feeds live replay. Fall back to it if the display read fails.
        try:
            display_history = db.get_messages_as_conversation(
                target, repair_alternation=False, include_row_ids=True
            )
        except Exception:
            logger.debug("child-watch display projection read failed", exc_info=True)
            display_history = history
        messages = services.history_to_messages(display_history)
        return services.ok(
            rid,
            {
                "session_id": sid,
                "resumed": target,
                "message_count": len(messages),
                "messages": messages,
                "info": services.lazy_resume_info(cwd, profile=profile),
                "inflight": None,
                "running": child_running,
                "session_key": target,
                "started_at": record["created_at"],
                "status": "streaming" if child_running else "idle",
            },
        )

    # Cold resume default: register the live session and read its stored
    # transcript, but build the agent OFF the response path. services.make_agent can
    # block for seconds (MCP discovery, prompt/skill build, AIAgent
    # construction), and every resume caller (desktop + Ink TUI) awaits this RPC
    # before it paints — so building eagerly is the bulk of the multi-second
    # "switching sessions is frozen" latency. Return the full display transcript
    # immediately and pre-warm the agent on a short timer (the same deferred-
    # build contract session.create uses); _sess() also builds on demand if the
    # first prompt beats the timer. A caller that needs the agent built
    # synchronously (e.g. tests of the build race) passes ``eager_build: true``
    # to fall through to the eager path below. Distinct from the lazy/watch
    # branch above: a normal resume restores the full ancestor history and the
    # session's persisted runtime identity, and is a real (upgradable) session.
    if not is_truthy_value(params.get("eager_build", False)):
        sid = uuid.uuid4().hex[:8]
        source = services.resolve_session_source(str(params.get("source") or "").strip() or None)
        lease = None  # claimed lazily on the first turn (_ensure_active_session_slot)
        # Interactive resume routes approvals/clarify through gateway prompts;
        # the deferred build wires the remaining per-session callbacks.
        services.enable_gateway_prompts()
        try:
            db.reopen_session(target)
            # One lineage SELECT feeds both projections (#67142-adjacent perf,
            # from the desktop audit): the model-fed copy is alternation-repaired
            # (raw_history → sanitize_replay_history → the resumed session's
            # working conversation) and the display copy stays verbatim —
            # inspection/export must show what is actually stored.
            raw_history, display_history = db.get_resume_conversations(target)
        except Exception as e:
            if lease is not None:
                lease.release()
            return services.err(rid, 5000, f"resume failed: {e}")
        # Display keeps the full transcript; the model-fed history drops a
        # dangling/interrupted tool-call tail so a session killed mid-loop does
        # not replay the unanswered call forever (#29086).
        prefix = db.get_ancestor_display_prefix(target)
        history = sanitize_replay_history(raw_history)
        # Restore the model/provider/reasoning/tier this chat last used so the
        # deferred build (and the info below) match the eager path — without them
        # the build drops the provider ("No LLM provider configured").
        overrides = services.stored_session_runtime_overrides(found) or {}
        model_override = overrides.get("model_override") or {}
        cwd = profile_resume_cwd or services.default_session_cwd()
        record = services.deferred_session_record(
            target,
            cols=cols,
            cwd=cwd,
            history=history,
            lease=lease,
            source=source,
            close_on_disconnect=is_truthy_value(params.get("close_on_disconnect", False)),
            display_history_prefix=prefix,
            profile_home=profile_home,
            model_override=overrides.get("model_override"),
            resume_runtime_overrides=overrides or None,
        )
        if (live := services.claim_or_reuse_live(sid, target, record, lease)) is not None:
            return services.ok(rid, _reuse_live_payload(*live))

        services.schedule_agent_build(sid)
        services.schedule_session_cap_enforcement()  # trim detached idle sessions over the cap
        auto_continue = services.maybe_schedule_auto_continue(sid, record, target)

        messages = services.history_to_messages(display_history)
        payload = {
            "session_id": sid,
            "resumed": target,
            "message_count": len(messages),
            "messages": messages,
            "info": services.lazy_resume_info(
                cwd,
                model=model_override.get("model") or "",
                provider=overrides.get("provider_override") or "",
                profile=profile,
            ),
            "inflight": None,
            "running": False,
            "session_key": target,
            "started_at": record["created_at"],
            "status": "idle",
        }
        if auto_continue is not None:
            payload["auto_continue"] = auto_continue
        return services.ok(rid, payload)

    # Build the agent OUTSIDE the lock — services.make_agent can block for seconds
    # (MCP discovery, prompt/skill build, AIAgent construction). Holding
    # services.session_resume_lock across it would stall session.close on the main
    # dispatch thread (it's not a _LONG_HANDLER), blocking fast-path RPCs.
    sid = uuid.uuid4().hex[:8]
    source = services.resolve_session_source(str(params.get("source") or "").strip() or None)
    lease = None  # claimed lazily on the first turn (_ensure_active_session_slot)
    services.enable_gateway_prompts()
    home_token = (
        set_hermes_home_override(str(profile_home)) if profile_home is not None else None
    )
    secret_token = (
        set_secret_scope(build_profile_secret_scope(Path(str(profile_home))))
        if profile_home is not None
        else None
    )
    try:
        db.reopen_session(target)
        # One lineage SELECT feeds both projections (see the interactive resume
        # above): the model-fed copy is alternation-repaired for LIVE REPLAY, the
        # display copy stays verbatim.
        raw_history, display_history = db.get_resume_conversations(target)
        # The display transcript keeps every row so the user still sees their
        # full history.  The model-fed history is sanitized: a session whose
        # last turn died mid-tool-loop persists a dangling assistant(tool_calls)
        # (or interrupted assistant→tool) tail; replaying it makes the model
        # re-issue the unanswered call forever — the permanent-"thinking" stuck
        # session in #29086.  The messaging gateway already strips this; this is
        # the WebUI/TUI resume path picking up the same cleanup.
        display_history_prefix = db.get_ancestor_display_prefix(target)
        history = sanitize_replay_history(raw_history)
        messages = services.history_to_messages(display_history)
        tokens = services.set_session_context(target)
        try:
            # Pass the profile's db so the agent persists turns to the right
            # state.db; home override is active here so config/skills/model
            # resolve to the profile too. Runtime identity is restored from the
            # stored session row so switching chats does not inherit whatever
            # global model another chat last selected.
            stored_runtime_overrides = services.stored_session_runtime_overrides(found)
            agent = services.make_agent(
                sid,
                target,
                session_id=target,
                session_db=db,
                platform_override=source,
                **stored_runtime_overrides,
            )
        finally:
            services.clear_session_context(tokens)
    except Exception as e:
        if lease is not None:
            lease.release()
        return services.err(rid, 5000, f"resume failed: {e}")
    finally:
        if home_token is not None:
            reset_hermes_home_override(home_token)
        if secret_token is not None:
            reset_secret_scope(secret_token)

    # Double-checked locking: another concurrent resume may have created the
    # live session while we were building. Re-check under the lock; if it won,
    # discard our just-built agent and reuse theirs (no worker/poller wired yet).
    with services.session_resume_lock:
        live = services.find_live_session_by_key(target)
        if live is not None:
            try:
                if hasattr(agent, "close"):
                    agent.close()
            except Exception:
                pass
            if lease is not None:
                lease.release()
            other_sid, other_session = live
            payload = services.live_session_payload(
                other_sid,
                other_session,
                cols=cols,
                touch=True,
                transport=current_transport() or services.stdio_transport,
            )
            payload["resumed"] = target
            return services.ok(rid, payload)
        try:
            init_home_token = (
                set_hermes_home_override(str(profile_home))
                if profile_home is not None
                else None
            )
            init_secret_token = (
                set_secret_scope(build_profile_secret_scope(Path(str(profile_home))))
                if profile_home is not None
                else None
            )
            try:
                services.init_session(
                    sid,
                    target,
                    agent,
                    history,
                    cols=cols,
                    cwd=profile_resume_cwd,
                    session_db=db,
                    source=source,
                )
            finally:
                if init_home_token is not None:
                    reset_hermes_home_override(init_home_token)
                if init_secret_token is not None:
                    reset_secret_scope(init_secret_token)
            sessions = services.sessions()
            if sid in sessions:
                if stored_runtime_overrides.get("model_override") is not None:
                    sessions[sid]["model_override"] = stored_runtime_overrides[
                        "model_override"
                    ]
                sessions[sid]["display_history_prefix"] = display_history_prefix
                # Remember the profile home so each turn re-binds HERMES_HOME (the
                # agent persists to its own db, but mid-turn home reads — memory,
                # skills — must resolve to the resumed profile too).
                if profile_home is not None:
                    sessions[sid]["profile_home"] = str(profile_home)
                sessions[sid]["active_session_lease"] = lease
        except Exception as e:
            if lease is not None:
                lease.release()
            return services.err(rid, 5000, f"resume failed: {e}")
        session = services.sessions().get(sid) or {}
    auto_continue = (
        services.maybe_schedule_auto_continue(sid, session, target) if session else None
    )
    payload = {
        "session_id": sid,
        "resumed": target,
        "message_count": len(messages),
        "messages": messages,
        "info": services.session_info(agent, session),
        "inflight": None,
        "running": False,
        "session_key": target,
        "started_at": float(session.get("created_at") or time.time()),
        "status": "idle",
    }
    if auto_continue is not None:
        payload["auto_continue"] = auto_continue
    return services.ok(rid, payload)


def session_cwd_set(services: SessionLifecycleServices, rid, params: dict) -> dict:
    session, err = services.sess_nowait(params, rid)
    if err:
        return err
    if session.get("running"):
        return services.err(rid, 4009, "session busy")
    raw = str(params.get("cwd", "") or "").strip()
    if not raw:
        return services.err(rid, 4016, "cwd required")
    try:
        cwd = services.set_session_cwd(session, raw)
    except ValueError as e:
        return services.err(rid, 4017, str(e))
    agent = session.get("agent")
    info = services.session_info(agent, session) if agent is not None else {
        "cwd": cwd,
        "branch": services.git_branch_for_cwd(cwd),
        "project": services.project_info_for_cwd(cwd),
        "lazy": True,
    }
    services.emit("session.info", params.get("session_id", ""), info)
    return services.ok(rid, info)


def session_active_list(services: SessionLifecycleServices, rid, params: dict) -> dict:
    """Return live TUI sessions in this gateway process.

    Unlike ``session.list`` this is not a historical DB browser: it reports only
    sessions with in-memory agents/workers that the current TUI can switch to
    without closing siblings.
    """
    current = str(params.get("current_session_id") or "")
    try:
        with services.sessions_lock:
            snapshot = list(services.sessions().items())
    except Exception as e:
        return services.err(rid, 5036, f"could not enumerate active sessions: {e}")

    # Liveness filter (#38950): a session whose teardown has begun (``_finalized``)
    # is dead — its agent/worker are being released and it is no longer
    # attachable — but it can briefly remain in ``services.sessions`` until the reaper
    # pops it (the WS grace-reap and idle reaper both set ``_finalized`` inside
    # ``_teardown_session`` before the pop). Counting these inflated the footer's
    # "N sessions" count, which only ever went up until a gateway restart. Drop
    # them here so the count reflects genuinely attachable sessions. We do NOT
    # filter on ``transport is _detached_ws_transport`` (the WS-detached drop
    # sentinel): a detached session is still attachable via a quick reconnect /
    # session.resume until the grace-reap finalizes it, and a standalone
    # ``hermes --tui`` session legitimately rides the real stdio transport and
    # must stay visible.
    # Keep the natural creation/insertion order from ``services.sessions``.  The
    # frontend marks the focused session with ``current``; it should not jump to
    # the top just because the user switched to it.
    rows = [
        services.session_live_item(sid, session, current)
        for sid, session in snapshot
        if not session.get("_finalized")
    ]
    return services.ok(rid, {"sessions": rows})


def session_activate(services: SessionLifecycleServices, rid, params: dict) -> dict:
    """Attach the frontend to an already-live TUI session.

    This intentionally does not close the previously focused session; it merely
    returns enough state for Ink to redraw around another live session id.
    """
    sid = str(params.get("session_id") or "")
    session, err = services.sess_nowait({"session_id": sid}, rid)
    if err:
        return err
    assert session is not None

    return services.ok(
        rid,
        services.live_session_payload(
            sid,
            session,
            touch=True,
            transport=current_transport() or services.stdio_transport,
        ),
    )


def session_delete(services: SessionLifecycleServices, rid, params: dict) -> dict:
    """Delete a stored session and its on-disk transcript files.

    Used by the TUI resume picker (``d`` key) so users can prune old
    sessions without dropping to the CLI.  Refuses to delete a session
    that is currently active in this gateway process — those rows are
    still being written to and removing them out from under the live
    agent corrupts message ordering and trips FK constraints when the
    next message append flushes.

    Honors ``params.profile`` so app-global remote mode deletes from the
    focused profile's ``state.db`` + sessions dir (mirrors ``session.resume``).
    """
    target = params.get("session_id", "")
    if not target:
        return services.err(rid, 4006, "session_id required")
    # Block deletion of any session currently bound to a live TUI session
    # in this process.  The picker hides the active session anyway, but a
    # racing caller could still target it.  Snapshot via ``list(...)``
    # because ``services.sessions`` is mutated by concurrent RPCs on the thread
    # pool — iterating the dict directly can raise ``RuntimeError:
    # dictionary changed size during iteration``.  If even the snapshot
    # raises, fail closed (refuse the delete) rather than fail open.
    try:
        with services.sessions_lock:
            snapshot = list(services.sessions().values())
    except Exception as e:
        return services.err(rid, 5036, f"could not enumerate active sessions: {e}")
    active = {s.get("session_key") for s in snapshot if s.get("session_key")}
    if target in active:
        return services.err(rid, 4023, "cannot delete an active session")
    profile = (params.get("profile") or "").strip() or None
    profile_home = services.profile_home(profile)
    with services.profile_db(params) as db:
        if db is None:
            return services.db_unavailable_error(rid, code=5036)
        if profile_home is not None:
            sessions_dir = Path(profile_home) / "sessions"
        else:
            sessions_dir = get_hermes_home() / "sessions"
        try:
            deleted = db.delete_session(target, sessions_dir=sessions_dir)
        except Exception as e:
            return services.err(rid, 5036, f"delete failed: {e}")
        if not deleted:
            return services.err(rid, 4007, "session not found")
        return services.ok(rid, {"deleted": target})


def session_title(services: SessionLifecycleServices, rid, params: dict) -> dict:
    session, err = services.sess_nowait(params, rid)
    if err:
        return err
    with services.session_db(session) as db:
        if db is None:
            return services.db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        if "title" not in params:
            fallback = session.get("pending_title") or ""
            try:
                resolved_title = db.get_session_title(key) or ""
                if fallback:
                    if db.set_session_title(key, fallback):
                        session["pending_title"] = None
                        resolved_title = fallback
                    else:
                        existing_row = db.get_session(key)
                        existing_title = ((existing_row or {}).get("title") or "").strip()
                        if existing_title == fallback:
                            session["pending_title"] = None
                            resolved_title = fallback
                        elif not resolved_title:
                            resolved_title = fallback
                elif resolved_title:
                    session["pending_title"] = None
            except Exception:
                resolved_title = fallback
            services.emit_session_info_for_session(params.get("session_id", ""), session)
            return services.ok(
                rid,
                {
                    "title": resolved_title,
                    "session_key": key,
                },
            )
        title = (params.get("title", "") or "").strip()
        if not title:
            return services.err(rid, 4021, "title required")
        try:
            if db.set_session_title(key, title):
                session["pending_title"] = None
                services.emit_session_info_for_session(params.get("session_id", ""), session)
                return services.ok(rid, {"pending": False, "title": title})
            # rowcount == 0 can mean "same value" as well as "missing row".
            existing_row = db.get_session(key)
            if existing_row:
                session["pending_title"] = None
                services.emit_session_info_for_session(params.get("session_id", ""), session)
                return services.ok(
                    rid,
                    {
                        "pending": False,
                        "title": (existing_row.get("title") or title),
                    },
                )
            # No row yet (the DB write is deferred to the first prompt so empty
            # drafts don't litter the sidebar). An explicit /title is clear user
            # intent, not an abandoned draft — so persist the row NOW and set the
            # title, mirroring the messaging gateway's _handle_title_command. The
            # old behavior only queued pending_title and relied on the post-turn
            # apply block; if that turn never landed under this session_key the
            # title was silently lost and the sidebar fell back to the message
            # preview. Creating the row up front removes that race entirely. The
            # min-messages sidebar filter keeps a titled 0-message row hidden, so
            # a /title'd-but-never-used draft still doesn't clutter the list.
            services.ensure_session_db_row(session)
            with services.session_db(session) as scoped_db:
                if scoped_db is not None and scoped_db.set_session_title(key, title):
                    session["pending_title"] = None
                    services.emit_session_info_for_session(params.get("session_id", ""), session)
                    return services.ok(rid, {"pending": False, "title": title})
            # Row creation didn't take (DB unavailable, or a concurrent writer) —
            # fall back to queuing so the post-turn apply block can still recover.
            session["pending_title"] = title
            services.emit_session_info_for_session(params.get("session_id", ""), session)
            return services.ok(rid, {"pending": True, "title": title})
        except ValueError as e:
            return services.err(rid, 4022, str(e))
        except Exception as e:
            return services.err(rid, 5007, str(e))


def register(methods: dict[str, Callable[..., dict]], services: SessionLifecycleServices) -> None:
    """Register lifecycle handlers with explicit service-bound closures."""
    methods["session.create"] = lambda rid, params: session_create(services, rid, params)
    methods["session.list"] = lambda rid, params: session_list(services, rid, params)
    methods["session.most_recent"] = lambda rid, params: session_most_recent(services, rid, params)
    methods["project.facts"] = lambda rid, params: project_facts(services, rid, params)
    verification_status_handler = services.profile_scoped(
        lambda rid, params: verification_status_rpc(services, rid, params)
    )
    verification_status_handler.__module__ = __name__
    methods["verification.status"] = verification_status_handler
    methods["session.resume"] = lambda rid, params: session_resume(services, rid, params)
    methods["session.cwd.set"] = lambda rid, params: session_cwd_set(services, rid, params)
    methods["session.active_list"] = lambda rid, params: session_active_list(services, rid, params)
    methods["session.activate"] = lambda rid, params: session_activate(services, rid, params)
    methods["session.delete"] = lambda rid, params: session_delete(services, rid, params)
    methods["session.title"] = lambda rid, params: session_title(services, rid, params)
