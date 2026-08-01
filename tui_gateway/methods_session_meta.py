"""Session metadata and handoff JSON-RPC handlers for the TUI gateway."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable


@dataclass(frozen=True)
class RpcServices:
    ok: Callable[[Any, dict], dict]
    err: Callable[[Any, int, str], dict]
    db_unavailable_error: Callable[..., dict]


@dataclass(frozen=True)
class SessionLookupServices:
    sess_nowait: Callable[[dict, Any], tuple[dict | None, dict | None]]
    get_session: Callable[[str], dict | None]


@dataclass(frozen=True)
class SessionStorageServices:
    session_db: Callable[[dict], AbstractContextManager[Any]]
    ensure_session_db_row: Callable[[dict], None]


@dataclass(frozen=True)
class LlmOneShotServices:
    run_oneshot: Callable[..., str]
    main_runtime_from_agent: Callable[[Any], dict | None]
    warn: Callable[[str, Any], None]


@dataclass(frozen=True)
class HandoffServices:
    platform_from_name: Callable[[str], Any]
    load_gateway_config: Callable[[], Any]


@dataclass(frozen=True)
class UsageServices:
    session_usage_snapshot: Callable[[dict | None], dict]
    get_usage: Callable[[Any], dict]
    metadata_mirror: Callable[[dict | None], dict]
    nous_credits_lines: Callable[[], list[str]]
    compute_session_context_breakdown: Callable[[Any, list[dict]], dict]


@dataclass(frozen=True)
class SessionMetaServices:
    rpc: RpcServices
    lookup: SessionLookupServices
    storage: SessionStorageServices
    llm: LlmOneShotServices
    handoff: HandoffServices
    usage: UsageServices


def default_session_meta_services(
    *,
    ok: Callable[[Any, dict], dict],
    err: Callable[[Any, int, str], dict],
    db_unavailable_error: Callable[..., dict],
    sess_nowait: Callable[[dict, Any], tuple[dict | None, dict | None]],
    get_session: Callable[[str], dict | None],
    session_db: Callable[[dict], AbstractContextManager[Any]],
    ensure_session_db_row: Callable[[dict], None],
    session_usage_snapshot: Callable[[dict | None], dict],
    get_usage: Callable[[Any], dict],
    metadata_mirror: Callable[[dict | None], dict],
    main_runtime_from_agent: Callable[[Any], dict | None],
    warn: Callable[[str, Any], None],
) -> SessionMetaServices:
    """Build production services with lazy imports for monkeypatch-friendly tests."""

    def run_oneshot(**kwargs: Any) -> str:
        from agent.oneshot import run_oneshot as service

        return service(**kwargs)

    def platform_from_name(name: str) -> Any:
        from gateway.config import Platform

        return Platform(name)

    def load_gateway_config() -> Any:
        from gateway.config import load_gateway_config as service

        return service()

    def nous_credits_lines() -> list[str]:
        from agent.account_usage import nous_credits_lines as service

        return service()

    def compute_session_context_breakdown(agent: Any, history: list[dict]) -> dict:
        from agent.context_breakdown import compute_session_context_breakdown as service

        return service(agent, history)

    return SessionMetaServices(
        rpc=RpcServices(
            ok=ok,
            err=err,
            db_unavailable_error=db_unavailable_error,
        ),
        lookup=SessionLookupServices(
            sess_nowait=sess_nowait,
            get_session=get_session,
        ),
        storage=SessionStorageServices(
            session_db=session_db,
            ensure_session_db_row=ensure_session_db_row,
        ),
        llm=LlmOneShotServices(
            run_oneshot=run_oneshot,
            main_runtime_from_agent=main_runtime_from_agent,
            warn=warn,
        ),
        handoff=HandoffServices(
            platform_from_name=platform_from_name,
            load_gateway_config=load_gateway_config,
        ),
        usage=UsageServices(
            session_usage_snapshot=session_usage_snapshot,
            get_usage=get_usage,
            metadata_mirror=metadata_mirror,
            nous_credits_lines=nous_credits_lines,
            compute_session_context_breakdown=compute_session_context_breakdown,
        ),
    )


def message_react(
    rid: Any,
    params: dict,
    *,
    rpc: RpcServices,
    lookup: SessionLookupServices,
    storage: SessionStorageServices,
) -> dict:
    """Set or clear one author's emoji reaction on a persisted message."""

    session, err = lookup.sess_nowait(params, rid)
    if err:
        return err

    newest_role = str(params.get("newest_role") or "").strip()
    row_id = params.get("row_id")
    if row_id is None and newest_role not in {"user", "assistant"}:
        return rpc.err(rid, 4023, "row_id or newest_role required")

    emoji = params.get("emoji")
    if emoji is not None:
        emoji = str(emoji).strip()
        if not emoji:
            return rpc.err(rid, 4024, "emoji must be a non-empty string or null")

    author = str(params.get("author") or "user").strip()
    if author not in {"user", "agent"}:
        return rpc.err(rid, 4025, "author must be 'user' or 'agent'")

    with storage.session_db(session) as db:
        if db is None:
            return rpc.db_unavailable_error(rid, code=5007)
        try:
            if row_id is None:
                row_id = db.latest_message_row_id(
                    session["session_key"], role=newest_role
                )
                if row_id is None:
                    return rpc.err(rid, 4040, "no message to react to yet")
            reactions = db.set_message_reaction(
                session["session_key"], int(row_id), emoji, author=author
            )
        except Exception as exc:
            return rpc.err(rid, 5007, str(exc))

    if reactions is None:
        return rpc.err(rid, 4040, "message not found in this session")

    return rpc.ok(rid, {"row_id": int(row_id), "reactions": reactions})


def llm_oneshot(
    rid: Any,
    params: dict,
    *,
    rpc: RpcServices,
    lookup: SessionLookupServices,
    llm: LlmOneShotServices,
) -> dict:
    """Run a single stateless LLM request outside any conversation."""

    template = (params.get("template") or "").strip() or None
    instructions = params.get("instructions") or ""
    user_input = params.get("input") or ""
    variables = params.get("variables") if isinstance(params.get("variables"), dict) else {}
    task = (params.get("task") or "title_generation").strip() or "title_generation"

    try:
        max_tokens = int(params.get("max_tokens") or 1024)
    except (TypeError, ValueError):
        max_tokens = 1024
    temperature = params.get("temperature")
    if temperature is not None:
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            temperature = None

    if not template and not str(instructions).strip() and not str(user_input).strip():
        return rpc.err(rid, 4030, "llm.oneshot requires a template or instructions/input")

    session = lookup.get_session(params.get("session_id") or "")
    main_runtime = llm.main_runtime_from_agent(session.get("agent")) if session else None

    try:
        text = llm.run_oneshot(
            instructions=instructions,
            user_input=user_input,
            template=template,
            variables=variables,
            task=task,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.3,
            main_runtime=main_runtime,
        )
    except KeyError as exc:
        return rpc.err(rid, 4031, str(exc))
    except ValueError as exc:
        return rpc.err(rid, 4032, str(exc))
    except Exception as exc:
        llm.warn("llm.oneshot failed: %s", exc)
        return rpc.err(rid, 5030, f"one-shot generation failed: {exc}")

    return rpc.ok(rid, {"text": text})


def handoff_request(
    rid: Any,
    params: dict,
    *,
    rpc: RpcServices,
    lookup: SessionLookupServices,
    storage: SessionStorageServices,
    handoff: HandoffServices,
) -> dict:
    """Queue a handoff of this session to a messaging platform."""

    session, err = lookup.sess_nowait(params, rid)
    if err:
        return err
    if session.get("running"):
        return rpc.err(
            rid,
            4009,
            "session busy — wait for the current turn to finish, then retry the handoff",
        )

    platform_name = (params.get("platform", "") or "").strip().lower()
    if not platform_name:
        return rpc.err(rid, 4023, "platform required")

    try:
        platform = handoff.platform_from_name(platform_name)
    except (ValueError, KeyError):
        return rpc.err(rid, 4024, f"unknown platform '{platform_name}'")
    except Exception as exc:
        return rpc.err(rid, 5021, f"could not load gateway config: {exc}")
    try:
        gw_config = handoff.load_gateway_config()
    except Exception as exc:
        return rpc.err(rid, 5021, f"could not load gateway config: {exc}")
    pcfg = gw_config.platforms.get(platform)
    if not pcfg or not pcfg.enabled:
        return rpc.err(
            rid,
            4025,
            f"platform '{platform_name}' is not configured/enabled in the gateway",
        )
    home = gw_config.get_home_channel(platform)
    if not home or not home.chat_id:
        return rpc.err(
            rid,
            4026,
            f"no home channel configured for {platform_name} — set one with "
            "/sethome on the destination chat first",
        )

    storage.ensure_session_db_row(session)

    with storage.session_db(session) as db:
        if db is None:
            return rpc.db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        try:
            if not db.get_session(key):
                db.set_session_title(key, f"handoff-{key[:8]}")
            ok = db.request_handoff(key, platform_name)
        except Exception as exc:
            return rpc.err(rid, 5007, str(exc))

    if not ok:
        return rpc.err(
            rid,
            4027,
            "session is already in flight for handoff — wait for it to settle, then retry",
        )
    return rpc.ok(
        rid,
        {
            "queued": True,
            "session_key": key,
            "platform": platform_name,
            "home_name": home.name,
        },
    )


def handoff_state(
    rid: Any,
    params: dict,
    *,
    rpc: RpcServices,
    lookup: SessionLookupServices,
    storage: SessionStorageServices,
) -> dict:
    """Poll the handoff state for a session."""

    session, err = lookup.sess_nowait(params, rid)
    if err:
        return err
    with storage.session_db(session) as db:
        if db is None:
            return rpc.db_unavailable_error(rid, code=5007)
        record = db.get_handoff_state(session["session_key"])

    record = record or {}
    return rpc.ok(
        rid,
        {
            "state": record.get("state") or "",
            "platform": record.get("platform") or "",
            "error": record.get("error") or "",
        },
    )


def handoff_fail(
    rid: Any,
    params: dict,
    *,
    rpc: RpcServices,
    lookup: SessionLookupServices,
    storage: SessionStorageServices,
) -> dict:
    """Mark an in-flight handoff as failed so the user can retry."""

    session, err = lookup.sess_nowait(params, rid)
    if err:
        return err
    reason = str(params.get("error") or "handoff failed").strip()[:500]
    with storage.session_db(session) as db:
        if db is None:
            return rpc.db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        record = db.get_handoff_state(key) or {}
        state = record.get("state") or ""
        if state in {"pending", "running"}:
            db.fail_handoff(key, reason)
            return rpc.ok(rid, {"failed": True, "state": "failed"})

    return rpc.ok(rid, {"failed": False, "state": state})


def session_usage(
    rid: Any,
    params: dict,
    *,
    rpc: RpcServices,
    lookup: SessionLookupServices,
    usage: UsageServices,
) -> dict:
    session, err = lookup.sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    payload: dict = usage.session_usage_snapshot(session)
    if agent is None and not payload:
        payload = {"calls": 0, "input": 0, "output": 0, "total": 0}
    try:
        credits = usage.nous_credits_lines()
        if credits:
            payload["credits_lines"] = credits
    except Exception:
        pass
    return rpc.ok(rid, payload)


def session_context_breakdown(
    rid: Any,
    params: dict,
    *,
    rpc: RpcServices,
    lookup: SessionLookupServices,
    usage: UsageServices,
) -> dict:
    session, err = lookup.sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    if agent is None:
        payload = usage.session_usage_snapshot(session) or usage.get_usage(None)
        return rpc.ok(
            rid,
            {
                "categories": [],
                "context_max": payload.get("context_max", 0) or 0,
                "context_percent": payload.get("context_percent", 0) or 0,
                "context_used": payload.get("context_used", 0) or 0,
                "estimated_total": payload.get("context_used", 0) or payload.get("total", 0) or 0,
                "model": usage.metadata_mirror(session).get("model", ""),
            },
        )
    with session["history_lock"]:
        history = list(session.get("history", []))
    try:
        payload = usage.compute_session_context_breakdown(agent, history)
    except Exception as exc:
        return rpc.err(rid, 5000, f"Could not compute context breakdown: {exc}")
    return rpc.ok(rid, payload)


def register(
    methods: dict[str, Callable[[Any, dict], dict]],
    *,
    services: SessionMetaServices,
) -> None:
    """Register ordinary session metadata callables under their JSON-RPC names."""

    methods["message.react"] = partial(
        message_react,
        rpc=services.rpc,
        lookup=services.lookup,
        storage=services.storage,
    )
    methods["llm.oneshot"] = partial(
        llm_oneshot,
        rpc=services.rpc,
        lookup=services.lookup,
        llm=services.llm,
    )
    methods["handoff.request"] = partial(
        handoff_request,
        rpc=services.rpc,
        lookup=services.lookup,
        storage=services.storage,
        handoff=services.handoff,
    )
    methods["handoff.state"] = partial(
        handoff_state,
        rpc=services.rpc,
        lookup=services.lookup,
        storage=services.storage,
    )
    methods["handoff.fail"] = partial(
        handoff_fail,
        rpc=services.rpc,
        lookup=services.lookup,
        storage=services.storage,
    )
    methods["session.usage"] = partial(
        session_usage,
        rpc=services.rpc,
        lookup=services.lookup,
        usage=services.usage,
    )
    methods["session.context_breakdown"] = partial(
        session_context_breakdown,
        rpc=services.rpc,
        lookup=services.lookup,
        usage=services.usage,
    )
