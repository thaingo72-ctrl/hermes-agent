"""Session lifecycle RPC extraction guards."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tui_gateway import methods_session_lifecycle as lifecycle
from tui_gateway import server


LIFECYCLE_METHODS = {
    "session.create",
    "session.list",
    "session.most_recent",
    "project.facts",
    "verification.status",
    "session.resume",
    "session.cwd.set",
    "session.active_list",
    "session.activate",
    "session.delete",
    "session.title",
}


def _unused(*_args, **_kwargs):
    raise AssertionError("unexpected service call")


class _RowsDb:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def list_sessions_rich(self, **_kwargs):
        return [
            {
                "id": f"{self.prefix}-session",
                "title": f"{self.prefix} title",
                "preview": f"{self.prefix} preview",
                "started_at": 1,
                "message_count": 2,
                "source": "tui",
            }
        ]


def _services(prefix: str) -> lifecycle.SessionLifecycleServices:
    @contextmanager
    def profile_db(_params):
        yield _RowsDb(prefix)

    return lifecycle.SessionLifecycleServices(
        child_run_active=_unused,
        claim_or_reuse_live=_unused,
        clear_session_context=_unused,
        coerce_seed_history=_unused,
        completion_cwd=_unused,
        db_unavailable_error=_unused,
        default_session_cwd=_unused,
        deferred_session_record=_unused,
        desktop_backend_contract=0,
        emit=_unused,
        emit_session_info_for_session=_unused,
        enable_gateway_prompts=_unused,
        ensure_session_db_row=_unused,
        err=lambda rid, code, msg: {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}},
        find_live_session_by_key=_unused,
        get_db=_unused,
        git_branch_for_cwd=_unused,
        history_to_messages=_unused,
        init_session=_unused,
        lazy_resume_info=_unused,
        live_session_payload=_unused,
        load_show_reasoning=_unused,
        load_tool_progress_mode=_unused,
        make_agent=_unused,
        maybe_schedule_auto_continue=_unused,
        new_session_key=_unused,
        ok=lambda rid, result: {"jsonrpc": "2.0", "id": rid, "result": result},
        profile_configured_cwd=_unused,
        profile_db=profile_db,
        profile_home=_unused,
        profile_scoped=lambda handler: handler,
        project_info_for_cwd=_unused,
        register_session_cwd=_unused,
        resolve_model=_unused,
        resolve_session_source=_unused,
        response_profile_name=_unused,
        schedule_agent_build=_unused,
        schedule_session_cap_enforcement=_unused,
        sess_nowait=_unused,
        session_db=_unused,
        session_info=_unused,
        session_live_item=lambda sid, session, current: {
            "id": sid,
            "current": sid == current,
            "title": session.get("pending_title") or "",
        },
        session_resume_lock=threading.Lock(),
        sessions=lambda: {},
        sessions_lock=threading.Lock(),
        set_session_context=_unused,
        set_session_cwd=_unused,
        stdio_transport=None,
        stored_session_runtime_overrides=_unused,
    )


def test_session_lifecycle_handlers_are_owned_by_lifecycle_module():
    for name in LIFECYCLE_METHODS:
        assert server._methods[name].__module__ == "tui_gateway.methods_session_lifecycle"


def test_session_lifecycle_services_are_frozen_and_handlers_accept_services():
    services = _services("frozen")
    with pytest.raises(FrozenInstanceError):
        services.sessions = {}

    for handler_name in (
        "session_create",
        "session_list",
        "session_most_recent",
        "project_facts",
        "verification_status_rpc",
        "session_resume",
        "session_cwd_set",
        "session_active_list",
        "session_activate",
        "session_delete",
        "session_title",
    ):
        params = list(inspect.signature(getattr(lifecycle, handler_name)).parameters)
        assert params[:3] == ["services", "rid", "params"]


def test_session_lifecycle_registration_keeps_service_instances_isolated_under_concurrency():
    methods_a: dict[str, object] = {}
    methods_b: dict[str, object] = {}
    lifecycle.register(methods_a, _services("a"))
    lifecycle.register(methods_b, _services("b"))

    def call(methods, rid):
        return methods["session.list"](rid, {"limit": 1})["result"]["sessions"][0]["id"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for index in range(100):
            methods = methods_a if index % 2 == 0 else methods_b
            futures.append(executor.submit(call, methods, str(index)))
        results = [future.result() for future in futures]

    assert set(results[0::2]) == {"a-session"}
    assert set(results[1::2]) == {"b-session"}
