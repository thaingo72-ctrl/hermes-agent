"""Session / delegation / spawn-tree / billing / pet JSON-RPC handlers (moved verbatim from server.py).

Handler bodies are byte-identical to their pre-split server.py form; they
are rebound onto server.py's globals at install time — see method_ctx.py.
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped





@method("session.status")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err

    from hermes_constants import display_hermes_home

    key = session.get("session_key") or params.get("session_id") or ""
    agent = session.get("agent")
    meta = {}
    # Prefer the live session's bound profile db, else params.profile, else launch.
    status_params = dict(params or {})
    if not status_params.get("profile") and session.get("profile_home"):
        # profile_home is a path; still allow _session_db via a synthetic session
        pass
    with _session_db(session) as db:
        if db is None:
            # Fall back to ~params.profile naming for not-yet-mapped sessions.
            with _profile_db(params) as db2:
                db = db2
                if db and key:
                    try:
                        meta = db.get_session(key) or {}
                    except Exception:
                        meta = {}
                db = None  # prevent double-use
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

    mirror = _metadata_mirror(session)
    usage = _session_usage_snapshot(session)
    provider = getattr(agent, "provider", None) or mirror.get("provider") or "unknown"
    model = getattr(agent, "model", None) or mirror.get("model") or "(unknown)"
    project = _project_info_for_cwd(_display_session_cwd(session))
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
    return _ok(rid, {"output": "\n".join(lines)})


@method("session.history")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    history = list(session.get("history", []))
    if session.get("session_key"):
        with _session_db(session) as db:
            if db is not None:
                try:
                    history = db.get_messages_as_conversation(
                        session["session_key"], include_ancestors=True
                    )
                except Exception:
                    pass
    return _ok(
        rid,
        {
            "count": len(history),
            "messages": _history_to_messages(history),
        },
    )


@method("session.undo")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    # Reject during an in-flight turn.  If we mutated history while
    # the agent thread is running, prompt.submit's post-run history
    # write would either clobber the undo (version matches) or
    # silently drop the agent's output (version mismatch, see below).
    # Neither is what the user wants — make them /interrupt first.
    if session.get("running"):
        return _err(
            rid, 4009, "session busy — /interrupt the current turn before /undo"
        )
    removed = 0
    with session["history_lock"]:
        history = session.get("history", [])
        # Truncate from the last *real* user turn (no display_kind). Popping
        # only trailing assistant/tool then one user left timeline markers
        # (async_delegation_complete, model_switch, …) as the undo target —
        # so session.undo removed bookkeeping instead of the last exchange.
        # Match list_recent_user_messages / CLI turn counting.
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
    return _ok(rid, {"removed": removed})


@method("session.compress")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    assert session is not None
    if _session_uses_compute_host(session):
        sid = str(params.get("session_id") or "")
        focus_topic = str(params.get("focus_topic", "") or "").strip()
        command = "/compress" + (f" {focus_topic}" if focus_topic else "")
        try:
            ack = _send_compute_host_control(
                sid,
                route_name="session.compress",
                command=command,
                wait=True,
                timeout=120.0,
            )
        except Exception as exc:
            return _err(rid, 5019, f"compute-host compress failed: {exc}")
        if ack.get("type") in {"control.error", "error"}:
            return _err(rid, 4009, str(ack.get("message") or "compute-host compress failed"))
        _apply_compute_host_metadata_mirror(session, ack)
        host_result = ack.get("result")
        if isinstance(host_result, dict):
            # The host owns the isolated session's agent/history, so preserve
            # its structured compression result verbatim. In particular this
            # carries `status: aborted` and `summary.aborted`; flattening the
            # old text-only acknowledgement made Desktop show aborted work as a
            # success toast.
            return _ok(rid, {**host_result, "turn_isolation": True})
        host_info = ack.get("session_info") if isinstance(ack.get("session_info"), dict) else {}
        host_messages = _history_to_messages(ack.get("messages")) if isinstance(ack.get("messages"), list) else []
        # `messages` is returned at top level for the desktop transcript
        # replacement. Keep the host acknowledgement metadata, but do not send
        # the same (potentially large) transcript a second time inside it.
        host_ack = {key: value for key, value in ack.items() if key != "messages"}
        return _ok(
            rid,
            {
                "status": "compressed",
                "turn_isolation": True,
                "host_ack": host_ack,
                "info": host_info,
                "messages": host_messages,
                "usage": host_info.get("usage") if isinstance(host_info.get("usage"), dict) else {},
            },
        )
    session, err = _sess(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(
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
        _agent = session["agent"]
        _sys_prompt = getattr(_agent, "_cached_system_prompt", "") or ""
        _tools = getattr(_agent, "tools", None) or None
        before_tokens = (
            estimate_request_tokens_rough(
                before_messages, system_prompt=_sys_prompt, tools=_tools
            )
            if before_count
            else 0
        )

        if before_count >= 4:
            focus_suffix = f', focus: "{focus_topic}"' if focus_topic else ""
            _status_update(
                sid,
                "compressing",
                f"⠋ compressing {before_count} messages "
                f"(~{before_tokens:,} tok){focus_suffix}…",
            )

        try:
            removed, usage = _compress_session_history(
                session,
                focus_topic,
                approx_tokens=before_tokens,
                before_messages=before_messages,
                history_version=history_version,
            )
            with session["history_lock"]:
                messages = list(session.get("history", []))
            after_count = len(messages)
            # Re-read system prompt + tools after compression — _compress_context
            # may have rebuilt the system prompt (_cached_system_prompt=None).
            _sys_prompt_after = (
                getattr(_agent, "_cached_system_prompt", "") or _sys_prompt
            )
            _tools_after = getattr(_agent, "tools", None) or _tools
            after_tokens = (
                estimate_request_tokens_rough(
                    messages,
                    system_prompt=_sys_prompt_after,
                    tools=_tools_after,
                )
                if after_count
                else 0
            )
            agent = session["agent"]
            _sync_session_key_after_compress(sid, session)
            summary = summarize_manual_compression(
                before_messages,
                messages,
                before_tokens,
                after_tokens,
                compression_state=getattr(agent, "context_compressor", None),
            )
            info = _session_info(agent, session)
            _emit("session.info", sid, info)
            finalize_context_engine_compression_notification(
                agent,
                committed=True,
            )
            return _ok(
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
                    # Keep this identical to session.resume / session.history:
                    # raw tool results can contain large or sensitive payloads
                    # that belong in persisted history, not the transcript
                    # replacement response.
                    "messages": _history_to_messages(messages),
                },
            )
        finally:
            # Always clear the pinned compressing status so the bar
            # reverts to neutral whether compaction succeeded, was a
            # no-op, or raised.
            _status_update(sid, "ready")
    except CompressionLockHeld as e:
        _status_update(sid, "ready")
        from agent.manual_compression_feedback import (
            describe_compression_lock_skip,
        )
        return _ok(rid, {
            "compressed": False,
            "lock_held": True,
            "message": describe_compression_lock_skip(e.holder),
        })
    except Exception as e:
        finalize_context_engine_compression_notification(
            session["agent"],
            committed=False,
        )
        return _err(rid, 5005, str(e))


@method("session.save")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err

    if _session_uses_compute_host(session):
        sid = str(params.get("session_id") or "")
        try:
            ack = _send_compute_host_control(
                sid,
                route_name="session.save",
                wait=True,
            )
        except Exception as exc:
            return _err(rid, 5011, f"compute-host session save failed: {exc}")
        if ack.get("type") in {"control.error", "error"}:
            return _err(rid, 5011, str(ack.get("message") or "compute-host session save failed"))
        result = ack.get("result")
        if not isinstance(result, dict):
            return _err(rid, 5011, "compute-host session save returned an invalid response")
        return _ok(rid, result)

    agent = session["agent"]
    # Mirror the classic CLI /save: snapshot under the Hermes profile home
    # (~/.hermes/sessions/saved/) rather than the project/workspace CWD, and
    # include the system prompt so the export matches the dashboard save.
    saved_dir = get_hermes_home() / "sessions" / "saved"
    try:
        saved_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return _err(rid, 5011, f"failed to create save directory {saved_dir}: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = saved_dir / f"hermes_conversation_{timestamp}.json"

    with session["history_lock"]:
        messages = list(session.get("history", []))

    session_id = getattr(agent, "session_id", None) or session.get("session_key") or ""
    # Prefer the agent's session_start datetime (matches the classic CLI export);
    # fall back to the gateway session's created_at timestamp.
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
        return _ok(rid, {"file": str(path)})
    except Exception as e:
        return _err(rid, 5011, str(e))


@method("session.close")
def _(rid, params: dict) -> dict:
    sid = params.get("session_id", "")
    # Serialize only the ownership claim against session.resume / the orphan
    # reaper. Finalization may run arbitrary plugin/agent cleanup and must not
    # keep every unrelated session.resume waiting behind it.
    with _session_resume_lock:
        session = _pop_session_by_id(sid)
    closed = _teardown_popped_session(session, end_reason="tui_close")
    return _ok(rid, {"closed": closed})


@method("session.branch")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    # Branch must write into the parent's profile-scoped state.db (app-global
    # remote mode). Using the launch handle would orphan branch rows + history.
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5008)
        old_key = session["session_key"]
        with session["history_lock"]:
            history = [dict(msg) for msg in session.get("history", [])]
        if not history:
            return _err(rid, 4008, "nothing to branch — send a message first")
        count = params.get("count")
        if isinstance(count, int) and count > 0:
            history = history[:count]
        new_key = _new_session_key()
        new_sid = uuid.uuid4().hex[:8]
        source = _session_source(session)
        lease = None  # claimed lazily on the first turn (_ensure_active_session_slot)
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
                model=_resolve_model(),
                # Stable _branched_from marker so list_sessions_rich() keeps the
                # branch visible in /resume and /sessions. The TUI branch leaves
                # the parent live (no end_reason='branched'), so the legacy
                # end_reason heuristic never matches it — the marker is the only
                # thing that surfaces TUI branches. See issue #20856.
                model_config={"_branched_from": old_key},
                parent_session_id=old_key,
                cwd=_session_cwd(session),
                # The branch stays on its parent's profile. Explicit stamp (not
                # just the parent-backfill) so it holds even when the parent row
                # predates the profile_name column.
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
                    # Preserve the parent's original message timestamps —
                    # branch copies are history, not new activity (9d73006ad).
                    timestamp=msg.get("timestamp"),
                )
            db.set_session_title(new_key, title)
        except Exception as e:
            if lease is not None:
                lease.release()
            return _err(rid, 5008, f"branch failed: {e}")
    try:
        # Bind the branched AGENT to the parent's profile, mirroring
        # session.create/resume: home override so config/skills/memory resolve
        # to the profile during the build, and the profile's own state.db
        # handle so the live agent's message flushes — and any later
        # compression rotation — persist there. Writing only the row to the
        # parent's db while the agent stayed on the launch handle would
        # recreate the cross-profile split one turn later.
        parent_home = session.get("profile_home")
        branch_db = None
        if parent_home:
            from hermes_state import SessionDB

            branch_db = SessionDB(db_path=Path(parent_home) / "state.db")
        home_token = (
            set_hermes_home_override(parent_home) if parent_home else None
        )
        try:
            tokens = _set_session_context(new_key)
            try:
                agent = _make_agent(
                    new_sid,
                    new_key,
                    session_id=new_key,
                    session_db=branch_db,
                    platform_override=source,
                )
            finally:
                _clear_session_context(tokens)
            _init_session(
                new_sid,
                new_key,
                agent,
                list(history),
                cols=session.get("cols", 80),
                cwd=_session_cwd(session),
                session_db=branch_db,
                source=source,
                profile_home=parent_home,
            )
        finally:
            if home_token is not None:
                reset_hermes_home_override(home_token)
        if new_sid in _sessions:
            _sessions[new_sid]["active_session_lease"] = lease
    except Exception as e:
        if lease is not None:
            lease.release()
        return _err(rid, 5000, f"agent init failed on branch: {e}")
    branched_session = _sessions.get(new_sid)
    return _ok(
        rid,
        {
            "session_id": new_sid,
            "stored_session_id": new_key,
            "title": title,
            "parent": old_key,
            "message_count": len(history),
            "messages": _history_to_messages(history),
            "info": _session_info(agent, branched_session),
        },
    )


@method("session.interrupt")
def _(rid, params: dict) -> dict:
    # Keypress barge-in: stopping the turn also silences its streaming TTS
    # (voice is process-global, so no per-session scoping is needed).
    _tts_stream_stop()
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if _session_uses_compute_host(session):
        sid = str(params.get("session_id") or "")
        if session.get("running"):
            try:
                _get_compute_host_supervisor().interrupt(sid, request_id=f"interrupt-{rid}")
            except Exception as exc:
                return _err(rid, 5019, f"compute-host interrupt failed: {exc}")
        with session["history_lock"]:
            session["_turn_cancel_requested"] = True
            session["queued_prompt"] = None
        _clear_pending(sid)
        try:
            from tools.approval import resolve_gateway_approval

            resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)
        except Exception:
            pass
        return _ok(rid, {"status": "interrupted", "turn_isolation": True})
    session, err = _sess(params, rid)
    if err:
        return err
    # Safety net: if the turn's run thread is already gone but `running` stayed
    # stuck (a crash/desync that skipped the run loop's `finally`), force-clear it
    # so the session can't be permanently bricked at 4009 "session busy" — every
    # send/restore/resume would otherwise reject until a full backend restart.
    # Always tell the agent to interrupt when the session claims a run is active:
    # stale flags are cleared below, and fresh turns clear the interrupt flag at
    # entry. This keeps a stale/missing thread handle from making Stop a no-op.
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
                _clear_inflight_turn(session)

    # Stop = stop the TURN (cooperative interrupt above also kills the in-flight
    # foreground subprocess). Background processes the agent started (dev servers,
    # watchers) are intentionally left running — kill those individually with the
    # "x" on the task row (process.kill). Don't reap them here.
    # Scope the pending-prompt release to THIS session.  A global
    # _clear_pending() would collaterally cancel clarify/sudo/secret
    # prompts on unrelated sessions sharing the same tui_gateway
    # process, silently resolving them to empty strings.
    _clear_pending(params.get("session_id", ""))
    try:
        from tools.approval import resolve_gateway_approval

        resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)
    except Exception:
        pass
    return _ok(rid, {"status": "interrupted"})


@method("delegation.status")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import (
        is_spawn_paused,
        list_active_subagents,
        _get_max_concurrent_children,
        _get_max_spawn_depth,
    )

    return _ok(
        rid,
        {
            "active": list_active_subagents(),
            "paused": is_spawn_paused(),
            "max_spawn_depth": _get_max_spawn_depth(),
            "max_concurrent_children": _get_max_concurrent_children(),
        },
    )


@method("delegation.pause")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import set_spawn_paused

    paused = bool(params.get("paused", True))
    return _ok(rid, {"paused": set_spawn_paused(paused)})


@method("subagent.interrupt")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import interrupt_subagent

    subagent_id = str(params.get("subagent_id") or "").strip()
    if not subagent_id:
        return _err(rid, 4000, "subagent_id required")
    ok = interrupt_subagent(subagent_id)
    return _ok(rid, {"found": ok, "subagent_id": subagent_id})


@method("spawn_tree.save")
def _(rid, params: dict) -> dict:
    session_id = str(params.get("session_id") or "").strip()
    subagents = params.get("subagents") or []
    if not isinstance(subagents, list) or not subagents:
        return _err(rid, 4000, "subagents list required")

    from datetime import datetime

    started_at = params.get("started_at")
    finished_at = params.get("finished_at") or time.time()
    label = str(params.get("label") or "")
    ts = datetime.utcfromtimestamp(float(finished_at)).strftime("%Y%m%dT%H%M%S")
    fname = f"{ts}.json"
    d = _spawn_tree_session_dir(session_id or "default")
    path = d / fname
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

    _append_spawn_tree_index(
        d,
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


@method("spawn_tree.list")
def _(rid, params: dict) -> dict:
    session_id = str(params.get("session_id") or "").strip()
    limit = int(params.get("limit") or 50)
    cross_session = bool(params.get("cross_session"))

    if cross_session:
        root = _spawn_trees_root()
        roots = [p for p in root.iterdir() if p.is_dir()]
    else:
        roots = [_spawn_tree_session_dir(session_id or "default")]

    entries: list[dict] = []
    for d in roots:
        indexed = _read_spawn_tree_index(d)
        if indexed:
            # Skip index entries whose snapshot file was manually deleted.
            entries.extend(
                e for e in indexed if (p := e.get("path")) and Path(p).exists()
            )
            continue

        # Fallback for legacy (pre-index) sessions: full scan.  O(N) reads
        # but only runs once per session until the next save writes the index.
        for p in d.glob("*.json"):
            if p.name == _SPAWN_TREE_INDEX:
                continue
            try:
                stat = p.stat()
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    raw = {}
                subagents = raw.get("subagents") or []
                entries.append(
                    {
                        "path": str(p),
                        "session_id": raw.get("session_id") or d.name,
                        "finished_at": raw.get("finished_at") or stat.st_mtime,
                        "started_at": raw.get("started_at"),
                        "label": raw.get("label") or "",
                        "count": len(subagents) if isinstance(subagents, list) else 0,
                    }
                )
            except OSError:
                continue

    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return _ok(rid, {"entries": entries[:limit]})


@method("spawn_tree.load")
def _(rid, params: dict) -> dict:
    from pathlib import Path

    raw_path = str(params.get("path") or "").strip()
    if not raw_path:
        return _err(rid, 4000, "path required")

    # Reject paths escaping the spawn-trees root.
    root = _spawn_trees_root().resolve()
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


@method("session.steer")
def _(rid, params: dict) -> dict:
    """Inject a user message into the next tool result without interrupting.

    Mirrors AIAgent.steer(). Safe to call while a turn is running — the text
    lands on the last tool result of the next tool batch and the model sees
    it on its next iteration. No interrupt, no new user turn, no role
    alternation violation.
    """
    text = (params.get("text") or "").strip()
    if not text:
        return _err(rid, 4002, "text is required")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    if agent is None or not hasattr(agent, "steer"):
        return _err(rid, 4010, "agent does not support steer")
    try:
        accepted = agent.steer(text)
    except Exception as exc:
        return _err(rid, 5000, f"steer failed: {exc}")
    if accepted:
        # Record the correction on the live turn exactly like session.redirect
        # does. Without this, a resume/reconnect while the turn is running
        # rebuilds the transcript from the inflight snapshot and the steered
        # text has no user bubble — the "my message vanished on reload" loss.
        with session["history_lock"]:
            _record_inflight_correction(session, text)
            session["last_active"] = time.time()
    return _ok(rid, {"status": "queued" if accepted else "rejected", "text": text})


@method("session.redirect")
def _(rid, params: dict) -> dict:
    """Redirect the active model turn while preserving valid work/context."""
    text = (params.get("text") or "").strip()
    if not text:
        return _err(rid, 4002, "text is required")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    # Turn-build window: a fresh turn flips running=True and kicks off an async
    # agent build, so session["agent"] is briefly None. That is not an
    # unsupported runtime — queue the correction server-side so it reaches the
    # model as the next turn, instead of a misleading 4010 the client silently
    # swallows into a lost follow-up.
    if agent is None and session.get("running"):
        _enqueue_prompt(session, text, current_transport() or _stdio_transport)
        session["last_active"] = time.time()
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
            _record_inflight_correction(session, text)
            session["last_active"] = time.time()
    return _ok(
        rid,
        {"status": "redirected" if accepted else "rejected", "text": text},
    )


@method("terminal.resize")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    session["cols"] = int(params.get("cols", 80))
    return _ok(rid, {"cols": session["cols"]})


def register(server) -> None:
    """Bind this module's handlers onto ``server``'s globals and registry."""
    _registry.install(server)
