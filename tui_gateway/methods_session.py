"""Session / delegation / spawn-tree / billing / pet JSON-RPC handlers (moved verbatim from server.py).

Handler bodies are byte-identical to their pre-split server.py form; they
are rebound onto server.py's globals at install time — see method_ctx.py.
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped




@method("message.react")
def _(rid, params: dict) -> dict:
    """Set or clear one author's emoji reaction on a persisted message.

    iOS Tapback semantics, enforced in the DB layer: one reaction per author
    per message, re-sending the same emoji retracts it. ``emoji: null`` clears
    unconditionally. ``row_id`` is the durable ``messages.id`` forwarded by
    ``_history_to_messages`` — the renderer's own message ids are ephemeral.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err

    # A live message hasn't round-tripped through a resume, so the desktop has
    # no durable row id for it yet. It can instead name the ROLE whose newest
    # row it means — which is the message the user just reacted to.
    newest_role = str(params.get("newest_role") or "").strip()
    row_id = params.get("row_id")
    if row_id is None and newest_role not in {"user", "assistant"}:
        return _err(rid, 4023, "row_id or newest_role required")

    emoji = params.get("emoji")
    if emoji is not None:
        emoji = str(emoji).strip()
        if not emoji:
            return _err(rid, 4024, "emoji must be a non-empty string or null")

    author = str(params.get("author") or "user").strip()
    if author not in {"user", "agent"}:
        return _err(rid, 4025, "author must be 'user' or 'agent'")

    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        try:
            if row_id is None:
                row_id = db.latest_message_row_id(
                    session["session_key"], role=newest_role
                )
                if row_id is None:
                    return _err(rid, 4040, "no message to react to yet")
            reactions = db.set_message_reaction(
                session["session_key"], int(row_id), emoji, author=author
            )
        except Exception as e:
            return _err(rid, 5007, str(e))

    if reactions is None:
        return _err(rid, 4040, "message not found in this session")

    return _ok(rid, {"row_id": int(row_id), "reactions": reactions})


@method("llm.oneshot")
def _(rid, params: dict) -> dict:
    """Run a single stateless LLM request outside any conversation.

    Generic helper for small generative chores (e.g. a commit message from a
    diff). Accepts either a named ``template`` + ``variables`` or an explicit
    ``instructions`` / ``input`` pair. When ``session_id`` resolves to a live
    session the call inherits that agent's model; otherwise it uses the
    configured auxiliary ``task`` backend. Never mutates session history, so
    prompt caching is untouched.
    """
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
        return _err(rid, 4030, "llm.oneshot requires a template or instructions/input")

    # Optional: inherit the live session's model (no error if absent).
    session = _sessions.get(params.get("session_id") or "")
    main_runtime = _main_runtime_from_agent(session.get("agent")) if session else None

    try:
        from agent.oneshot import run_oneshot

        text = run_oneshot(
            instructions=instructions,
            user_input=user_input,
            template=template,
            variables=variables,
            task=task,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.3,
            main_runtime=main_runtime,
        )
    except KeyError as e:
        return _err(rid, 4031, str(e))
    except ValueError as e:
        return _err(rid, 4032, str(e))
    except Exception as e:
        logger.warning("llm.oneshot failed: %s", e)
        return _err(rid, 5030, f"one-shot generation failed: {e}")

    return _ok(rid, {"text": text})


@method("handoff.request")
def _(rid, params: dict) -> dict:
    """Queue a handoff of this session to a messaging platform.

    Desktop parity with the CLI ``/handoff`` command: we only write
    ``handoff_state='pending'`` onto the persisted session row. The actual
    transfer is performed by the separate ``hermes gateway`` process, whose
    ``_handoff_watcher`` claims the row, re-binds the session to the platform's
    home channel, and forges a synthetic turn. The desktop then polls
    ``handoff.state`` for the terminal result.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(
            rid,
            4009,
            "session busy — wait for the current turn to finish, then retry the handoff",
        )

    platform_name = (params.get("platform", "") or "").strip().lower()
    if not platform_name:
        return _err(rid, 4023, "platform required")

    # Validate against the live gateway config — an unconfigured platform or a
    # missing home channel would leave the handoff pending forever, so reject
    # up front with a clear, actionable message (mirrors cli.py).
    try:
        from gateway.config import Platform, load_gateway_config
    except Exception as e:  # pragma: no cover — gateway pkg always ships
        return _err(rid, 5021, f"could not load gateway config: {e}")
    try:
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        return _err(rid, 4024, f"unknown platform '{platform_name}'")
    try:
        gw_config = load_gateway_config()
    except Exception as e:
        return _err(rid, 5021, f"could not load gateway config: {e}")
    pcfg = gw_config.platforms.get(platform)
    if not pcfg or not pcfg.enabled:
        return _err(
            rid,
            4025,
            f"platform '{platform_name}' is not configured/enabled in the gateway",
        )
    home = gw_config.get_home_channel(platform)
    if not home or not home.chat_id:
        return _err(
            rid,
            4026,
            f"no home channel configured for {platform_name} — set one with "
            "/sethome on the destination chat first",
        )

    # The watcher transfers a persisted DB row, so make sure one exists even
    # for a brand-new empty chat (mirrors the CLI's set_session_title stub).
    _ensure_session_db_row(session)

    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        try:
            if not db.get_session(key):
                db.set_session_title(key, f"handoff-{key[:8]}")
            ok = db.request_handoff(key, platform_name)
        except Exception as e:
            return _err(rid, 5007, str(e))

    if not ok:
        return _err(
            rid,
            4027,
            "session is already in flight for handoff — wait for it to settle, then retry",
        )
    return _ok(
        rid,
        {
            "queued": True,
            "session_key": key,
            "platform": platform_name,
            "home_name": home.name,
        },
    )


@method("handoff.state")
def _(rid, params: dict) -> dict:
    """Poll the handoff state for a session.

    Returns ``{state, platform, error}`` where ``state`` is one of
    ``pending|running|completed|failed`` (or empty when no handoff record
    exists). Desktop polls this after ``handoff.request``.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        record = db.get_handoff_state(session["session_key"])

    record = record or {}
    return _ok(
        rid,
        {
            "state": record.get("state") or "",
            "platform": record.get("platform") or "",
            "error": record.get("error") or "",
        },
    )


@method("handoff.fail")
def _(rid, params: dict) -> dict:
    """Mark an in-flight handoff as failed so the user can retry.

    Desktop calls this when its bounded poll times out. Only pending/running
    rows are changed so a late success from the gateway watcher is not clobbered.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    reason = str(params.get("error") or "handoff failed").strip()[:500]
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        record = db.get_handoff_state(key) or {}
        state = record.get("state") or ""
        if state in {"pending", "running"}:
            db.fail_handoff(key, reason)
            return _ok(rid, {"failed": True, "state": "failed"})

    return _ok(rid, {"failed": False, "state": state})


@method("session.usage")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    usage: dict = _session_usage_snapshot(session)
    if agent is None and not usage:
        usage = {"calls": 0, "input": 0, "output": 0, "total": 0}
    # Nous credits block — agent-independent (a portal fetch), so it shows even
    # with zero API calls or on a resumed session. The TUI /usage panel renders
    # these lines regardless of `calls`. Fail-open: [] when not logged into Nous
    # or on any portal hiccup.
    try:
        from agent.account_usage import nous_credits_lines

        credits = nous_credits_lines()
        if credits:
            usage["credits_lines"] = credits
    except Exception:
        pass
    return _ok(rid, usage)


@method("session.context_breakdown")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    if agent is None:
        usage = _session_usage_snapshot(session) or _get_usage(None)
        return _ok(
            rid,
            {
                "categories": [],
                "context_max": usage.get("context_max", 0) or 0,
                "context_percent": usage.get("context_percent", 0) or 0,
                "context_used": usage.get("context_used", 0) or 0,
                "estimated_total": usage.get("context_used", 0) or usage.get("total", 0) or 0,
                "model": _metadata_mirror(session).get("model", ""),
            },
        )
    with session["history_lock"]:
        history = list(session.get("history", []))
    try:
        from agent.context_breakdown import compute_session_context_breakdown

        payload = compute_session_context_breakdown(agent, history)
    except Exception as exc:
        return _err(rid, 5000, f"Could not compute context breakdown: {exc}")
    return _ok(rid, payload)


@method("pet.info")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Return the active petdex pet for surfaces that render sprites.

    Shared by the desktop (canvas) and the TUI (half-block). Carries the
    spritesheet bytes (base64) plus the engine's frame geometry + state-row
    taxonomy so the renderer is a thin, framework-native consumer. The
    activity→state decision is mirrored from ``agent.pet.state`` client-side.

    Agent-independent (reads config + disk), so it works on any session and
    before the agent finishes building. Fail-open: returns ``enabled=False``
    on any error rather than erroring the surface.
    """
    try:
        enabled, pet, scale = _pet_active_selection()

        if not enabled or pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})

        return _ok(rid, {"enabled": True, **_pet_sprite_payload(pet, scale=scale)})
    except Exception as exc:  # noqa: BLE001 - cosmetic, never break the surface
        logger.debug("pet.info failed: %s", exc)
        return _ok(rid, {"enabled": False})


@method("pet.info.meta")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Cheap active-pet metadata used to avoid full payload refreshes."""
    try:
        enabled, pet, scale = _pet_active_selection()
        if not enabled or pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})
        return _ok(
            rid,
            {
                "enabled": True,
                "slug": pet.slug,
                "displayName": pet.display_name,
                "scale": scale,
                "spritesheetRevision": _pet_sheet_revision(pet.spritesheet),
            },
        )
    except Exception as exc:  # noqa: BLE001 - cosmetic, never break the surface
        logger.debug("pet.info.meta failed: %s", exc)
        return _ok(rid, {"enabled": False})


@method("pet.cells")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Return half-block cell frames for one pet state (TUI renderer).

    The TUI can't draw a canvas, so the engine downsamples the spritesheet to
    a grid of half-block cells and the Ink side paints them with native color
    props. Each cell is ``[tr,tg,tb,ta, br,bg,bb,ba]`` (top + bottom pixel).

    Params: ``state`` (idle/run/review/failed/wave/jump), ``cols`` (width).
    Fail-open: ``enabled=False`` on any problem.
    """
    try:
        from agent.pet import constants, render, store
        from agent.pet.render import PetRenderer

        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
            pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
        except Exception:
            pet_cfg = {}

        if not bool(pet_cfg.get("enabled")):
            return _ok(rid, {"enabled": False})

        pet = store.resolve_active_pet(str(pet_cfg.get("slug", "") or ""))
        if pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})

        state = str(params.get("state") or constants.PetState.IDLE.value)
        scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
        cols = int(params.get("cols") or 0) or constants.resolve_cols(scale, pet_cfg.get("unicode_cols", 0))

        # Graphics path: when the TUI is attached to a real TTY (``graphics``)
        # and the terminal speaks the kitty protocol, return a Unicode-
        # placeholder payload for a crisp image instead of half-blocks. Env
        # detection (KITTY_WINDOW_ID / TERM / TERM_PROGRAM) is shared with the
        # Ink process since it spawns us; the dashboard PTY (xterm.js) has no
        # such env, so it falls through to half-blocks automatically. Only
        # kitty is grid-safe in Ink — iTerm/sixel stay on the fallback.
        if params.get("graphics"):
            configured = str(pet_cfg.get("render_mode", "auto") or "auto").lower()
            gmode = render.detect_terminal_graphics() if configured in ("", "auto") else configured
            if gmode == "kitty":
                image_id = render.kitty_image_id(pet.slug)
                # kitty sizes from scaled pixels (_cell_box), so unicode_cols is moot here.
                payload = PetRenderer(
                    str(pet.spritesheet), mode="kitty", scale=scale
                ).kitty_payload(state, image_id=image_id)
                if payload:
                    kcount = len(payload["frames"]) or 1
                    return _ok(
                        rid,
                        {
                            "enabled": True,
                            "slug": pet.slug,
                            "displayName": pet.display_name,
                            "state": state,
                            "graphics": "kitty",
                            "imageId": image_id,
                            "color": render.kitty_color_hex(image_id),
                            "cols": payload["cols"],
                            "rows": payload["rows"],
                            "placeholder": payload["placeholder"],
                            "frames": payload["frames"],
                            "frameMs": constants.LOOP_MS / max(1, kcount),
                            "scale": scale,
                        },
                    )

        renderer = PetRenderer(
            str(pet.spritesheet),
            mode="unicode",
            scale=scale,
            unicode_cols=cols,
        )
        count = renderer.frame_count(state) or 1
        frames = []
        for i in range(count):
            grid = renderer.cells(state, i, cols=cols)
            frames.append(
                [[[*top, *bottom] for (top, bottom) in row] for row in grid]
            )

        return _ok(
            rid,
            {
                "enabled": True,
                "slug": pet.slug,
                "displayName": pet.display_name,
                "state": state,
                "cols": cols,
                "frameMs": constants.LOOP_MS / max(1, count),
                "frames": frames,
                "scale": scale,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.cells failed: %s", exc)
        return _ok(rid, {"enabled": False})


@method("pet.gallery")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """List adoptable pets for the desktop appearance picker.

    Returns the petdex gallery merged with local install state plus the
    current config (active slug + enabled). Agent-independent. Fail-open:
    returns whatever is installed locally if the gallery can't be reached, so
    the picker still works offline.

    Param ``localOnly`` (bool): skip the remote petdex manifest fetch and return
    only locally-installed pets. The desktop loads this first so the user's own
    pets render instantly instead of waiting on the (possibly slow) manifest.
    """
    local_only = bool(params.get("localOnly"))
    try:
        from agent.pet import store

        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
            pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
        except Exception:
            pet_cfg = {}

        installed = {p.slug: p for p in store.installed_pets()}

        gallery: list[dict] = []
        seen: set[str] = set()
        try:
            from agent.pet.manifest import fetch_manifest, prefetch

            # Local-only: skip the network entirely, but kick off a background
            # warm so the follow-up full request usually hits a cached manifest.
            if local_only:
                prefetch()

            for entry in [] if local_only else fetch_manifest():
                seen.add(entry.slug)
                gallery.append(
                    {
                        "slug": entry.slug,
                        "displayName": entry.display_name,
                        "installed": entry.slug in installed,
                        "spritesheetUrl": entry.spritesheet_url,
                        # petdex exposes no popularity metric; "curated" (its
                        # hand-picked/official set, identified by the asset path)
                        # is the closest signal, so the picker can surface it first.
                        "curated": "/curated/" in entry.spritesheet_url,
                        "generated": entry.slug in installed and installed[entry.slug].generated,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - offline: fall back to installed
            logger.debug("pet.gallery manifest fetch failed: %s", exc)

        # Always include locally-installed pets even if the gallery is unreachable.
        for slug, pet in installed.items():
            if slug not in seen:
                gallery.append(
                    {
                        "slug": slug,
                        "displayName": pet.display_name,
                        "installed": True,
                        "spritesheetUrl": "",
                        "generated": pet.generated,
                    }
                )

        return _ok(
            rid,
            {
                "enabled": bool(pet_cfg.get("enabled")),
                "active": str(pet_cfg.get("slug", "") or ""),
                "pets": gallery,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.gallery failed: %s", exc)
        return _ok(rid, {"enabled": False, "active": "", "pets": []})


@method("pet.select")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Adopt a pet from the desktop picker: install (if needed) + activate.

    Params: ``slug`` (required). Writes ``display.pet.*`` to config and returns
    ``{ok, slug, displayName}``. The surface re-pulls ``pet.info`` to render it.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        from agent.pet import store
        from agent.pet.manifest import ManifestError
        from hermes_cli.pets import _set_active

        try:
            pet = store.install_pet(slug)
        except (store.PetStoreError, ManifestError) as exc:
            return _err(rid, 5031, f"could not adopt '{slug}': {exc}")
        _set_active(slug)
        return _ok(rid, {"ok": True, "slug": slug, "displayName": pet.display_name})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.select failed: %s", exc)
        return _err(rid, 5031, f"pet.select failed: {exc}")


@method("pet.remove")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Uninstall a pet from the desktop picker (delete its on-disk directory).

    Params: ``slug`` (required). If the removed pet was the active one, the
    display is turned off so nothing tries to render a now-missing sprite.
    Returns ``{ok, slug}`` where ``ok`` reflects whether a directory was deleted.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        from agent.pet import store
        from hermes_cli.pets import _clear_active_if

        removed = store.remove_pet(slug)

        # If that was the active pet, stop surfaces pointing at a deleted sprite.
        try:
            _clear_active_if(slug)
        except Exception as exc:  # noqa: BLE001 - removal already succeeded
            logger.debug("pet.remove config update failed: %s", exc)

        return _ok(rid, {"ok": removed, "slug": slug})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.remove failed: %s", exc)
        return _err(rid, 5031, f"pet.remove failed: {exc}")


@method("pet.export")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Export an installed pet as a re-importable ``.zip`` (pet.json + sprite).

    Params: ``slug`` (required). Returns ``{ok, filename, zipBase64}`` — the
    client decodes the base64 and saves it. Heavy-ish (reads + zips files) but
    small; runs inline.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        import base64

        from agent.pet import store

        filename, data = store.export_pet(slug)
        return _ok(
            rid,
            {"ok": True, "filename": filename, "zipBase64": base64.standard_b64encode(data).decode("ascii")},
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.export failed: %s", exc)
        return _err(rid, 5031, f"pet.export failed: {exc}")


@method("pet.rename")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Rename an installed pet's display name + realign its slug/dir.

    Params: ``slug`` + ``name`` (both required). Lets the generate flow hatch
    with a provisional name and apply the user's chosen name at adopt time.
    Returns ``{ok, slug, displayName}`` with the (possibly new) slug.
    """
    slug = str(params.get("slug") or "").strip()
    name = str(params.get("name") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    if not name:
        return _err(rid, 4004, "missing name")
    try:
        from agent.pet import store

        new_slug = store.rename_pet(slug, name)
        if not new_slug:
            return _err(rid, 5031, "pet.rename failed")

        # The dir may have moved; if the renamed pet was active, follow the slug
        # in config so surfaces don't point at the old (now-missing) directory.
        if new_slug != slug:
            try:
                from hermes_cli.pets import _rename_active_if

                _rename_active_if(slug, new_slug)
            except Exception as exc:  # noqa: BLE001 - rename already succeeded
                logger.debug("pet.rename config update failed: %s", exc)

        return _ok(rid, {"ok": True, "slug": new_slug, "displayName": name})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.rename failed: %s", exc)
        return _err(rid, 5031, f"pet.rename failed: {exc}")


@method("pet.thumb")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Return a small idle-frame PNG (data URI) for one pet — the picker preview.

    Cropped + cached server-side so the renderer gets a same-origin data URL
    instead of a CDN ``<img>`` (which the desktop CSP / R2 hotlink rules break).
    Params: ``slug`` (required), ``url`` (optional petdex spritesheet URL used
    only for not-yet-installed pets). Fail-open: ``{ok: false}`` with no error.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        import base64

        from agent.pet import store

        data = store.thumbnail_png(slug, source_url=str(params.get("url") or ""))
        if not data:
            return _ok(rid, {"ok": False, "slug": slug})

        return _ok(
            rid,
            {
                "ok": True,
                "slug": slug,
                "dataUri": "data:image/png;base64," + base64.standard_b64encode(data).decode("ascii"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.thumb failed: %s", exc)
        return _ok(rid, {"ok": False, "slug": slug})


@method("pet.disable")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Turn the pet off from the desktop picker (``display.pet.enabled=false``)."""
    try:
        from hermes_cli.pets import _set_enabled

        _set_enabled(False)
        return _ok(rid, {"ok": True})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.disable failed: %s", exc)
        return _err(rid, 5031, f"pet.disable failed: {exc}")


@method("pet.scale")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Persist ``display.pet.scale`` from the desktop slider. Params: ``scale``.

    Clamped to the engine bounds. The renderer updates its own ``$petInfo`` for
    instant feedback; this just makes the change durable + visible to the other
    terminal surfaces on their next read.
    """
    try:
        from hermes_cli.pets import set_pet_scale

        scale, err = set_pet_scale(params.get("scale"))
        if err:
            return _err(rid, 4004, err)
        return _ok(rid, {"ok": True, "scale": scale})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.scale failed: %s", exc)
        return _err(rid, 5031, f"pet.scale failed: {exc}")


@method("pet.cancel")
def _(rid, params: dict) -> dict:
    """Signal an in-flight ``pet.generate``/``pet.hatch`` (by token) to stop.

    Best-effort + idempotent: cancelling an unknown/finished token is a no-op.
    Stays off the worker pool so it lands while a heavy generation is occupying
    it. Returns ``{ok: True}``.
    """
    token = str(params.get("token") or "").strip()
    if token:
        _pet_cancel_request(token)
    return _ok(rid, {"ok": True})


@method("pet.generate.status")
def _(rid, params: dict) -> dict:
    """Whether pet generation is possible right now.

    True only when a reference-capable image backend (Nous Portal / OpenRouter /
    OpenAI gpt-image) is configured — the desktop checks this on open so it can
    offer setup instead of a dead prompt. Cheap (config + plugin discovery).
    """
    try:
        from agent.pet.generate.imagegen import (
            GenerationError,
            list_sprite_providers,
            resolve_provider,
        )

        try:
            resolve_provider(require_references=True)
            available = True
        except GenerationError:
            available = False
        try:
            providers = list_sprite_providers()
        except Exception as exc:  # noqa: BLE001 - picker is best-effort
            logger.debug("pet provider list failed: %s", exc)
            providers = []
        return _ok(rid, {"available": available, "providers": providers})
    except Exception as exc:  # noqa: BLE001 - never break the surface
        logger.debug("pet.generate.status failed: %s", exc)
        return _ok(rid, {"available": False, "providers": []})


@method("pet.generate")
def _(rid, params: dict) -> dict:
    """Generate candidate base looks for a new pet (the draft/variant step).

    Params: ``prompt`` (required unless ``referenceImage`` is given), ``count``
    (default 4), ``style`` (default ``auto``), ``referenceImage`` (optional data
    URL — a user photo/reference every draft is grounded on, e.g. to make *their*
    pet). Returns ``{ok, token, drafts:[{index, dataUri}]}`` — the token keys the
    staged base images for a later ``pet.hatch``. Heavy (network): worker pool.
    """
    prompt = str(params.get("prompt") or "").strip()
    ref_raw = str(params.get("referenceImage") or "").strip()
    if not prompt and not ref_raw:
        return _err(rid, 4004, "missing prompt")
    try:
        count = max(1, min(4, int(params.get("count") or 4)))
    except (TypeError, ValueError):
        count = 4
    style = str(params.get("style") or "auto").strip() or "auto"

    try:
        import shutil
        import uuid

        from agent.pet.generate import generate_base_drafts
        from agent.pet.generate.imagegen import GenerationError, resolve_provider

        root = _pet_gen_root()
        _pet_gen_sweep(root)

        # Token up front so each draft can be staged + streamed the moment it
        # lands, instead of the user staring at a blank grid until all N finish.
        token = uuid.uuid4().hex[:12]
        _pet_cancel_arm(token)
        stage = root / token
        stage.mkdir(parents=True, exist_ok=True)

        reference_images = None
        if ref_raw:
            try:
                reference_images = _pet_reference_images_from_data_url(ref_raw, stage)
            except ValueError as exc:
                _pet_cancel_release(token)
                return _err(rid, 4004, str(exc))

        # Optional desktop picker override: resolve the chosen provider up front so
        # a bad/uncredentialed pick fails fast instead of mid-fan-out.
        provider_name = str(params.get("provider") or "").strip()
        sprite = None
        if provider_name:
            try:
                sprite = resolve_provider(require_references=bool(reference_images), prefer=provider_name)
            except GenerationError as exc:
                _pet_cancel_release(token)
                return _err(rid, 5031, str(exc))

        concept = prompt or "a pet based on the reference image"
        out: list[dict] = []

        # Hand the token to the client up front (token-only init event) so a Stop
        # fired before the first draft lands can still target this run.
        try:
            _emit("pet.generate.progress", "", {"token": token, "count": count})
        except Exception as exc:  # noqa: BLE001 - streaming is best-effort
            logger.debug("pet.generate init emit failed: %s", exc)

        def _on_draft(index: int, src) -> None:
            dest = stage / f"draft-{index}.png"
            try:
                shutil.copyfile(src, dest)
                data_uri = _pet_png_data_uri(dest)
            except Exception as exc:  # noqa: BLE001 - skip a bad draft, keep the rest
                logger.debug("pet.generate draft %d failed: %s", index, exc)
                return
            out.append({"index": index, "dataUri": data_uri})
            # Stream this draft to the client so the grid fills in live. Best-
            # effort: a transport hiccup must not abort the generation itself.
            try:
                _emit(
                    "pet.generate.progress",
                    "",
                    {"token": token, "index": index, "dataUri": data_uri, "count": count},
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("pet.generate progress emit failed: %s", exc)

        try:
            generate_base_drafts(
                concept,
                n=count,
                style=style,
                reference_images=reference_images,
                provider=sprite,
                on_draft=_on_draft,
                is_cancelled=lambda: _pet_is_cancelled(token),
            )
        except GenerationError as exc:
            _pet_cancel_release(token)
            return _err(rid, 5031, str(exc))

        cancelled = _pet_is_cancelled(token)
        _pet_cancel_release(token)
        if cancelled:
            return _err(rid, 5031, "generation cancelled")
        if not out:
            return _err(rid, 5031, "generation produced no usable drafts")
        out.sort(key=lambda d: d["index"])
        return _ok(rid, {"ok": True, "token": token, "drafts": out})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.generate failed: %s", exc)
        return _err(rid, 5031, f"pet.generate failed: {exc}")


@method("pet.hatch")
def _(rid, params: dict) -> dict:
    """Turn a chosen base draft into a full pet — installed but NOT yet active.

    Generation is expensive and the result varies, so hatch produces a *preview*
    the surface plays (all frames) before the user commits: the pet is written to
    the store (so it can be rendered + later activated) but the active pet is left
    untouched. Adopt with ``pet.select`` or throw it away with ``pet.remove``.

    Params: ``token`` + ``index`` (from ``pet.generate``), ``name`` (required),
    ``description`` (optional), ``prompt`` (optional concept for row prompts),
    ``style`` (optional). Returns ``{ok, slug, displayName, warnings, pet}`` where
    ``pet`` is the renderer payload. Heavy (network + raster): worker pool.
    """
    token = str(params.get("token") or "").strip()
    # Hatch cancellation rides its own key, not the generation token: hatching a
    # draft mid-generation means pet.generate is still releasing `token`, which
    # would otherwise wipe the arm we set here. Falls back to `token` for clients
    # that don't send one.
    cancel_token = str(params.get("cancelToken") or "").strip() or token
    index = params.get("index", 0)
    name = str(params.get("name") or "").strip()
    if not token:
        return _err(rid, 4004, "missing token")
    if not name:
        return _err(rid, 4004, "missing name")
    try:
        index = int(index)
    except (TypeError, ValueError):
        index = 0

    try:
        from agent.pet import store
        from agent.pet.generate import hatch_pet
        from agent.pet.generate.imagegen import GenerationError, resolve_provider

        base = _pet_gen_root() / token / f"draft-{index}.png"
        if not base.is_file():
            return _err(rid, 4004, "draft expired — generate again")

        # Optional desktop picker override (rows always need reference grounding).
        provider_name = str(params.get("provider") or "").strip()
        sprite = None
        if provider_name:
            try:
                sprite = resolve_provider(require_references=True, prefer=provider_name)
            except GenerationError as exc:
                return _err(rid, 5031, str(exc))

        _pet_cancel_arm(cancel_token)
        slug = store.unique_slug(name)

        def _on_progress(event: str, detail: str) -> None:
            # Row progress is encoded as "<state>:<done>:<total>" so the egg
            # screen can show "Drawing <state>… (n/total)"; other phases
            # (compose, save) pass through as-is. Best-effort streaming.
            payload: dict = {"event": event, "detail": detail}
            if event == "row" and detail.count(":") == 2:
                state, done, total = detail.split(":")
                payload = {"event": "row", "state": state, "done": done, "total": total}
            try:
                _emit("pet.hatch.progress", "", payload)
            except Exception as exc:  # noqa: BLE001
                logger.debug("pet.hatch progress emit failed: %s", exc)

        try:
            result = hatch_pet(
                base_image=base,
                slug=slug,
                display_name=name,
                description=str(params.get("description") or ""),
                concept=str(params.get("prompt") or name),
                style=str(params.get("style") or "auto").strip() or "auto",
                provider=sprite,
                on_progress=_on_progress,
                is_cancelled=lambda: _pet_is_cancelled(cancel_token),
            )
        except GenerationError as exc:
            return _err(rid, 5031, str(exc))
        finally:
            _pet_cancel_release(cancel_token)

        pet = store.load_pet(result.slug)
        payload = _pet_sprite_payload(pet, scale=_pet_config_scale()) if pet else {}
        return _ok(
            rid,
            {
                "ok": True,
                "slug": result.slug,
                "displayName": result.display_name,
                "warnings": result.validation.get("warnings", []),
                "pet": payload,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.hatch failed: %s", exc)
        return _err(rid, 5031, f"pet.hatch failed: {exc}")


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
