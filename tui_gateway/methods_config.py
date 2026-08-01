"""Configuration, projects, and setup JSON-RPC handlers.

Handlers receive an explicit dependency context as their first argument and are
registered as ordinary callables by ``method_ctx.HandlerRegistry``.
"""
from .method_ctx import HandlerRegistry
_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

@method('projects.discover_repos')
def _(ctx, rid, params: dict) -> dict:
    """Repos for the desktop overview: scanned-from-disk (cached) ∪ session-derived."""
    try:
        db = ctx._get_db()
        if db is None:
            return ctx._ok(rid, {'repos': []})
        from hermes_cli import projects_db as pdb
        policy = ctx._repo_discovery_policy()
        policy_key = ctx._repo_discovery_policy_key(policy)
        with pdb.connect_closing() as conn:
            pdb.reconcile_discovered_repos_policy(conn, policy_key, preserve_unversioned=ctx._repo_discovery_policy_is_default(policy))
            repos = ctx._discover_repos_payload(db, conn=conn, include_cached=policy['enabled'])
        return ctx._ok(rid, {'repos': repos, 'discovery_policy': policy})
    except Exception as e:
        return ctx._err(rid, 5061, str(e))

@method('projects.record_repos')
def _(ctx, rid, params: dict) -> dict:
    """Persist git repo roots found by the client's filesystem scan, then return
    the merged repo list. The native crawl runs on the desktop (local fs); this
    caches the result so later reads are instant instead of re-walking disk."""
    try:
        from hermes_cli import projects_db as pdb
        policy = ctx._repo_discovery_policy()
        policy_key = ctx._repo_discovery_policy_key(policy)
        incoming_raw = params.get('discovery_policy')
        incoming_policy = ctx._repo_discovery_policy(incoming_raw) if isinstance(incoming_raw, dict) else None
        incoming_matches = incoming_policy is not None and ctx._repo_discovery_policy_key(incoming_policy) == policy_key
        accept_legacy_default = incoming_policy is None and ctx._repo_discovery_policy_is_default(policy)
        pairs: list[tuple[str, str | None]] = []
        for item in params.get('repos') or []:
            if isinstance(item, str):
                pairs.append((item, None))
            elif isinstance(item, dict) and item.get('root'):
                pairs.append((str(item['root']), item.get('label')))
        with pdb.connect_closing() as conn:
            pdb.reconcile_discovered_repos_policy(conn, policy_key, preserve_unversioned=ctx._repo_discovery_policy_is_default(policy))
            accepted = bool(policy['enabled'] and (incoming_matches or accept_legacy_default))
            if accepted:
                pdb.record_discovered_repos(conn, pairs, replace=True, policy_key=policy_key)
            elif not policy['enabled']:
                pdb.clear_discovered_repos(conn, policy_key=policy_key)
        db = ctx._get_db()
        return ctx._ok(rid, {'repos': ctx._discover_repos_payload(db, include_cached=policy['enabled']) if db is not None else [], 'accepted': accepted, 'discovery_policy': policy})
    except Exception as e:
        return ctx._err(rid, 5061, str(e))

@method('projects.tree')
def _(ctx, rid, params: dict) -> dict:
    """Authoritative project overview: project -> repo -> lane structure with
    counts + a few preview sessions per project, plus the flat set of session
    ids claimed by any project (so the desktop excludes them from flat Recents).
    Lanes carry no session rows here; drill-in uses ``projects.project_sessions``.
    """
    try:
        db = ctx._get_db()
        if db is None:
            return ctx._ok(rid, {'projects': [], 'active_id': None, 'scoped_session_ids': []})
        tree, active_id = ctx._build_project_tree(db, preview_limit=int(params.get('preview_limit') or 3), hydrate=False, session_limit=int(params.get('session_limit') or 2000), include_discovered=True)
        return ctx._ok(rid, {'projects': tree['projects'], 'active_id': active_id, 'scoped_session_ids': tree['scoped_session_ids']})
    except Exception as e:
        return ctx._err(rid, 5061, str(e))

@method('projects.project_sessions')
def _(ctx, rid, params: dict) -> dict:
    """Fully hydrated lanes (repo -> lane -> session rows) for one project,
    built from the same authoritative grouping as ``projects.tree`` so ids and
    membership match exactly. Used when the user enters a project."""
    try:
        project_id = str(params.get('project_id') or '')
        if not project_id:
            return ctx._err(rid, 5063, 'project_id required')
        db = ctx._get_db()
        if db is None:
            return ctx._ok(rid, {'project': None})
        tree, _active = ctx._build_project_tree(db, preview_limit=0, hydrate=True, session_limit=int(params.get('session_limit') or 5000), include_discovered=False)
        proj = next((p for p in tree['projects'] if p['id'] == project_id), None)
        return ctx._ok(rid, {'project': proj})
    except Exception as e:
        return ctx._err(rid, 5061, str(e))

@method('config.get')
def _(ctx, rid, params: dict) -> dict:
    key = params.get('key', '')
    if key == 'provider':
        try:
            from hermes_cli.models import list_available_providers, normalize_provider
            model = ctx._resolve_model()
            parts = model.split('/', 1)
            return ctx._ok(rid, {'model': model, 'provider': normalize_provider(parts[0]) if len(parts) > 1 else 'unknown', 'providers': list_available_providers()})
        except Exception as e:
            return ctx._err(rid, 5013, str(e))
    if key == 'profile':
        from hermes_constants import display_hermes_home
        return ctx._ok(rid, {'home': str(ctx._hermes_home), 'display': display_hermes_home()})
    if key == 'project':
        cfg_terminal = ctx._load_cfg().get('terminal') or {}
        raw = str(params.get('cwd', '') or cfg_terminal.get('cwd', '') or '').strip()
        cwd = ctx._completion_cwd({'cwd': raw} if raw else {})
        return ctx._ok(rid, {'cwd': cwd, 'branch': ctx._git_branch_for_cwd(cwd)})
    if key == 'full':
        return ctx._ok(rid, {'config': ctx._load_cfg()})
    if key == 'prompt':
        return ctx._ok(rid, {'prompt': ctx._load_cfg().get('custom_prompt', '')})
    if key == 'skin':
        return ctx._ok(rid, {'value': (ctx._load_cfg().get('display') or {}).get('skin', 'default')})
    if key == 'indicator':
        raw = (ctx._load_cfg().get('display') or {}).get('tui_status_indicator', '')
        norm = str(raw).strip().lower()
        return ctx._ok(rid, {'value': norm if norm in ctx._INDICATOR_STYLES else ctx._INDICATOR_DEFAULT})
    if key == 'personality':
        return ctx._ok(rid, {'value': (ctx._load_cfg().get('display') or {}).get('personality') or 'none'})
    if key == 'reasoning':
        cfg = ctx._load_cfg()
        session = ctx._sessions.get(params.get('session_id', ''))
        reasoning_config = None
        if session is not None:
            if isinstance(session.get('create_reasoning_override'), dict):
                reasoning_config = session.get('create_reasoning_override')
            else:
                agent = session.get('agent')
                agent_reasoning = getattr(agent, 'reasoning_config', None)
                if isinstance(agent_reasoning, dict):
                    reasoning_config = agent_reasoning
        if isinstance(reasoning_config, dict):
            if reasoning_config.get('enabled') is False:
                effort = 'none'
            else:
                effort = str(reasoning_config.get('effort') or 'medium')
        else:
            raw_effort = (cfg.get('agent') or {}).get('reasoning_effort', '')
            if raw_effort is False:
                effort = 'none'
            else:
                effort = str(raw_effort or 'medium')
        display = 'show' if bool((cfg.get('display') or {}).get('show_reasoning', True)) else 'hide'
        return ctx._ok(rid, {'value': effort, 'display': display})
    if key == 'fast':
        session = ctx._sessions.get(params.get('session_id', ''))
        tier = None
        if session is not None:
            agent = session.get('agent')
            if agent is not None:
                tier = getattr(agent, 'service_tier', None)
            elif session.get('create_service_tier_override') is not None:
                tier = session['create_service_tier_override']
        if tier is None:
            tier = ctx._load_service_tier()
        return ctx._ok(rid, {'value': 'fast' if tier == 'priority' else 'normal'})
    if key == 'busy':
        return ctx._ok(rid, {'value': ctx._load_busy_input_mode()})
    if key in {'approval_mode', 'approvals.mode'}:
        try:
            return ctx._ok(rid, {'value': ctx._load_approval_mode()})
        except Exception as e:
            return ctx._err(rid, 5001, str(e))
    if key == 'details_mode':
        allowed_dm = frozenset({'hidden', 'collapsed', 'expanded'})
        raw = str((ctx._load_cfg().get('display') or {}).get('details_mode', 'collapsed') or 'collapsed').strip().lower()
        nv = raw if raw in allowed_dm else 'collapsed'
        return ctx._ok(rid, {'value': nv})
    if key == 'thinking_mode':
        allowed_tm = frozenset({'collapsed', 'truncated', 'full'})
        cfg = ctx._load_cfg()
        raw = str((cfg.get('display') or {}).get('thinking_mode', '') or '').strip().lower()
        if raw in allowed_tm:
            nv = raw
        else:
            dm = str((cfg.get('display') or {}).get('details_mode', 'collapsed') or 'collapsed').strip().lower()
            nv = 'full' if dm == 'expanded' else 'collapsed'
        return ctx._ok(rid, {'value': nv})
    if key == 'density':
        on = bool((ctx._load_cfg().get('display') or {}).get('tui_compact', False))
        return ctx._ok(rid, {'value': 'on' if on else 'off'})
    if key == 'theme':
        display = ctx._load_cfg().get('display')
        raw = str(display.get('tui_theme', 'auto') if isinstance(display, dict) else 'auto').strip().lower()
        return ctx._ok(rid, {'value': raw if raw in {'auto', 'light', 'dark'} else 'auto'})
    if key == 'statusbar':
        display = ctx._load_cfg().get('display')
        raw = display.get('tui_statusbar', 'top') if isinstance(display, dict) else 'top'
        return ctx._ok(rid, {'value': ctx._coerce_statusbar(raw)})
    if key == 'focus':
        display = ctx._load_cfg().get('display')
        on = bool(display.get('focus_view', False)) if isinstance(display, dict) else False
        return ctx._ok(rid, {'value': 'on' if on else 'off', 'tool_progress': ctx._load_tool_progress_mode()})
    if key == 'mouse':
        display = ctx._load_cfg().get('display')
        return ctx._ok(rid, {'value': ctx._display_mouse_tracking(display)})
    if key == 'mtime':
        cfg_path = ctx._hermes_home / 'config.yaml'
        try:
            mtime = cfg_path.stat().st_mtime if cfg_path.exists() else 0
        except Exception:
            return ctx._ok(rid, {'mtime': 0})
        return ctx._ok(rid, {'mtime': mtime, 'mcp_rev': ctx._compute_mcp_rev()})
    return ctx._err(rid, 4002, f'unknown config key: {key}')

@method('config.set')
def _(ctx, rid, params: dict) -> dict:
    key, value = (params.get('key', ''), params.get('value', ''))
    session = ctx._sessions.get(params.get('session_id', ''))
    if key == 'model':
        try:
            if not value:
                return ctx._err(rid, 4002, 'model value required')
            if session:
                from hermes_cli.model_switch import parse_model_switch_args
                if session.get('running'):
                    parsed = parse_model_switch_args(value)
                    try:
                        pending_model = parsed.model_input
                    except Exception:
                        pending_model = str(value)
                    session['pending_model_switch'] = {'raw': value, 'confirm_expensive_model': bool(params.get('confirm_expensive_model', False)), 'display_model': pending_model, 'display_provider': (getattr(parsed, 'explicit_provider', '') or '').strip()}
                    return ctx._ok(rid, {'key': key, 'value': pending_model, 'warning': '', 'confirm_required': False, 'confirm_message': '', 'scope': 'session', 'deferred': True})
                parsed_flags = parse_model_switch_args(value)
                explicit_provider = parsed_flags.explicit_provider
                if session.get('agent') is None and (not explicit_provider.strip()):
                    session_id = params.get('session_id', '')
                    ctx._start_agent_build(session_id, session)
                    init_err = ctx._wait_agent(session, rid)
                    if init_err:
                        return init_err
                    if session.get('agent') is None:
                        return ctx._err(rid, 5032, 'agent initialization failed')
                result = ctx._apply_model_switch(params.get('session_id', ''), session, value, confirm_expensive_model=bool(params.get('confirm_expensive_model', False)), parsed_flags=parsed_flags)
            else:
                result = ctx._apply_model_switch('', {'agent': None}, value, confirm_expensive_model=bool(params.get('confirm_expensive_model', False)))
            return ctx._ok(rid, {'key': key, 'value': result['value'], 'warning': result['warning'], 'confirm_required': result.get('confirm_required', False), 'confirm_message': result.get('confirm_message', ''), 'scope': result.get('scope', 'session')})
        except Exception as e:
            return ctx._err(rid, 5001, str(e))
    if key == 'fast':
        raw = str(value or '').strip().lower()
        agent = session.get('agent') if session else None
        if agent is not None:
            current_fast = getattr(agent, 'service_tier', None) == 'priority'
        elif session is not None and session.get('create_service_tier_override') is not None:
            current_fast = session['create_service_tier_override'] == 'priority'
        else:
            current_fast = ctx._load_service_tier() == 'priority'
        if raw in {'status'}:
            return ctx._ok(rid, {'key': key, 'value': 'fast' if current_fast else 'normal'})
        if raw in {'', 'toggle'}:
            nv = 'normal' if current_fast else 'fast'
        elif raw in {'fast', 'on'}:
            nv = 'fast'
        elif raw in {'normal', 'off'}:
            nv = 'normal'
        else:
            return ctx._err(rid, 4002, f'unknown fast mode: {value}')
        overrides = None
        if nv == 'fast':
            from hermes_cli.models import resolve_fast_mode_overrides
            if agent is not None:
                target_model = getattr(agent, 'model', None)
            else:
                session_override = (session or {}).get('model_override') or {}
                target_model = (session_override.get('model') if isinstance(session_override, dict) else None) or ctx._resolve_model()
            if not target_model:
                return ctx._err(rid, 4002, 'fast mode is not available without a selected model')
            overrides = resolve_fast_mode_overrides(target_model)
            if overrides is None:
                return ctx._err(rid, 4002, 'fast mode is not available for this model')
        if session is not None:
            session['create_service_tier_override'] = 'priority' if nv == 'fast' else ''
        else:
            ctx._write_config_key('agent.service_tier', nv)
        if agent is not None:
            agent.service_tier = 'priority' if nv == 'fast' else None
            current_overrides = dict(getattr(agent, 'request_overrides', {}) or {})
            current_overrides.pop('service_tier', None)
            current_overrides.pop('speed', None)
            if nv == 'fast':
                current_overrides.update(overrides)
            agent.request_overrides = current_overrides
            ctx._persist_live_session_runtime(session)
            ctx._emit('session.info', params.get('session_id', ''), ctx._session_info(agent, session))
        return ctx._ok(rid, {'key': key, 'value': nv})
    if key == 'busy':
        raw = str(value or '').strip().lower()
        if raw in {'', 'status'}:
            return ctx._ok(rid, {'key': key, 'value': ctx._load_busy_input_mode()})
        if raw not in {'queue', 'steer', 'interrupt'}:
            return ctx._err(rid, 4002, f'unknown busy mode: {value}')
        ctx._write_config_key('display.busy_input_mode', raw)
        return ctx._ok(rid, {'key': key, 'value': raw})
    if key == 'verbose':
        cycle = ['off', 'new', 'all', 'verbose']
        cur = session.get('tool_progress_mode', ctx._load_tool_progress_mode()) if session else ctx._load_tool_progress_mode()
        if value and value != 'cycle':
            nv = str(value).strip().lower()
            if nv not in cycle:
                return ctx._err(rid, 4002, f'unknown verbose mode: {value}')
        else:
            try:
                idx = cycle.index(cur)
            except ValueError:
                idx = 2
            nv = cycle[(idx + 1) % len(cycle)]
        ctx._write_config_key('display.tool_progress', nv)
        if session:
            session['tool_progress_mode'] = nv
            agent = session.get('agent')
            if agent is not None:
                agent.verbose_logging = nv == 'verbose'
        return ctx._ok(rid, {'key': key, 'value': nv})
    if key == 'focus':
        from hermes_cli.focus_view import FOCUS_TOOL_PROGRESS_MODE, normalize_tool_progress_mode, resolve_focus_arg
        cfg_f = ctx._load_cfg()
        _display_f = cfg_f.get('display')
        d_f: dict = _display_f if isinstance(_display_f, dict) else {}
        cur_focus = bool(d_f.get('focus_view', False))
        action, target = resolve_focus_arg(str(value or ''), cur_focus)
        if action == 'usage':
            return ctx._err(rid, 4002, f'unknown focus value: {value} (use on|off|status)')
        if action == 'status' or target is None:
            return ctx._ok(rid, {'key': key, 'value': 'on' if cur_focus else 'off', 'tool_progress': ctx._load_tool_progress_mode()})
        if target:
            saved = normalize_tool_progress_mode(d_f.get('focus_saved_tool_progress') or ctx._load_tool_progress_mode() if cur_focus else ctx._load_tool_progress_mode())
            ctx._write_config_key('display.focus_saved_tool_progress', saved)
            ctx._write_config_key('display.tool_progress', FOCUS_TOOL_PROGRESS_MODE)
            effective = FOCUS_TOOL_PROGRESS_MODE
        else:
            saved = normalize_tool_progress_mode(d_f.get('focus_saved_tool_progress') or 'all')
            ctx._write_config_key('display.tool_progress', saved)
            effective = saved
        ctx._write_config_key('display.focus_view', bool(target))
        if session:
            session['focus_view'] = bool(target)
            session['tool_progress_mode'] = effective
            agent_f = session.get('agent')
            if agent_f is not None:
                try:
                    agent_f.tool_progress_mode = effective
                except Exception:
                    pass
        return ctx._ok(rid, {'key': key, 'value': 'on' if target else 'off', 'tool_progress': effective})
    if key in {'approval_mode', 'approvals.mode'}:
        raw = str(value or '').strip().lower()
        if raw not in ctx._APPROVAL_MODES:
            return ctx._err(rid, 4002, f'unknown approval mode: {value}; pick one of manual|smart|off')
        ctx._write_config_key('approvals.mode', raw)
        for sid, sess in list(ctx._sessions.items()):
            agent = sess.get('agent')
            if agent is not None:
                ctx._emit('session.info', sid, ctx._session_info(agent, sess))
        return ctx._ok(rid, {'key': 'approvals.mode', 'value': raw})
    if key == 'yolo':
        scope = str(params.get('scope') or 'session').strip().lower()
        try:
            from tools.approval import disable_session_yolo, enable_session_yolo, is_session_yolo_enabled
            raw = str(value or '').strip().lower()

            def _resolve_toggle(current: bool) -> bool:
                if raw in {'1', 'on', 'true', 'yes'}:
                    return True
                if raw in {'0', 'off', 'false', 'no'}:
                    return False
                return not current
            if scope == 'global':
                from tools.approval import _normalize_approval_mode
                cfg = ctx._load_cfg()
                appr = cfg.get('approvals') if isinstance(cfg, dict) else None
                if not isinstance(appr, dict):
                    appr = {}
                current = _normalize_approval_mode(appr.get('mode', 'manual')) == 'off'
                enable = _resolve_toggle(current)
                ctx._write_config_key('approvals.mode', 'off' if enable else 'manual')
                nv = '1' if enable else '0'
                for sid, sess in list(ctx._sessions.items()):
                    agent = sess.get('agent')
                    if agent is not None:
                        ctx._emit('session.info', sid, ctx._session_info(agent, sess))
                return ctx._ok(rid, {'key': key, 'value': nv, 'scope': 'global'})
            if session:
                current = is_session_yolo_enabled(session['session_key'])
                enable = _resolve_toggle(current)
                if enable:
                    enable_session_yolo(session['session_key'])
                    nv = '1'
                else:
                    disable_session_yolo(session['session_key'])
                    nv = '0'
                agent = session.get('agent')
                if agent is not None:
                    ctx._emit('session.info', params.get('session_id', ''), ctx._session_info(agent, session))
            else:
                current = ctx.is_truthy_value(ctx.os.environ.get('HERMES_YOLO_MODE'))
                enable = _resolve_toggle(current)
                if enable:
                    ctx.os.environ['HERMES_YOLO_MODE'] = '1'
                    nv = '1'
                else:
                    ctx.os.environ.pop('HERMES_YOLO_MODE', None)
                    nv = '0'
            return ctx._ok(rid, {'key': key, 'value': nv, 'scope': 'session'})
        except Exception as e:
            return ctx._err(rid, 5001, str(e))
    if key == 'reasoning':
        try:
            from hermes_constants import parse_reasoning_effort
            arg = str(value or '').strip().lower()
            scope = str(params.get('scope') or '').strip().lower()
            global_scope = scope == 'global'
            if arg in {'show', 'on'}:
                cfg = ctx._load_cfg_raw()
                display = cfg.get('display') if isinstance(cfg.get('display'), dict) else {}
                sections = display.get('sections') if isinstance(display.get('sections'), dict) else {}
                display['show_reasoning'] = True
                sections['thinking'] = 'expanded'
                display['sections'] = sections
                cfg['display'] = display
                ctx._save_cfg(cfg)
                if session:
                    session['show_reasoning'] = True
                return ctx._ok(rid, {'key': key, 'value': 'show'})
            if arg in {'hide', 'off'}:
                cfg = ctx._load_cfg_raw()
                display = cfg.get('display') if isinstance(cfg.get('display'), dict) else {}
                sections = display.get('sections') if isinstance(display.get('sections'), dict) else {}
                display['show_reasoning'] = False
                sections['thinking'] = 'hidden'
                display['sections'] = sections
                cfg['display'] = display
                ctx._save_cfg(cfg)
                if session:
                    session['show_reasoning'] = False
                return ctx._ok(rid, {'key': key, 'value': 'hide'})
            if arg in {'full', 'all'}:
                cfg = ctx._load_cfg_raw()
                display = cfg.get('display') if isinstance(cfg.get('display'), dict) else {}
                sections = display.get('sections') if isinstance(display.get('sections'), dict) else {}
                display['reasoning_full'] = True
                sections['thinking'] = 'expanded'
                display['sections'] = sections
                cfg['display'] = display
                ctx._save_cfg(cfg)
                return ctx._ok(rid, {'key': key, 'value': 'full'})
            if arg in {'clamp', 'collapse', 'short'}:
                cfg = ctx._load_cfg_raw()
                display = cfg.get('display') if isinstance(cfg.get('display'), dict) else {}
                sections = display.get('sections') if isinstance(display.get('sections'), dict) else {}
                display['reasoning_full'] = False
                sections['thinking'] = 'collapsed'
                display['sections'] = sections
                cfg['display'] = display
                ctx._save_cfg(cfg)
                return ctx._ok(rid, {'key': key, 'value': 'clamp'})
            parsed = parse_reasoning_effort(arg)
            if parsed is None:
                return ctx._err(rid, 4002, f'unknown reasoning value: {value}')
            if global_scope or session is None:
                ctx._write_config_key('agent.reasoning_effort', arg)
                if session is not None:
                    session.pop('create_reasoning_override', None)
            else:
                session['create_reasoning_override'] = parsed
            if session and session.get('agent') is not None:
                session['agent'].reasoning_config = parsed
                ctx._persist_live_session_runtime(session)
                ctx._emit('session.info', params.get('session_id', ''), ctx._session_info(session['agent'], session))
            return ctx._ok(rid, {'key': key, 'value': arg})
        except Exception as e:
            return ctx._err(rid, 5001, str(e))
    if key == 'details_mode':
        nv = str(value or '').strip().lower()
        if nv not in ctx._DETAIL_MODES:
            return ctx._err(rid, 4002, f'unknown details_mode: {value}')
        cfg = ctx._load_cfg_raw()
        display = cfg.get('display') if isinstance(cfg.get('display'), dict) else {}
        sections = display.get('sections') if isinstance(display.get('sections'), dict) else {}
        display['details_mode'] = nv
        for section in ctx._DETAIL_SECTION_NAMES:
            sections[section] = nv
        display['sections'] = sections
        cfg['display'] = display
        ctx._save_cfg(cfg)
        return ctx._ok(rid, {'key': key, 'value': nv})
    if key.startswith('details_mode.'):
        section = key.split('.', 1)[1]
        if section not in ctx._DETAIL_SECTION_NAMES:
            return ctx._err(rid, 4002, f'unknown section: {section}')
        cfg = ctx._load_cfg_raw()
        display = cfg.get('display') if isinstance(cfg.get('display'), dict) else {}
        sections_cfg = display.get('sections') if isinstance(display.get('sections'), dict) else {}
        nv = str(value or '').strip().lower()
        if not nv:
            sections_cfg.pop(section, None)
            display['sections'] = sections_cfg
            cfg['display'] = display
            ctx._save_cfg(cfg)
            return ctx._ok(rid, {'key': key, 'value': ''})
        if nv not in ctx._DETAIL_MODES:
            return ctx._err(rid, 4002, f'unknown details_mode: {value}')
        sections_cfg[section] = nv
        display['sections'] = sections_cfg
        cfg['display'] = display
        ctx._save_cfg(cfg)
        return ctx._ok(rid, {'key': key, 'value': nv})
    if key == 'thinking_mode':
        nv = str(value or '').strip().lower()
        allowed_tm = frozenset({'collapsed', 'truncated', 'full'})
        if nv not in allowed_tm:
            return ctx._err(rid, 4002, f'unknown thinking_mode: {value}')
        ctx._write_config_key('display.thinking_mode', nv)
        ctx._write_config_key('display.details_mode', 'expanded' if nv == 'full' else 'collapsed')
        return ctx._ok(rid, {'key': key, 'value': nv})
    if key == 'density':
        raw = str(value or '').strip().lower()
        cfg0 = ctx._load_cfg()
        d0 = cfg0.get('display') if isinstance(cfg0.get('display'), dict) else {}
        cur_b = bool(d0.get('tui_compact', False))
        if raw in {'', 'toggle'}:
            nv_b = not cur_b
        elif raw == 'on':
            nv_b = True
        elif raw == 'off':
            nv_b = False
        else:
            return ctx._err(rid, 4002, f'unknown density value: {value}')
        ctx._write_config_key('display.tui_compact', nv_b)
        return ctx._ok(rid, {'key': key, 'value': 'on' if nv_b else 'off'})
    if key == 'battery':
        raw = str(value or '').strip().lower()
        cfg0 = ctx._load_cfg()
        d0 = cfg0.get('display') if isinstance(cfg0.get('display'), dict) else {}
        cur_b = bool(d0.get('battery', False))
        if raw in {'', 'toggle'}:
            nv_b = not cur_b
        elif raw in {'on', 'true', 'yes'}:
            nv_b = True
        elif raw in {'off', 'false', 'no'}:
            nv_b = False
        else:
            return ctx._err(rid, 4002, f'unknown battery value: {value}')
        ctx._write_config_key('display.battery', nv_b)
        return ctx._ok(rid, {'key': key, 'value': 'on' if nv_b else 'off'})
    if key == 'theme':
        raw = str(value or '').strip().lower()
        if raw not in {'auto', 'light', 'dark'}:
            return ctx._err(rid, 4002, f'unknown theme value: {value} (use auto|light|dark)')
        ctx._write_config_key('display.tui_theme', raw)
        return ctx._ok(rid, {'key': key, 'value': raw})
    if key == 'statusbar':
        raw = str(value or '').strip().lower()
        display = ctx._load_cfg().get('display')
        d0 = display if isinstance(display, dict) else {}
        current = ctx._coerce_statusbar(d0.get('tui_statusbar', 'top'))
        if raw in {'', 'toggle'}:
            nv = 'top' if current == 'off' else 'off'
        elif raw == 'on':
            nv = 'top'
        elif raw in ctx._STATUSBAR_MODES:
            nv = raw
        else:
            return ctx._err(rid, 4002, f'unknown statusbar value: {value}')
        ctx._write_config_key('display.tui_statusbar', nv)
        return ctx._ok(rid, {'key': key, 'value': nv})
    if key == 'mouse':
        raw = ('' if value is None else str(value)).strip().lower()
        cfg = ctx._load_cfg()
        display = cfg.get('display') if isinstance(cfg.get('display'), dict) else {}
        current = ctx._display_mouse_tracking(display)
        if raw in {'', 'toggle'}:
            nv = 'all' if current == 'off' else 'off'
        elif raw in ctx._MOUSE_TRACKING_ALIASES:
            nv = ctx._MOUSE_TRACKING_ALIASES[raw]
        else:
            return ctx._err(rid, 4002, f'unknown mouse value: {value}')
        ctx._write_config_key('display.mouse_tracking', nv)
        return ctx._ok(rid, {'key': key, 'value': nv})
    if key == 'indicator':
        raw = ('' if value is None else str(value)).strip().lower()
        if raw not in ctx._INDICATOR_STYLES:
            return ctx._err(rid, 4002, f"unknown indicator: {raw!r}; pick one of {'|'.join(ctx._INDICATOR_STYLES)}")
        ctx._write_config_key('display.tui_status_indicator', raw)
        return ctx._ok(rid, {'key': key, 'value': raw})
    if key in {'cwd', 'terminal.cwd', 'workdir'}:
        raw = str(value or '').strip()
        if not raw:
            return ctx._err(rid, 4002, 'cwd required')
        cwd = ctx.os.path.abspath(ctx.os.path.expanduser(raw))
        if not ctx.os.path.isdir(cwd):
            return ctx._err(rid, 4002, f'working directory does not exist: {raw}')
        ctx._write_config_key('terminal.cwd', cwd)
        ctx.os.environ['TERMINAL_CWD'] = cwd
        return ctx._ok(rid, {'key': 'terminal.cwd', 'value': cwd, 'cwd': cwd, 'branch': ctx._git_branch_for_cwd(cwd)})
    if key in {'prompt', 'personality', 'skin'}:
        try:
            cfg = ctx._load_cfg_raw()
            if key == 'prompt':
                if value == 'clear':
                    cfg.pop('custom_prompt', None)
                    nv = ''
                else:
                    cfg['custom_prompt'] = value
                    nv = value
                ctx._save_cfg(cfg)
            elif key == 'personality':
                sid_key = params.get('session_id', '')
                pname, new_prompt = ctx._validate_personality(str(value or ''), cfg)
                ctx._write_config_key('display.personality', pname)
                ctx._write_config_key('agent.system_prompt', new_prompt)
                nv = str(value or 'none')
                history_reset, info = ctx._apply_personality_to_session(sid_key, session, new_prompt, pname)
            else:
                ctx._write_config_key(f'display.{key}', value)
                nv = value
                if key == 'skin':
                    ctx._broadcast_global_event('skin.changed', ctx.resolve_skin())
                    ctx._note_skin_broadcast()
            resp = {'key': key, 'value': nv}
            if key == 'personality':
                resp['history_reset'] = history_reset
                if info is not None:
                    resp['info'] = info
            return ctx._ok(rid, resp)
        except Exception as e:
            return ctx._err(rid, 5001, str(e))
    return ctx._err(rid, 4002, f'unknown config key: {key}')

@method('setup.status')
def _(ctx, rid, params: dict) -> dict:
    try:
        from hermes_cli.main import _has_any_provider_configured
        return ctx._ok(rid, {'provider_configured': bool(_has_any_provider_configured())})
    except Exception as e:
        return ctx._err(rid, 5016, str(e))

@method('setup.runtime_check')
def _(ctx, rid, params: dict) -> dict:
    """Strict provider check: does the configured/default model actually resolve to a usable runtime?

    Unlike setup.status (which returns True if ANY provider auth state is
    discoverable, including indirect fallbacks like ``gh auth token`` for
    Copilot), this runs the same resolve_runtime_provider() call the agent
    uses on session creation. It returns ok=False with the auth error message
    when the user's configured model cannot actually be served, so UIs can
    surface onboarding before the user submits a doomed prompt.
    """
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.auth import has_usable_secret
        from hermes_cli.main import _has_any_provider_configured
        requested = str(params.get('provider') or '').strip() or None
        runtime = resolve_runtime_provider(requested=requested)
        provider_configured = bool(_has_any_provider_configured())
        provider = runtime.get('provider') or 'provider'
        source = str(runtime.get('source') or '')
        if not provider_configured and provider == 'bedrock' and (source in {'iam-role', 'aws-sdk-default-chain'}):
            return ctx._ok(rid, {'ok': False, 'provider': provider, 'model': runtime.get('model'), 'source': source, 'error': 'No Hermes provider is configured.'})
        api_key = runtime.get('api_key')
        api_key_text = '' if callable(api_key) else str(api_key or '').strip()
        credential_ok = callable(api_key) or api_key_text in {'aws-sdk', 'no-key-required'} or has_usable_secret(api_key_text) or bool(runtime.get('command'))
        if not credential_ok:
            return ctx._ok(rid, {'ok': False, 'provider': provider, 'model': runtime.get('model'), 'source': runtime.get('source'), 'error': f'No usable credentials found for {provider}.'})
        return ctx._ok(rid, {'ok': True, 'provider': runtime.get('provider'), 'model': runtime.get('model'), 'source': runtime.get('source')})
    except Exception as e:
        return ctx._ok(rid, {'ok': False, 'error': str(e)})

def register(server) -> None:
    """Bind this module's handlers onto ``server``'s globals and registry."""
    _registry.install(server)
