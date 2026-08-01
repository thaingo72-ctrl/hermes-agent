"""Single source of truth for runtime working directories.

Session/task CWD is durable state, keyed by the raw session/task id. ContextVars
only carry the current request identity so prompt/context builders can find the
right record without consulting terminal backends or duplicated override maps.
"""

import logging
import os
from contextvars import ContextVar, Token
from pathlib import Path
import threading
from typing import Any

logger = logging.getLogger(__name__)

_UNSET: Any = object()

_CURRENT_SESSION_KEY: ContextVar = ContextVar("HERMES_CURRENT_SESSION_KEY", default=_UNSET)
_SESSION_CWDS: dict[str, str] = {}
_SESSION_CWDS_LOCK = threading.Lock()

# The Python package/source root (this file lives at <root>/agent/runtime_cwd.py).
# When a backend is launched from, or self-spawns into, this tree (the desktop
# app default), an os.getcwd() fallback would inject this repo's contributor
# AGENTS.md as authoritative project context. Context discovery must never
# resolve here.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _is_install_tree(p: Path) -> bool:
    # True only when p IS the package root or sits inside it. Ancestors of the
    # package root (a user home that happens to contain the checkout, a --user
    # site-packages parent) are legitimate workspaces and must not be blocked.
    try:
        p = p.resolve()
    except Exception:
        return False
    return p == _PACKAGE_ROOT or _PACKAGE_ROOT in p.parents


def _record_key(session_key: str | None) -> str:
    return str(session_key or "default")


def bind_current_session_key(session_key: str | None) -> Token:
    """Bind the current request to an existing runtime-CWD record."""
    return _CURRENT_SESSION_KEY.set(_record_key(session_key))


def clear_current_session_key() -> None:
    _CURRENT_SESSION_KEY.set("")


def current_session_key() -> str:
    value = _CURRENT_SESSION_KEY.get()
    if value is _UNSET:
        return ""
    return str(value).strip()


def record_session_cwd(session_key: str | None, cwd: str | None) -> None:
    """Record *cwd* as the authoritative logical cwd for *session_key*."""
    if not isinstance(cwd, str) or not cwd.strip():
        return
    with _SESSION_CWDS_LOCK:
        _SESSION_CWDS[_record_key(session_key)] = cwd.strip()


def get_session_cwd(session_key: str | None) -> str | None:
    """Return the recorded cwd for *session_key*, or None when uninitialized."""
    with _SESSION_CWDS_LOCK:
        return _SESSION_CWDS.get(_record_key(session_key))


def clear_session_cwd(session_key: str | None) -> None:
    """Clear a durable CWD record. Only call from true session reset/delete."""
    with _SESSION_CWDS_LOCK:
        _SESSION_CWDS.pop(_record_key(session_key), None)


def initialize_session_cwd(session_key: str | None, cwd: str | None) -> None:
    """Seed a session CWD record once at a session boundary."""
    if not isinstance(cwd, str) or not cwd.strip():
        return
    key = _record_key(session_key)
    with _SESSION_CWDS_LOCK:
        _SESSION_CWDS.setdefault(key, cwd.strip())


def resolve_agent_cwd() -> Path:
    session_key = current_session_key()
    recorded = get_session_cwd(session_key) if session_key else None
    if recorded:
        p = Path(recorded).expanduser()
        if p.is_dir():
            return p
        logger.warning("session working directory does not exist: %s", recorded)
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if p.is_dir():
            return p
        logger.warning("TERMINAL_CWD does not exist: %s", raw)
    return Path(os.getcwd())


def resolve_context_cwd() -> Path | None:
    # None means "no configured cwd": build_context_files_prompt then falls back
    # to the launch dir (os.getcwd()), correct for a local CLI launched inside a
    # real project. A configured path is validated here (previously it was passed
    # through unchecked, diverging from resolve_agent_cwd). An explicitly
    # configured path is otherwise honored verbatim — including the Hermes
    # source tree itself, which is a legitimate workspace when the user is
    # developing Hermes (per-surface policy for fallback-picked directories
    # lives in build_context_files_prompt; see #64590).
    session_key = current_session_key()
    recorded = get_session_cwd(session_key) if session_key else None
    if recorded:
        p = Path(recorded).expanduser()
        if not p.is_dir():
            logger.warning("session working directory does not exist: %s", recorded)
        else:
            return p
        return None
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_dir():
            logger.warning("TERMINAL_CWD does not exist: %s", raw)
        else:
            return p
    return None
