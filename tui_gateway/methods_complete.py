"""Completion, model picker, and paste JSON-RPC handlers for the TUI gateway."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, MutableMapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompletionServices:
    sessions: MutableMapping[str, dict]
    hermes_home: Callable[[], Path]
    profile_home: Callable[[str | None], Path | None]
    load_cfg: Callable[[], dict]
    apply_managed: Callable[[dict], dict]
    resolve_model: Callable[[], str]


def _ok(rid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}


_paste_counter = 0


def paste_collapse(rid, params: dict, services: CompletionServices) -> dict:
    global _paste_counter
    text = params.get("text", "")
    if not text:
        return _err(rid, 4004, "empty paste")

    _paste_counter += 1
    line_count = text.count("\n") + 1
    paste_dir = services.hermes_home() / "pastes"
    paste_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    paste_file = (
        paste_dir / f"paste_{_paste_counter}_{datetime.now().strftime('%H%M%S')}.txt"
    )
    paste_file.write_text(text, encoding="utf-8")

    placeholder = (
        f"[Pasted text #{_paste_counter}: {line_count} lines \u2192 {paste_file}]"
    )
    return _ok(
        rid, {"placeholder": placeholder, "path": str(paste_file), "lines": line_count}
    )


def complete_path(rid, params: dict, services: CompletionServices) -> dict:
    word = params.get("word", "")
    if not word:
        return _ok(rid, {"items": []})

    items: list[dict] = []
    try:
        root = _completion_cwd(params, services)
        is_context = word.startswith("@")
        query = word[1:] if is_context else word

        if is_context and not query:
            items = [
                {"text": "@diff", "display": "@diff", "meta": "git diff"},
                {"text": "@staged", "display": "@staged", "meta": "staged diff"},
                {"text": "@file:", "display": "@file:", "meta": "attach file"},
                {"text": "@folder:", "display": "@folder:", "meta": "attach folder"},
                {"text": "@url:", "display": "@url:", "meta": "fetch url"},
                {"text": "@git:", "display": "@git:", "meta": "git log"},
            ]
            return _ok(rid, {"items": items})

        # Accept both `@folder:path` and the bare `@folder` form so the user
        # sees directory listings as soon as they finish typing the keyword,
        # without first accepting the static `@folder:` hint.
        if is_context and query in {"file", "folder"}:
            prefix_tag, path_part = query, ""
        elif is_context and query.startswith(("file:", "folder:")):
            prefix_tag, _, tail = query.partition(":")
            path_part = tail
        else:
            prefix_tag = ""
            path_part = query if is_context else query

        # `@/foo` almost always means "foo, from here" rather than the absolute
        # `/foo`: the `@` already says "this is a path", so the slash reads as a
        # separator people type out of habit. Take the absolute reading only
        # when something is actually there, else drop the slash and resolve
        # relative to the cwd — otherwise `@/Desktop` dead-ends on a directory
        # that exists one level down. Real absolute paths (`@/usr/local`,
        # `@/etc/hosts`) still resolve, since those prefixes do exist.
        if (
            is_context
            and path_part.startswith("/")
            and not path_part.startswith("//")
            and not _abs_completion_prefix_exists(path_part)
        ):
            path_part = path_part.lstrip("/")

        # Fuzzy basename search across the repo when the user types a bare
        # name with no path separator — `@appChrome` surfaces every file
        # whose basename matches, regardless of directory depth. Matches what
        # editors like Cursor / VS Code do for Cmd-P. Path-ish queries (with
        # `/`, `./`, `~/`, `/abs`) fall through to the directory-listing
        # path so explicit navigation intent is preserved.
        if (
            is_context
            and path_part
            and len(path_part.strip()) >= 2
            and "/" not in path_part
            and prefix_tag != "folder"
        ):
            ranked: list[tuple[tuple[int, int], str, str, bool]] = []
            walked_dirs: set[str] = set()
            seen: set[str] = set()
            want_hidden = path_part.startswith(".")

            def _consider(rel: str, name: str, is_dir: bool) -> None:
                if rel in seen or (name.startswith(".") and not want_hidden):
                    return
                rank = _fuzzy_basename_rank(name, path_part)
                if rank is not None:
                    seen.add(rel)
                    ranked.append((rank, rel, name, is_dir))

            # Seed with root's immediate children. `_list_repo_files` is capped
            # at _FUZZY_CACHE_MAX_FILES, and outside a git repo the fallback
            # walk can burn that whole budget on one deep subtree before ever
            # reaching a sibling — which is why `@Desk` in a non-repo $HOME
            # found nothing. One listdir keeps the top level always reachable.
            try:
                for entry in os.listdir(root):
                    if entry not in _FUZZY_FALLBACK_EXCLUDES:
                        _consider(entry, entry, os.path.isdir(os.path.join(root, entry)))
            except OSError:
                pass

            for rel in _list_repo_files(root):
                _consider(rel, os.path.basename(rel), False)

                # Directories are only implied by the file listing, so rank each
                # ancestor too. Without this a bare `@Desktop` finds nothing —
                # a folder with no name-matching file inside it is invisible to
                # a file-only scan, which is the "can't @ a folder by name" bug.
                parent = os.path.dirname(rel)
                while parent and parent not in walked_dirs:
                    walked_dirs.add(parent)
                    _consider(parent, os.path.basename(parent), True)
                    parent = os.path.dirname(parent)

            # Same rank tier: folders first, so `@Desktop` leads with the folder
            # rather than a file that merely fuzzy-matches the same letters.
            ranked.sort(key=lambda r: (r[0], not r[3], len(r[1]), r[1]))
            tag = prefix_tag or "file"
            for _, rel, basename, is_dir in ranked[:30]:
                items.append(
                    {
                        "text": f"@{'folder' if is_dir else tag}:{rel}{'/' if is_dir else ''}",
                        "display": basename + ("/" if is_dir else ""),
                        "meta": "dir" if is_dir else os.path.dirname(rel),
                    }
                )

            return _ok(rid, {"items": items})

        expanded = _normalize_completion_path(path_part) if path_part else "."
        if expanded == "." or not expanded:
            search_dir, match = ".", ""
        elif expanded.endswith("/"):
            search_dir, match = expanded, ""
        else:
            search_dir = os.path.dirname(expanded) or "."
            match = os.path.basename(expanded)

        search_dir = (
            search_dir if os.path.isabs(search_dir) else os.path.join(root, search_dir)
        )
        if not os.path.isdir(search_dir):
            return _ok(rid, {"items": []})

        want_dir = prefix_tag == "folder"
        match_lower = match.lower()
        for entry in sorted(os.listdir(search_dir)):
            if match and not entry.lower().startswith(match_lower):
                continue
            if is_context and entry in _FUZZY_FALLBACK_EXCLUDES:
                continue
            if is_context and not prefix_tag and entry.startswith("."):
                continue
            full = os.path.join(search_dir, entry)
            is_dir = os.path.isdir(full)
            # Explicit `@folder:` / `@file:` — honour the user's filter.  Skip
            # the opposite kind instead of auto-rewriting the completion tag,
            # which used to defeat the prefix and let `@folder:` list files.
            if prefix_tag and want_dir != is_dir:
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            suffix = "/" if is_dir else ""

            if is_context and prefix_tag:
                text = f"@{prefix_tag}:{rel}{suffix}"
            elif is_context:
                kind = "folder" if is_dir else "file"
                text = f"@{kind}:{rel}{suffix}"
            elif word.startswith("~"):
                text = "~/" + os.path.relpath(full, os.path.expanduser("~")) + suffix
            elif word.startswith("./"):
                text = "./" + rel + suffix
            else:
                text = rel + suffix

            items.append(
                {
                    "text": text,
                    "display": entry + suffix,
                    "meta": "dir" if is_dir else "",
                }
            )
            if len(items) >= 30:
                break
    except Exception as e:
        return _err(rid, 5021, str(e))

    return _ok(rid, {"items": items})


def complete_slash(rid, params: dict, services: CompletionServices) -> dict:
    text = params.get("text", "")
    if not text.startswith("/"):
        return _ok(rid, {"items": []})

    try:
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document
        from prompt_toolkit.formatted_text import to_plain_text

        from agent.skill_commands import get_skill_commands
        from agent.skill_bundles import get_skill_bundles

        completer = SlashCommandCompleter(
            skill_commands_provider=lambda: get_skill_commands(),
            skill_bundles_provider=lambda: get_skill_bundles(),
        )
        doc = Document(text, len(text))
        # Skill commands and bundles are the only completions offered for an
        # inline `/skill` reference typed mid-message, so the class has to
        # reach the TUI as data. Derived from the same providers the completer
        # uses; display glyphs are not protocol data.
        skill_names = {
            key.lstrip("/").lower()
            for key in (*get_skill_commands(), *get_skill_bundles())
        }
        items = [
            {
                "text": c.text,
                # prompt_toolkit gives us FormattedText (a list of (style,
                # text) tuples) for display/display_meta. Serialize both as
                # plain strings — the TUI's CompletionItem.display contract
                # is a string, and sending the raw list trips Ink's row
                # layout into 1-char truncation of the next column.
                "display": to_plain_text(c.display) if c.display else c.text,
                "meta": to_plain_text(c.display_meta) if c.display_meta else "",
                "kind": (
                    "skill"
                    if c.text.strip().lstrip("/").lower() in skill_names
                    else "command"
                ),
            }
            for c in completer.get_completions(doc, None)
        ]

        # Rank and bound the list (see _rank_slash_completions) while a
        # `/token` is under the cursor — the one stage skills are offered at.
        # An argument stage (`/personality `, `/details c`) keeps the order
        # its own command chose.
        if text.rsplit(" ", 1)[-1].startswith("/"):
            usage, origin_of = _skill_usage_lookup()
            items = _rank_slash_completions(
                items, usage, origin_of, browsing=text == "/"
            )
        else:
            items = items[:_SLASH_COMPLETION_LIMIT]

        text_lower = text.lower()
        extras = [
            {
                "text": "/density",
                "display": "/density",
                "meta": "Toggle compact display mode",
                "kind": "command",
            },
            {
                "text": "/details",
                "display": "/details",
                "meta": "Control agent detail visibility",
                "kind": "command",
            },
            {
                "text": "/logs",
                "display": "/logs",
                "meta": "Show recent gateway log lines",
                "kind": "command",
            },
            {
                "text": "/mouse",
                "display": "/mouse",
                "meta": "Set mouse tracking preset [on|off|toggle|wheel|buttons|all]",
                "kind": "command",
            },
        ]
        for extra in extras:
            if extra["text"].startswith(text_lower) and not any(
                item["text"] == extra["text"] for item in items
            ):
                items.append(extra)

        details_items = _details_completions(text)
        if details_items is not None:
            return _ok(
                rid,
                {
                    "items": details_items,
                    "replace_from": text.rfind(" ") + 1 if " " in text else len(text),
                },
            )

        return _ok(
            rid,
            {"items": items, "replace_from": text.rfind(" ") + 1 if " " in text else 1},
        )
    except Exception as e:
        return _err(rid, 5020, str(e))


def model_options(rid, params: dict, services: CompletionServices) -> dict:
    try:
        from hermes_cli.inventory import build_model_options_payload

        session = services.sessions.get(params.get("session_id", ""))
        agent = session.get("agent") if session else None
        # Layer agent-session state on top of disk config — once an agent
        # is spawned, IT owns the live provider/model/base_url. Empty
        # agent attributes must NOT clobber disk config (with_overrides
        # is truthy-only).
        ctx = _model_picker_context(agent, services)
        payload = build_model_options_payload(
            ctx,
            explicit_only=bool(params.get("explicit_only")),
            include_unconfigured=bool(params.get("include_unconfigured")),
            refresh=bool(params.get("refresh")),
        )
        return _ok(rid, payload)
    except Exception as e:
        return _err(rid, 5033, str(e))


def model_save_key(rid, params: dict, services: CompletionServices) -> dict:
    """Save an API key for a provider, then return its refreshed model list.

    Params:
        slug: provider slug (e.g. "deepseek", "xai")
        api_key: the key value to save

    Returns the provider dict with models populated (same shape as
    model.options entries) on success.
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.config import is_managed
        from hermes_cli.inventory import build_models_payload

        slug = (params.get("slug") or "").strip()
        api_key = (params.get("api_key") or "").strip()
        if not slug or not api_key:
            return _err(rid, 4001, "slug and api_key are required")

        if is_managed():
            return _err(rid, 4006, "managed install — credentials are read-only")

        pconfig = PROVIDER_REGISTRY.get(slug)
        if not pconfig:
            return _err(rid, 4002, f"unknown provider: {slug}")
        if pconfig.auth_type != "api_key":
            return _err(
                rid,
                4003,
                f"{pconfig.name} uses {pconfig.auth_type} auth — "
                f"run `hermes model` to configure",
            )
        if not pconfig.api_key_env_vars:
            return _err(rid, 4004, f"no env var defined for {pconfig.name}")

        # Save the key to ~/.hermes/.env via the unified credential lifecycle
        # so any stale config.yaml mirror of the previous key (model.api_key,
        # custom_providers[*].api_key) is rotated in the same action (#62269).
        env_var = pconfig.api_key_env_vars[0]
        from hermes_cli.credential_lifecycle import save_provider_env_credential

        save_provider_env_credential(env_var, api_key)
        # Also set in current process so the refreshed inventory sees it.
        import os

        os.environ[env_var] = api_key

        # Refresh provider data via the shared inventory builder so this
        # surface stays in lock-step with model.options + dashboard
        # /api/model/options. picker_hints=True ensures the returned row
        # carries `authenticated` for the TUI frontend.
        session = services.sessions.get(params.get("session_id", ""))
        agent = session.get("agent") if session else None
        ctx = _model_picker_context(agent, services)
        payload = build_models_payload(
            ctx, picker_hints=True, max_models=50,
        )
        provider_data = next(
            (p for p in payload["providers"] if p["slug"] == slug), None
        )
        if provider_data is None:
            # Key was saved but provider didn't appear — still return success.
            provider_data = {
                "slug": slug,
                "name": pconfig.name,
                "is_current": False,
                "models": [],
                "total_models": 0,
                "authenticated": True,
            }
        # picker_hints sets `authenticated` from the row state, but the
        # synthetic fallback above doesn't go through that path.
        provider_data["authenticated"] = True
        return _ok(rid, {"provider": provider_data})
    except Exception as e:
        return _err(rid, 5034, str(e))


def model_disconnect(rid, params: dict, services: CompletionServices) -> dict:
    """Remove credentials for a provider.

    Params:
        slug: provider slug (e.g. "deepseek", "xai")

    Returns success status and the provider's slug.
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY, clear_provider_auth
        from hermes_cli.credential_lifecycle import remove_provider_env_credential

        slug = (params.get("slug") or "").strip()
        if not slug:
            return _err(rid, 4001, "slug is required")

        pconfig = PROVIDER_REGISTRY.get(slug)
        cleared_env = False
        cleared_auth = False

        # Remove API key env vars from .env and process, plus every mirror
        # (env-seeded credential_pool entries, provider model cache rows,
        # value-matched config.yaml api_key copies) via the unified helper —
        # otherwise the provider resurrects in the picker after restart
        # (#51071 / #59761).
        if pconfig and pconfig.api_key_env_vars:
            for ev in pconfig.api_key_env_vars:
                if remove_provider_env_credential(ev).get("found"):
                    cleared_env = True

        # Clear OAuth / credential pool state. This is a full provider
        # disconnect (TUI "disconnect" action), so removing OAuth grants
        # here is the documented intent — unlike the key-only delete paths.
        cleared_auth = clear_provider_auth(slug)

        if not cleared_env and not cleared_auth:
            return _err(rid, 4005, f"no credentials found for {slug}")

        provider_name = pconfig.name if pconfig else slug
        return _ok(
            rid,
            {
                "slug": slug,
                "name": provider_name,
                "disconnected": True,
            },
        )
    except Exception as e:
        return _err(rid, 5035, str(e))


_CWD_PLACEHOLDERS = {".", "auto", "cwd"}


def _configured_cwd_from_cfg(cfg: dict | None) -> str | None:
    if not isinstance(cfg, dict):
        return None
    terminal_cfg = cfg.get("terminal")
    if not isinstance(terminal_cfg, dict):
        return None
    raw = str(terminal_cfg.get("cwd") or "").strip()
    if not raw or raw in _CWD_PLACEHOLDERS:
        return None
    resolved = os.path.abspath(os.path.expanduser(raw))
    return resolved if os.path.isdir(resolved) else None


def _profile_configured_cwd(
    profile_home: Path | None, services: CompletionServices
) -> str | None:
    if profile_home is None:
        return None
    try:
        from hermes_cli.config import _expand_env_vars, read_user_config_raw

        p = Path(profile_home) / "config.yaml"
        if not p.exists():
            return None
        data = services.apply_managed(read_user_config_raw(p))
        expanded = _expand_env_vars(data)
        if isinstance(expanded, dict):
            data = expanded
        return _configured_cwd_from_cfg(data)
    except Exception:
        return None


def _launch_configured_cwd(services: CompletionServices) -> str | None:
    try:
        return _configured_cwd_from_cfg(services.load_cfg())
    except Exception:
        return None


def _normalize_completion_path(path_part: str) -> str:
    expanded = os.path.expanduser(path_part)
    if os.name != "nt":
        normalized = expanded.replace("\\", "/")
        if (
            len(normalized) >= 3
            and normalized[1] == ":"
            and normalized[2] == "/"
            and normalized[0].isalpha()
        ):
            return f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    return expanded


def _completion_cwd(
    params: dict | None = None, services: CompletionServices | None = None
) -> str:
    if services is None:
        services = _standalone_services()
    params = params or {}
    raw = (
        params.get("cwd")
        or services.sessions.get(params.get("session_id") or "", {}).get("cwd")
        or _profile_configured_cwd(services.profile_home(params.get("profile")), services)
        or _launch_configured_cwd(services)
        or os.environ.get("TERMINAL_CWD")
        or os.getcwd()
    )
    try:
        resolved = os.path.abspath(os.path.expanduser(str(raw)))
        if os.path.isdir(resolved):
            return resolved
    except Exception:
        pass
    return os.getcwd()


def _skill_usage_lookup():
    try:
        from tools.skill_usage import (
            _read_bundled_manifest_names,
            _read_hub_installed_names,
            activity_count,
            load_usage,
        )

        records = load_usage()
        bundled = _read_bundled_manifest_names()
        hub = _read_hub_installed_names()
    except Exception as e:
        logger.debug("skill usage lookup unavailable: %s", e)
        return (lambda _name: 0), (lambda _name: "local")

    def usage(name: str) -> int:
        try:
            return activity_count(records.get(name) or {})
        except Exception:
            return 0

    def origin(name: str) -> str:
        if name in hub:
            return "hub"
        if name in bundled:
            return "bundled"
        return "local"

    return usage, origin


_SLASH_COMPLETION_LIMIT = 30


def _rank_slash_completions(
    items: list[dict],
    usage,
    origin_of,
    *,
    browsing: bool,
) -> list[dict]:
    """Rank slash completions while preserving registry-command order."""

    def name_of(item: dict) -> str:
        return str(item.get("text", "")).strip().lstrip("/").lower()

    commands = [item for item in items if item.get("kind") != "skill"]
    skills = [item for item in items if item.get("kind") == "skill"]

    if browsing:
        skills = [
            item
            for item in skills
            if origin_of(name_of(item)) != "bundled" or usage(name_of(item)) > 0
        ]

    skills.sort(key=lambda item: (-usage(name_of(item)), name_of(item)))
    return commands[:_SLASH_COMPLETION_LIMIT] + skills[:_SLASH_COMPLETION_LIMIT]


_FUZZY_CACHE_TTL_S = 5.0
_FUZZY_CACHE_MAX_FILES = 20000
_FUZZY_FALLBACK_EXCLUDES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".next",
        ".cache",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        "target",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)
_fuzzy_cache_lock = threading.Lock()
_fuzzy_cache: dict[str, tuple[float, list[str]]] = {}


def _list_repo_files(root: str) -> list[str]:
    now = time.monotonic()
    with _fuzzy_cache_lock:
        cached = _fuzzy_cache.get(root)
        if cached and now - cached[0] < _FUZZY_CACHE_TTL_S:
            return cached[1]

    files: list[str] = []
    from hermes_cli._subprocess_compat import windows_hide_flags

    try:
        top_result = subprocess.run(
            ["git", "-C", root, "rev-parse", "--show-toplevel"],
            capture_output=True,
            timeout=2.0,
            check=False,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        if top_result.returncode == 0:
            top = top_result.stdout.decode("utf-8", "replace").strip()
            list_result = subprocess.run(
                [
                    "git",
                    "-C",
                    top,
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                ],
                capture_output=True,
                timeout=2.0,
                check=False,
                stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags(),
            )
            if list_result.returncode == 0:
                for p in list_result.stdout.decode("utf-8", "replace").split("\0"):
                    if not p:
                        continue
                    rel = os.path.relpath(os.path.join(top, p), root).replace(
                        os.sep, "/"
                    )
                    if rel.startswith("../"):
                        continue
                    files.append(rel)
                    if len(files) >= _FUZZY_CACHE_MAX_FILES:
                        break
    except (OSError, subprocess.TimeoutExpired):
        pass

    if not files:
        try:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in _FUZZY_FALLBACK_EXCLUDES and not d.startswith(".")
                ]
                rel_dir = os.path.relpath(dirpath, root)
                for f in filenames:
                    rel = f if rel_dir == "." else f"{rel_dir}/{f}"
                    files.append(rel.replace(os.sep, "/"))
                    if len(files) >= _FUZZY_CACHE_MAX_FILES:
                        break
                if len(files) >= _FUZZY_CACHE_MAX_FILES:
                    break
        except OSError:
            pass

    with _fuzzy_cache_lock:
        _fuzzy_cache[root] = (now, files)
    return files


def _fuzzy_basename_rank(name: str, query: str) -> tuple[int, int] | None:
    if not query:
        return (3, len(name))
    nl = name.lower()
    ql = query.lower()
    if nl == ql:
        return (0, len(name))
    if nl.startswith(ql):
        return (1, len(name))

    parts: list[str] = []
    buf = ""
    for ch in name:
        if ch in "-_." or (ch.isupper() and buf and not buf[-1].isupper()):
            if buf:
                parts.append(buf)
            buf = ch if ch not in "-_." else ""
        else:
            buf += ch
    if buf:
        parts.append(buf)
    for p in parts:
        if p.lower().startswith(ql):
            return (2, len(name))
    if ql in nl:
        return (3, len(name))

    i = 0
    for ch in nl:
        if ch == ql[i]:
            i += 1
            if i == len(ql):
                return (4, len(name))
    return None


def _abs_completion_prefix_exists(path_part: str) -> bool:
    expanded = _normalize_completion_path(path_part)
    parent = os.path.dirname(expanded.rstrip("/")) or "/"
    tail = os.path.basename(expanded.rstrip("/"))
    if not os.path.isdir(parent):
        return False
    if not tail or expanded.endswith("/"):
        return os.path.isdir(expanded) or expanded == "/"
    try:
        tail_lower = tail.lower()
        return any(e.lower().startswith(tail_lower) for e in os.listdir(parent))
    except OSError:
        return False


def _details_completion_item(value: str, meta: str = "") -> dict:
    return {"text": value, "display": value, "meta": meta}


def _details_root_completion_item(
    value: str, meta: str, needs_leading_space: bool
) -> dict:
    return _details_completion_item(
        f" {value}" if needs_leading_space else value,
        meta,
    )


def _details_completions(text: str) -> list[dict] | None:
    if not text.lower().startswith("/details"):
        return None

    stripped = text.strip()
    if stripped and not "/details".startswith(stripped.lower().split()[0]):
        return None

    body = text[len("/details") :]
    if body.startswith(" "):
        body = body[1:]
    parts = body.split()
    has_trailing_space = text.endswith(" ")
    sections = ("thinking", "tools", "subagents", "activity")
    modes = ("hidden", "collapsed", "expanded")

    if not body or (len(parts) == 0 and has_trailing_space):
        return [
            *[
                _details_root_completion_item(
                    mode, "global mode", not has_trailing_space
                )
                for mode in modes
            ],
            _details_root_completion_item(
                "cycle", "cycle global mode", not has_trailing_space
            ),
            *[
                _details_root_completion_item(
                    section, "section override", not has_trailing_space
                )
                for section in sections
            ],
        ]

    if len(parts) == 1 and not has_trailing_space:
        prefix = parts[0].lower()
        candidates = [*modes, "cycle", *sections]
        return [
            _details_completion_item(
                candidate,
                (
                    "section override"
                    if candidate in sections
                    else "cycle global mode" if candidate == "cycle" else "global mode"
                ),
            )
            for candidate in candidates
            if candidate.startswith(prefix) and candidate != prefix
        ]

    if len(parts) == 1 and has_trailing_space and parts[0].lower() in sections:
        return [
            *[
                _details_completion_item(mode, f"set {parts[0].lower()}")
                for mode in modes
            ],
            _details_completion_item("reset", f"clear {parts[0].lower()} override"),
        ]

    if len(parts) == 2 and not has_trailing_space and parts[0].lower() in sections:
        prefix = parts[1].lower()
        return [
            _details_completion_item(
                candidate,
                (
                    f"clear {parts[0].lower()} override"
                    if candidate == "reset"
                    else f"set {parts[0].lower()}"
                ),
            )
            for candidate in (*modes, "reset")
            if candidate.startswith(prefix) and candidate != prefix
        ]

    return []


def _model_picker_context(agent, services: CompletionServices | None = None):
    if services is None:
        services = _standalone_services()
    from hermes_cli.inventory import load_picker_context

    ctx = load_picker_context()
    provider = getattr(agent, "provider", "") if agent else ""
    base_url = getattr(agent, "base_url", "") if agent else ""
    if str(provider or "").strip().lower() == "custom":
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity

            provider = (
                canonical_custom_identity(
                    base_url=base_url or None,
                    config_provider=ctx.current_provider,
                    model=(getattr(agent, "model", "") if agent else "") or None,
                )
                or provider
            )
        except Exception:
            logger.debug(
                "custom provider identity recovery failed (model picker)",
                exc_info=True,
            )

    return ctx.with_overrides(
        current_provider=provider,
        current_model=(getattr(agent, "model", "") if agent else "")
        or services.resolve_model(),
        current_base_url=base_url,
    )


def _standalone_services() -> CompletionServices:
    from hermes_constants import get_hermes_home

    def load_cfg() -> dict:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        return cfg if isinstance(cfg, dict) else {}

    def profile_home(_profile: str | None) -> Path | None:
        return None

    def resolve_model() -> str:
        cfg = load_cfg()
        model = cfg.get("model", "")
        if isinstance(model, dict):
            return str(model.get("default", "") or "").strip()
        return str(model or "").strip()

    return CompletionServices(
        sessions={},
        hermes_home=get_hermes_home,
        profile_home=profile_home,
        load_cfg=load_cfg,
        apply_managed=lambda cfg: cfg,
        resolve_model=resolve_model,
    )


def register(
    methods: dict[str, Callable],
    *,
    services: CompletionServices,
) -> None:
    """Register ordinary completion callables under the existing JSON-RPC names."""

    def bind(handler: Callable[[object, dict, CompletionServices], dict]):
        return lambda rid, params: handler(rid, params, services)

    methods["paste.collapse"] = bind(paste_collapse)
    methods["complete.path"] = bind(complete_path)
    methods["complete.slash"] = bind(complete_slash)
    methods["model.options"] = bind(model_options)
    methods["model.save_key"] = bind(model_save_key)
    methods["model.disconnect"] = bind(model_disconnect)
