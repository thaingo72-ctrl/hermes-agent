"""
Platform Adapter Registry

Allows platform adapters (built-in and plugin) to self-register so the gateway
can discover and instantiate them without hardcoded if/elif chains.

Usage (plugin side):

    from gateway.platform_registry import platform_registry, PlatformEntry

    platform_registry.register(PlatformEntry(
        name="irc",
        label="IRC",
        adapter_factory=lambda cfg: IRCAdapter(cfg),
        check_fn=check_requirements,
        validate_config=lambda cfg: bool(cfg.extra.get("server")),
        required_env=["IRC_SERVER"],
        install_hint="pip install irc",
    ))

Usage (gateway side):

    adapter = platform_registry.create_adapter("irc", platform_config)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

_SOURCE_PRIORITY = {
    "builtin": 10,
    "plugin": 100,
}


def _source_priority(source: str) -> int:
    return _SOURCE_PRIORITY.get(source, 50)


@dataclass
class _DeferredPlatform:
    loader: Callable[[], None]
    source: str = "plugin"


@dataclass
class PlatformEntry:
    """Metadata and factory for a single platform adapter."""

    # Identifier used in config.yaml (e.g. "irc", "viber").
    name: str

    # Human-readable label (e.g. "IRC", "Viber").
    label: str

    # Factory callable: receives a PlatformConfig, returns an adapter instance.
    # Using a factory instead of a bare class lets plugins do custom init
    # (e.g. passing extra kwargs, wrapping in try/except).
    adapter_factory: Callable[[Any], Any]

    # Returns True when the platform's dependencies are available.
    check_fn: Callable[[], bool]

    # Optional: given a PlatformConfig, is it properly configured?
    # If None, the registry skips config validation and lets the adapter
    # fail at connect() time with a descriptive error.
    validate_config: Optional[Callable[[Any], bool]] = None

    # Optional: given a PlatformConfig, is the platform connected/enabled?
    # Used by ``GatewayConfig.get_connected_platforms()`` and setup UI status.
    # If None, falls back to ``validate_config`` or ``check_fn``.
    is_connected: Optional[Callable[[Any], bool]] = None

    # Env vars this platform needs (for ``hermes setup`` display).
    required_env: list = field(default_factory=list)

    # Hint shown when check_fn returns False.
    install_hint: str = ""

    # Optional setup function for interactive configuration.
    # Signature: () -> None (prompts user, saves env vars).
    # If None, falls back to _setup_standard_platform (needs token_var + vars)
    # or a generic "set these env vars" display.
    setup_fn: Optional[Callable[[], None]] = None

    # "builtin" or "plugin"
    source: str = "plugin"

    # Name of the plugin manifest that registered this entry (empty for
    # built-ins).  Used by ``hermes gateway setup`` to auto-enable the
    # owning plugin when the user configures its platform.
    plugin_name: str = ""

    # ── Auth env var names (for _is_user_authorized integration) ──
    # E.g. "IRC_ALLOWED_USERS" — checked for comma-separated user IDs.
    allowed_users_env: str = ""
    # E.g. "IRC_ALLOW_ALL_USERS" — if truthy, all users authorized.
    allow_all_env: str = ""

    # ── Message limits ──
    # Max message length for smart-chunking.  0 = no limit.
    max_message_length: int = 0

    # ── Privacy ──
    # If True, session descriptions redact PII (phone numbers, etc.)
    pii_safe: bool = False

    # ── Display ──
    # Emoji for CLI/gateway display (e.g. "💬")
    emoji: str = "🔌"

    # Whether this platform should appear in _UPDATE_ALLOWED_PLATFORMS
    # (allows /update command from this platform).
    allow_update_command: bool = True

    # ── LLM guidance ──
    # Platform hint injected into the system prompt (e.g. "You are on IRC.
    # Do not use markdown.").  Empty string = no hint.
    platform_hint: str = ""

    # ── Env-driven auto-configuration ──
    # Optional: read env vars, return a dict of ``PlatformConfig.extra`` fields
    # to seed when the platform is auto-enabled.  Called during
    # ``_apply_env_overrides`` BEFORE the adapter is constructed, so
    # ``gateway status`` etc. can reflect env-only configuration without
    # instantiating the adapter.  Return ``None`` (or an empty dict) to skip.
    # Signature: () -> Optional[dict[str, Any]]
    env_enablement_fn: Optional[Callable[[], Optional[dict]]] = None

    # ── YAML→env config bridge ──
    # Optional: translate this platform's ``config.yaml`` keys into env vars
    # and/or seed ``PlatformConfig.extra`` directly.  Lets a plugin own its
    # YAML config translation instead of forcing core ``gateway/config.py``
    # to know every platform's schema.
    #
    # Signature: (yaml_cfg: dict, platform_cfg: dict) -> Optional[dict]
    # Called from ``load_gateway_config()`` after the generic shared-key loop
    # and before ``_apply_env_overrides``.  Mutating ``os.environ`` is allowed
    # (use ``not os.getenv(...)`` guards to preserve env > YAML precedence);
    # any returned dict is merged into ``PlatformConfig.extra``.  Exceptions
    # are caught and logged at debug level.
    # See website/docs/developer-guide/adding-platform-adapters.md for the
    # full contract and a worked example.
    apply_yaml_config_fn: Optional[Callable[[dict, dict], Optional[dict]]] = None

    # Optional: home-channel env var name for cron/notification delivery
    # (e.g. ``"IRC_HOME_CHANNEL"``).  When set, ``cron.scheduler`` treats this
    # platform as a valid ``deliver=<name>`` target and reads the env var to
    # resolve the default chat/room ID.  Empty = no cron home-channel support.
    cron_deliver_env_var: str = ""

    # ── Standalone (out-of-process) sending ──
    # Optional: async coroutine that delivers a message without a live
    # gateway adapter.  Called by ``tools/send_message_tool._send_via_adapter``
    # when ``cron`` runs in a separate process from the gateway and the
    # in-process adapter weakref is therefore ``None``.
    #
    # Signature:
    #     async (pconfig, chat_id, message, *, thread_id=None,
    #            media_files=None, force_document=False) -> dict
    #
    # Returns ``{"success": True, "message_id": ...}`` on success or
    # ``{"error": str}`` on failure.  Plugin authors typically open an
    # ephemeral connection / acquire a fresh OAuth token, send, and close.
    # Without this hook, plugin platforms cannot serve as cron ``deliver=``
    # targets when the gateway is not co-resident with the cron process.
    standalone_sender_fn: Optional[Callable[..., Awaitable[dict]]] = None


class PlatformRegistry:
    """Central registry of platform adapters.

    Thread-safe for reads (dict lookups are atomic under GIL).
    Writes happen at startup during sequential discovery.
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, PlatformEntry]] = {}
        # Deferred platform loaders: name -> zero-arg callable that imports the
        # owning plugin module (which calls register() and populates _entries).
        #
        # Why this exists: platform adapter modules import heavy, platform-
        # specific SDKs at module level (lark_oapi, microsoft_teams, discord.py,
        # slack_bolt, ...). Eagerly loading all ~20 bundled platform plugins at
        # plugin-discovery time added several seconds to *every* `hermes`
        # invocation -- including plain `hermes chat`, which never touches any
        # gateway platform. Discovery now registers a cheap deferred loader per
        # platform; the real module is imported only when a registry lookup
        # actually asks for that platform (gateway start, cron delivery,
        # `hermes setup`/`gateway status`, send_message).
        self._deferred: dict[str, dict[str, _DeferredPlatform]] = {}

    # -- deferred loading ----------------------------------------------------

    def register_deferred(
        self,
        name: str,
        loader: Callable[[], None],
        *,
        source: str = "plugin",
    ) -> None:
        """Register a lazy loader for a platform that hasn't been imported yet.

        *loader* is a zero-arg callable that imports the owning plugin module,
        which is expected to call :meth:`register` with the real entry for
        *name*.  The loader runs at most once, the first time *name* is looked
        up (or when the full entry list is materialized).  Concrete entries and
        deferred loaders coexist by source priority, so a deferred plugin can
        still override an already-registered built-in.
        """
        if source in self._entries.get(name, {}):
            # Already concretely registered for this source; no need to defer.
            return
        self._deferred.setdefault(name, {})[source] = _DeferredPlatform(
            loader=loader,
            source=source,
        )

    def _resolve(self, name: str, source: str | None = None) -> None:
        """Run pending deferred loaders for *name*."""
        pending_by_source = self._deferred.get(name)
        if not pending_by_source:
            return
        sources = [source] if source is not None else sorted(
            pending_by_source,
            key=_source_priority,
            reverse=True,
        )
        for pending_source in sources:
            pending = pending_by_source.pop(pending_source, None)
            if pending is None:
                continue
            try:
                pending.loader()
            except Exception as e:
                logger.warning(
                    "Deferred load of platform '%s' failed: %s",
                    name,
                    e,
                    exc_info=True,
                )
        if not pending_by_source:
            self._deferred.pop(name, None)

    def _resolve_all(self) -> None:
        """Run every pending deferred loader.

        Used by the iterate-all accessors (``all_entries``/``plugin_entries``),
        which are only called by paths that genuinely need every adapter:
        gateway startup, ``hermes setup``/``gateway status``, channel
        directory.  CLI chat never iterates the full set.
        """
        if not self._deferred:
            return
        # Snapshot keys -- loaders mutate _deferred as they resolve.
        for name in list(self._deferred):
            self._resolve(name)

    def register(self, entry: PlatformEntry) -> None:
        """Register a platform adapter entry.

        If an entry with the same name and source exists, it is replaced.
        Different sources coexist and are selected by explicit source priority.
        """
        # A concrete registration supersedes any pending deferred loader for
        # the same source, but not a lower/higher-priority entry for fallback.
        if entry.name in self._deferred:
            self._deferred[entry.name].pop(entry.source, None)
            if not self._deferred[entry.name]:
                self._deferred.pop(entry.name, None)

        entries = self._entries.setdefault(entry.name, {})
        if entry.source in entries:
            prev = entries[entry.source]
            logger.info(
                "Platform '%s' re-registered (was %s, now %s)",
                entry.name,
                prev.source,
                entry.source,
            )
        entries[entry.source] = entry
        logger.debug("Registered platform adapter: %s (%s)", entry.name, entry.source)

    def unregister(self, name: str) -> bool:
        """Remove a platform entry.  Returns True if it existed."""
        self._deferred.pop(name, None)
        return self._entries.pop(name, None) is not None

    def get(self, name: str) -> Optional[PlatformEntry]:
        """Look up a platform entry by name."""
        self._resolve(name)
        return self._highest_priority_entry(name)

    def all_entries(self) -> list[PlatformEntry]:
        """Return all registered platform entries."""
        self._resolve_all()
        return [
            entry
            for name in self._entries
            if (entry := self._highest_priority_entry(name)) is not None
        ]

    def plugin_entries(self) -> list[PlatformEntry]:
        """Return only plugin-registered platform entries."""
        self._resolve_all()
        return [e for e in self.all_entries() if e.source == "plugin"]

    def is_registered(self, name: str) -> bool:
        # A deferred (not-yet-imported) platform still counts as registered --
        # the loader will materialize it on first real use.  This keeps cheap
        # membership checks (toolset resolution, webhook deliver-target checks)
        # from triggering a heavy import.
        return name in self._entries or name in self._deferred

    def _highest_priority_entry(self, name: str) -> Optional[PlatformEntry]:
        entries = self._entries.get(name)
        if not entries:
            return None
        return max(entries.values(), key=lambda entry: _source_priority(entry.source))

    def _candidate_entries(self, name: str) -> list[PlatformEntry]:
        self._resolve(name)
        entries = self._entries.get(name)
        if not entries:
            return []
        return sorted(
            entries.values(),
            key=lambda entry: _source_priority(entry.source),
            reverse=True,
        )

    def create_adapter(self, name: str, config: Any) -> Optional[Any]:
        """Create an adapter instance for the given platform name.

        Returns None if:
        - No entry registered for *name*
        - check_fn() returns False (missing deps)
        - validate_config() returns False (misconfigured)
        - The factory raises an exception
        """
        entries = self._candidate_entries(name)
        if not entries:
            return None

        for index, entry in enumerate(entries):
            has_fallback = index + 1 < len(entries)
            if not self._requirements_available(entry):
                if has_fallback:
                    continue
                return None

            if entry.validate_config is not None:
                try:
                    if not entry.validate_config(config):
                        logger.warning(
                            "Platform '%s' config validation failed",
                            entry.label,
                        )
                        if has_fallback:
                            continue
                        return None
                except Exception as e:
                    logger.warning(
                        "Platform '%s' config validation error: %s",
                        entry.label,
                        e,
                    )
                    if has_fallback:
                        continue
                    return None

            try:
                adapter = entry.adapter_factory(config)
                return adapter
            except Exception as e:
                logger.error(
                    "Failed to create adapter for platform '%s': %s",
                    entry.label,
                    e,
                    exc_info=True,
                )
                if has_fallback:
                    continue
                return None
        return None

    def _requirements_available(self, entry: PlatformEntry) -> bool:
        try:
            available = entry.check_fn()
        except Exception as e:
            hint = f" ({entry.install_hint})" if entry.install_hint else ""
            logger.warning(
                "Platform '%s' requirements check failed%s: %s",
                entry.label,
                hint,
                e,
                exc_info=True,
            )
            return False
        if not available:
            hint = f" ({entry.install_hint})" if entry.install_hint else ""
            logger.warning(
                "Platform '%s' requirements not met%s",
                entry.label,
                hint,
            )
            return False
        return True


def _lazy_callable(module_name: str, attr_name: str) -> Callable[..., Any]:
    def _call(*args: Any, **kwargs: Any) -> Any:
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, attr_name)(*args, **kwargs)

    return _call


def _lazy_yuanbao_requirements() -> bool:
    import importlib

    module = importlib.import_module("gateway.platforms.yuanbao")
    return bool(getattr(module, "WEBSOCKETS_AVAILABLE", False))


def _register_builtin_platforms(registry: PlatformRegistry) -> None:
    builtin_specs = [
        (
            "whatsapp_cloud",
            "WhatsApp Cloud",
            "gateway.platforms.whatsapp_cloud",
            "WhatsAppCloudAdapter",
            "check_whatsapp_cloud_requirements",
            None,
            "aiohttp/httpx missing; reinstall hermes-agent",
        ),
        (
            "signal",
            "Signal",
            "gateway.platforms.signal",
            "SignalAdapter",
            "check_signal_requirements",
            "validate_signal_config",
            "signal-cli-rest-api not configured; set SIGNAL_HTTP_URL and SIGNAL_ACCOUNT",
        ),
        (
            "weixin",
            "Weixin",
            "gateway.platforms.weixin",
            "WeixinAdapter",
            "check_weixin_requirements",
            None,
            "aiohttp/cryptography not installed",
        ),
        (
            "api_server",
            "API Server",
            "gateway.platforms.api_server",
            "APIServerAdapter",
            "check_api_server_requirements",
            None,
            "aiohttp not installed",
        ),
        (
            "webhook",
            "Webhook",
            "gateway.platforms.webhook",
            "WebhookAdapter",
            "check_webhook_requirements",
            None,
            "aiohttp not installed",
        ),
        (
            "msgraph_webhook",
            "MSGraph webhook",
            "gateway.platforms.msgraph_webhook",
            "MSGraphWebhookAdapter",
            "check_msgraph_webhook_requirements",
            None,
            "aiohttp not installed",
        ),
        (
            "bluebubbles",
            "BlueBubbles",
            "gateway.platforms.bluebubbles",
            "BlueBubblesAdapter",
            "check_bluebubbles_requirements",
            None,
            "aiohttp/httpx missing or BLUEBUBBLES_SERVER_URL/BLUEBUBBLES_PASSWORD not configured",
        ),
        (
            "qqbot",
            "QQBot",
            "gateway.platforms.qqbot",
            "QQAdapter",
            "check_qq_requirements",
            None,
            "aiohttp/httpx missing or QQ_APP_ID/QQ_CLIENT_SECRET not configured",
        ),
    ]
    for name, label, module_name, adapter_name, check_name, validate_name, hint in builtin_specs:
        registry.register(
            PlatformEntry(
                name=name,
                label=label,
                adapter_factory=_lazy_callable(module_name, adapter_name),
                check_fn=_lazy_callable(module_name, check_name),
                validate_config=(
                    _lazy_callable(module_name, validate_name)
                    if validate_name is not None
                    else None
                ),
                install_hint=hint,
                source="builtin",
            )
        )

    registry.register(
        PlatformEntry(
            name="yuanbao",
            label="Yuanbao",
            adapter_factory=_lazy_callable("gateway.platforms.yuanbao", "YuanbaoAdapter"),
            check_fn=_lazy_yuanbao_requirements,
            install_hint="Run: pip install websockets",
            source="builtin",
        )
    )


# Module-level singleton
platform_registry = PlatformRegistry()
_register_builtin_platforms(platform_registry)
