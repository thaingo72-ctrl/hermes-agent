"""Built-in gateway platforms must be constructed through PlatformEntry."""

from __future__ import annotations

from unittest.mock import MagicMock
import inspect

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.run import GatewayRunner


FORMER_HARDCODED_BUILTINS = [
    Platform.WHATSAPP_CLOUD,
    Platform.SIGNAL,
    Platform.WEIXIN,
    Platform.API_SERVER,
    Platform.WEBHOOK,
    Platform.MSGRAPH_WEBHOOK,
    Platform.BLUEBUBBLES,
    Platform.QQBOT,
    Platform.YUANBAO,
]


@pytest.fixture
def isolated_registry():
    original_entries = dict(platform_registry._entries)
    original_deferred = dict(platform_registry._deferred)
    platform_registry._entries.clear()
    platform_registry._deferred.clear()
    try:
        yield platform_registry
    finally:
        platform_registry._entries.clear()
        platform_registry._entries.update(original_entries)
        platform_registry._deferred.clear()
        platform_registry._deferred.update(original_deferred)


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    return runner


def test_former_hardcoded_builtins_register_as_platform_entries(isolated_registry):
    from gateway.platform_registry import register_builtin_platforms

    register_builtin_platforms(isolated_registry)

    entries = {
        platform.value: isolated_registry.get(platform.value)
        for platform in FORMER_HARDCODED_BUILTINS
    }
    assert set(entries) == {platform.value for platform in FORMER_HARDCODED_BUILTINS}
    for name, entry in entries.items():
        assert isinstance(entry, PlatformEntry), name
        assert entry.source == "builtin"
        assert callable(entry.adapter_factory)


def test_builtin_registration_preserves_plugin_override_precedence(isolated_registry):
    from gateway.platform_registry import register_builtin_platforms

    plugin_entry = PlatformEntry(
        name=Platform.WEBHOOK.value,
        label="Plugin Webhook",
        adapter_factory=lambda cfg: MagicMock(name="plugin-webhook"),
        check_fn=lambda: True,
        source="plugin",
    )
    isolated_registry.register(plugin_entry)
    isolated_registry.register_deferred(Platform.SIGNAL.value, lambda: None)

    register_builtin_platforms(isolated_registry)

    assert isolated_registry.get(Platform.WEBHOOK.value) is plugin_entry
    assert Platform.SIGNAL.value not in isolated_registry._entries
    assert Platform.SIGNAL.value in isolated_registry._deferred


@pytest.mark.parametrize("platform", FORMER_HARDCODED_BUILTINS)
def test_former_hardcoded_builtin_is_created_via_platform_registry(
    isolated_registry,
    platform: Platform,
):
    adapter = MagicMock(name=f"{platform.value}-adapter")
    factory = MagicMock(return_value=adapter)
    config = PlatformConfig(enabled=True)
    isolated_registry.register(
        PlatformEntry(
            name=platform.value,
            label=platform.value,
            adapter_factory=factory,
            check_fn=lambda: True,
            source="builtin",
        )
    )

    created = _runner()._create_adapter(platform, config)

    assert created is adapter
    factory.assert_called_once_with(config)
    assert adapter.gateway_runner is not None


def test_create_adapter_has_no_former_builtin_fallback_branches():
    source = inspect.getsource(GatewayRunner._create_adapter)
    for platform in FORMER_HARDCODED_BUILTINS:
        assert f"Platform.{platform.name}" not in source
    assert "Fall through to built-in" not in source


def test_registered_builtin_creation_failure_does_not_fall_through(
    isolated_registry,
):
    factory = MagicMock(return_value=None)
    isolated_registry.register(
        PlatformEntry(
            name=Platform.WEBHOOK.value,
            label="Webhook",
            adapter_factory=factory,
            check_fn=lambda: True,
            source="builtin",
        )
    )

    assert _runner()._create_adapter(Platform.WEBHOOK, PlatformConfig(enabled=True)) is None
    factory.assert_called_once()
