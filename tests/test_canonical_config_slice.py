"""Behavioral regressions for the canonical typed configuration slice."""

import importlib
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError


def _write_config(home, payload):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _reset_config_modules():
    import hermes_cli.config as cli_config

    cli_config._LOAD_CONFIG_CACHE.clear()
    cli_config._RAW_CONFIG_CACHE.clear()
    cli_config._LAST_EXPANDED_CONFIG_BY_PATH.clear()


def test_gateway_profile_routes_json_roundtrip_with_semantic_equality(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    route = {
        "name": "eng-discord",
        "platform": "discord",
        "profile": "engineering",
        "guild_id": "guild-1",
        "chat_id": "chan-2",
        "thread_id": "thread-3",
        "enabled": True,
    }
    _write_config(home, {"gateway": {"profile_routes": [route]}})
    monkeypatch.setenv("HERMES_HOME", str(home))
    _reset_config_modules()

    from gateway.config import load_gateway_config

    loaded = load_gateway_config()
    dumped = loaded.model_dump(mode="json")
    encoded = json.dumps(dumped)
    reloaded = type(loaded).model_validate(json.loads(encoded))

    assert reloaded.profile_routes == loaded.profile_routes
    assert reloaded.model_dump(mode="json")["profile_routes"] == [route]


def test_load_typed_config_can_explicitly_suppress_user_config(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    _write_config(home, {"model": "openrouter/user-model", "agent": {"max_turns": 17}})
    monkeypatch.setenv("HERMES_HOME", str(home))
    _reset_config_modules()

    from hermes_cli.config import load_typed_config

    included = load_typed_config(include_user_config=True)
    suppressed = load_typed_config(include_user_config=False)

    assert included.model.default == "openrouter/user-model"
    assert included.agent.max_turns == 17
    assert suppressed.model.default != "openrouter/user-model"
    assert suppressed.agent.max_turns != 17


def test_supported_scalar_model_yaml_normalizes_and_cli_initializes(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    _write_config(home, {"model": "openrouter/scalar-model"})
    monkeypatch.setenv("HERMES_HOME", str(home))
    _reset_config_modules()

    from hermes_cli.config import load_typed_config

    typed = load_typed_config()
    assert typed.model.model_dump(mode="json") == {
        "default": "openrouter/scalar-model",
        "base_url": "",
        "provider": "auto",
    }

    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "dotenv": MagicMock(load_dotenv=lambda *args, **kwargs: True),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(os.environ, {"HERMES_HOME": str(home)}, clear=False):
        import cli as cli_mod

        cli_mod = importlib.reload(cli_mod)
        with patch.object(cli_mod, "get_tool_definitions", return_value=[]):
            cli_obj = cli_mod.HermesCLI()

    assert cli_obj.model == "openrouter/scalar-model"


def test_env_expanded_scalars_coerce_valid_values_before_strict_validation(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    _write_config(
        home,
        {
            "agent": {"max_turns": "${HERMES_TEST_MAX_TURNS}"},
            "gateway": {
                "loop_watchdog": "${HERMES_TEST_LOOP_WATCHDOG}",
                "max_concurrent_sessions": "${HERMES_TEST_MAX_CONCURRENT}",
            },
        },
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_TEST_MAX_TURNS", "42")
    monkeypatch.setenv("HERMES_TEST_LOOP_WATCHDOG", "false")
    monkeypatch.setenv("HERMES_TEST_MAX_CONCURRENT", "8")
    _reset_config_modules()

    from hermes_cli.config import load_typed_config
    from gateway.config import load_gateway_config

    typed = load_typed_config()
    gateway = load_gateway_config()

    assert typed.agent.max_turns == 42
    assert typed.gateway.loop_watchdog is False
    assert gateway.max_concurrent_sessions == 8
    assert gateway.loop_watchdog is False


def test_invalid_env_expanded_scalar_reports_precise_diagnostic_and_gateway_keeps_valid_fields(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    _write_config(
        home,
        {
            "agent": {"max_turns": "${HERMES_TEST_BAD_MAX_TURNS}"},
            "gateway": {"loop_watchdog": False, "max_concurrent_sessions": "6"},
        },
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_TEST_BAD_MAX_TURNS", "forty-two")
    _reset_config_modules()

    from hermes_cli.config import load_typed_config
    from gateway.config import load_gateway_config

    with pytest.raises(ValidationError) as excinfo:
        load_typed_config()
    errors = excinfo.value.errors()
    assert errors[0]["loc"] == ("agent", "max_turns")
    assert "valid integer" in errors[0]["msg"].lower()

    gateway = load_gateway_config()
    assert gateway.loop_watchdog is False
    assert gateway.max_concurrent_sessions == 6
