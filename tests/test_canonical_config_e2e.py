"""End-to-end invariants for the canonical typed config loader."""

from __future__ import annotations

import importlib
import os
import sys
import textwrap

import pytest
from pydantic import ValidationError


def _drop_cli_module() -> None:
    sys.modules.pop("cli", None)


def _clear_config_caches() -> None:
    from hermes_cli import config as cfg_mod
    from hermes_cli import managed_scope

    cfg_mod._LOAD_CONFIG_CACHE.clear()
    cfg_mod._RAW_CONFIG_CACHE.clear()
    cfg_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    managed_scope.invalidate_managed_cache()


def test_cli_gateway_and_canonical_loader_share_one_config_path(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    managed_home = tmp_path / "managed"
    hermes_home.mkdir()
    managed_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_home))
    monkeypatch.setenv("CANONICAL_MODEL_ID", "test/env-expanded")
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    _drop_cli_module()
    _clear_config_caches()

    (hermes_home / "config.yaml").write_text(
        textwrap.dedent(
            """
            model:
              default: ${CANONICAL_MODEL_ID}
              provider: openrouter
            terminal:
              backend: docker
              cwd: /workspace/user
            auxiliary:
              vision:
                provider: openai
                model: gpt-4.1-mini
            gateway:
              multiplex_profiles: false
              max_concurrent_sessions: 7
              streaming:
                mode: draft
              platforms:
                api_server:
                  enabled: true
                  key: abcdefghijklmnop
                  port: 8777
            platforms:
              slack:
                enabled: true
                token: xoxb-user
                extra:
                  plugin_owned: preserved
            """
        ),
        encoding="utf-8",
    )
    (managed_home / "config.yaml").write_text(
        textwrap.dedent(
            """
            model:
              provider: managed-provider
            gateway:
              multiplex_profiles: true
            """
        ),
        encoding="utf-8",
    )

    from hermes_cli.config import load_config
    from gateway.config import Platform, load_gateway_config

    canonical = load_config()
    cli_mod = importlib.import_module("cli")
    gateway = load_gateway_config()

    assert cli_mod.CLI_CONFIG["model"] == canonical["model"]
    assert cli_mod.CLI_CONFIG["model"]["default"] == "test/env-expanded"
    assert cli_mod.CLI_CONFIG["model"]["provider"] == "managed-provider"
    assert cli_mod.CLI_CONFIG["terminal"]["backend"] == "docker"
    assert cli_mod.CLI_CONFIG["terminal"]["cwd"] == "/workspace/user"
    assert os.environ["TERMINAL_ENV"] == "docker"

    assert gateway.multiplex_profiles is True
    assert gateway.max_concurrent_sessions == 7
    assert gateway.streaming.enabled is True
    assert gateway.streaming.transport == "draft"
    assert gateway.platforms[Platform.API_SERVER].extra["port"] == 8777
    assert gateway.platforms[Platform.SLACK].token == "xoxb-user"
    assert gateway.platforms[Platform.SLACK].extra["plugin_owned"] == "preserved"


def test_canonical_loader_rejects_malformed_typed_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "no-managed"))
    _clear_config_caches()
    (tmp_path / "config.yaml").write_text(
        "terminal: not-a-section\n",
        encoding="utf-8",
    )

    from hermes_cli.config import load_config

    with pytest.raises(ValidationError):
        load_config()


def test_canonical_loader_rejects_malformed_gateway_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "no-managed"))
    _clear_config_caches()
    (tmp_path / "config.yaml").write_text(
        "gateway:\n  max_concurrent_sessions: many\n",
        encoding="utf-8",
    )

    from hermes_cli.config import load_config

    with pytest.raises(ValidationError):
        load_config()


def test_gateway_uses_canonical_gateway_section_not_obsolete_top_level_aliases(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "no-managed"))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    _clear_config_caches()
    (tmp_path / "config.yaml").write_text(
        textwrap.dedent(
            """
            group_sessions_per_user: false
            max_concurrent_sessions: 99
            gateway:
              group_sessions_per_user: true
              max_concurrent_sessions: 5
            """
        ),
        encoding="utf-8",
    )

    from pydantic import BaseModel
    from gateway.config import GatewayConfig, load_gateway_config

    config = load_gateway_config()

    assert isinstance(config, BaseModel)
    assert issubclass(GatewayConfig, BaseModel)
    assert config.group_sessions_per_user is True
    assert config.max_concurrent_sessions == 5
