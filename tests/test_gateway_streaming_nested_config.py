"""Regression test for #25676 — nested gateway.streaming config must be loaded."""
import tempfile
from pathlib import Path

import yaml



def _load_with_yaml_dict(yaml_dict: dict):
    """Load gateway config through the canonical config.yaml path."""
    import os

    from hermes_cli import config as cfg_mod
    from hermes_cli import managed_scope
    from gateway.config import load_gateway_config

    tmp = tempfile.TemporaryDirectory()
    home = Path(tmp.name)
    (home / "config.yaml").write_text(yaml.safe_dump(yaml_dict), encoding="utf-8")
    os.environ["HERMES_HOME"] = str(home)
    os.environ["HERMES_MANAGED_DIR"] = str(home / "no-managed")
    cfg_mod._LOAD_CONFIG_CACHE.clear()
    cfg_mod._RAW_CONFIG_CACHE.clear()
    cfg_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    managed_scope.invalidate_managed_cache()
    try:
        return load_gateway_config()
    finally:
        tmp.cleanup()


class TestStreamingConfigNested:
    def test_gateway_streaming(self):
        cfg = _load_with_yaml_dict(
            {"gateway": {"streaming": {"enabled": True, "transport": "draft"}}}
        )
        assert cfg.streaming.enabled is True
        assert cfg.streaming.transport == "draft"


    def test_root_streaming_alias_is_ignored(self):
        cfg = _load_with_yaml_dict({
            "streaming": {"enabled": True, "transport": "edit"},
            "gateway": {"streaming": {"enabled": False, "transport": "draft"}},
        })
        assert cfg.streaming.enabled is False
        assert cfg.streaming.transport == "draft"


class TestStreamingModeAlias:
    """``streaming: {mode: ...}`` is an alias that also implies ``enabled``.

    Regression for a live config footgun: ``streaming: {mode: auto}`` was
    silently ignored (mode was never read), so streaming stayed disabled and
    the whole reply buffered before the first Telegram send.
    """

    def test_mode_auto_enables_streaming(self):
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict({"mode": "auto"})
        assert sc.enabled is True
        assert sc.transport == "auto"

    def test_mode_edit_enables_streaming(self):
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict({"mode": "edit"})
        assert sc.enabled is True
        assert sc.transport == "edit"



    def test_explicit_enabled_overrides_mode(self):
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict({"mode": "auto", "enabled": False})
        assert sc.enabled is False
        # transport still resolves from mode
        assert sc.transport == "auto"


    def test_empty_block_stays_disabled(self):
        from gateway.config import StreamingConfig

        sc = StreamingConfig.from_dict({})
        assert sc.enabled is False



class TestStreamingYamlBooleanQuirk:
    """YAML 1.1 parses bare ``off``/``on`` as booleans; ``mode``/``transport``
    must normalize those back to canonical string tokens.

    Regression for the review on PR #62873: bare ``mode: off`` arrived as
    Python ``False`` and stringified to ``"false"``, which is not ``"off"``,
    so streaming was enabled instead of honoring the advertised disable.
    """

    def test_mode_bare_off_boolean_disables(self):
        from gateway.config import StreamingConfig

        # yaml.safe_load("off") -> False
        sc = StreamingConfig.from_dict({"mode": False})
        assert sc.enabled is False
        assert sc.transport == "off"

    def test_mode_bare_on_boolean_enables(self):
        from gateway.config import StreamingConfig

        # yaml.safe_load("on") -> True
        sc = StreamingConfig.from_dict({"mode": True})
        assert sc.enabled is True
        assert sc.transport == "auto"


    def test_loader_normalizes_bare_yaml_off(self):
        """End-to-end through load_gateway_config(): unquoted ``mode: off``
        (a YAML boolean) must keep streaming disabled."""
        cfg = _load_with_yaml_dict({"gateway": {"streaming": {"mode": False}}})
        assert cfg.streaming.enabled is False
        assert cfg.streaming.transport == "off"
