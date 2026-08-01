"""Regression tests for invalid gateway routing rows.

Corrupted ``gateway_routing`` entries (e.g. a bare bool where a dict is
expected) must not crash the entire session loading loop or block valid rows.
"""

import json
from pathlib import Path

from gateway.config import GatewayConfig
from gateway.session import SessionStore


class TestSessionLoadBoolCorruption:
    """Verify that non-dict routing entries are skipped, not fatal."""

    def _make_store(self, tmp_path: Path, sessions_data: dict) -> SessionStore:
        """Create a SessionStore with pre-populated state.db routing rows."""
        store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
        store._db.replace_gateway_routing_entries(
            {
                key: json.dumps(value) if not isinstance(value, str) else value
                for key, value in sessions_data.items()
            },
            scope=store._routing_scope(),
        )
        store._entries.clear()
        store._loaded = False
        return store

    def _valid_entry(self, session_id: str = "20260101_120000_abc12345") -> dict:
        return {
            "session_key": "agent:main:telegram:dm:123456",
            "session_id": session_id,
            "created_at": "2026-01-01T12:00:00",
            "updated_at": "2026-01-01T12:30:00",
            "origin": {
                "platform": "telegram",
                "chat_id": "123456",
                "chat_type": "dm",
            },
        }

    def test_bool_entry_skipped_not_fatal(self, tmp_path):
        """A bool entry must not crash the loop or block other sessions."""
        data = {
            "_README": "test sentinel",
            "corrupted_key": True,
            "valid_key": self._valid_entry(),
        }
        store = self._make_store(tmp_path, data)
        store._ensure_loaded()

        # The valid entry must still be loaded
        assert "valid_key" in store._entries
        assert store._entries["valid_key"].session_id == "20260101_120000_abc12345"
        # The corrupted entry must NOT be loaded
        assert "corrupted_key" not in store._entries

    def test_string_entry_skipped(self, tmp_path):
        """A string entry must also be skipped without crashing."""
        data = {
            "bad_string": json.dumps("not a dict"),
            "valid_key": self._valid_entry("20260101_130000_def67890"),
        }
        store = self._make_store(tmp_path, data)
        store._ensure_loaded()

        assert "valid_key" in store._entries
        assert "bad_string" not in store._entries

