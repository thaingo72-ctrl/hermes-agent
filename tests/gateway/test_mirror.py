"""Tests for gateway/mirror.py — session mirroring."""

import json
from unittest.mock import MagicMock, patch

import gateway.mirror as mirror_mod
from gateway.mirror import (
    mirror_to_session,
    _find_session_id,
)


def _setup_state_db(tmp_path, sessions_data):
    """Seed gateway session rows in state.db, the only mirror lookup store."""
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    try:
        for key, entry in sessions_data.items():
            origin = entry["origin"]
            session_id = entry["session_id"]
            db.create_session(
                session_id,
                origin["platform"],
                user_id=origin.get("user_id"),
                session_key=key,
                chat_id=origin.get("chat_id"),
                chat_type=entry.get("chat_type"),
                thread_id=origin.get("thread_id"),
            )
            db.record_gateway_session_peer(
                session_id,
                source=origin["platform"],
                user_id=origin.get("user_id"),
                session_key=key,
                chat_id=origin.get("chat_id"),
                chat_type=entry.get("chat_type"),
                thread_id=origin.get("thread_id"),
                display_name=origin.get("chat_name") or origin.get("user_name"),
                origin_json=json.dumps(origin),
            )
    finally:
        db.close()
    return db_path


class TestFindSessionId:
    def test_finds_matching_session(self, tmp_path):
        db_path = _setup_state_db(tmp_path, {
            "agent:main:telegram:dm": {
                "session_id": "sess_abc",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })

        with patch("hermes_state.DEFAULT_DB_PATH", db_path):
            result = _find_session_id("telegram", "12345")

        assert result == "sess_abc"

    def test_returns_most_recent(self, tmp_path):
        db_path = _setup_state_db(tmp_path, {
            "old": {
                "session_id": "sess_old",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "new": {
                "session_id": "sess_new",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch("hermes_state.DEFAULT_DB_PATH", db_path):
            result = _find_session_id("telegram", "12345")

        assert result == "sess_new"

    def test_thread_id_disambiguates_same_chat(self, tmp_path):
        db_path = _setup_state_db(tmp_path, {
            "topic_a": {
                "session_id": "sess_topic_a",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "10"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "topic_b": {
                "session_id": "sess_topic_b",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "11"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch("hermes_state.DEFAULT_DB_PATH", db_path):
            result = _find_session_id("telegram", "-1001", thread_id="10")

        assert result == "sess_topic_a"


class TestMirrorToSession:


    def test_successful_mirror_uses_user_id_for_group_session(self, tmp_path):
        db_path = _setup_state_db(tmp_path, {
            "alice": {
                "session_id": "sess_alice",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "alice"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "bob": {
                "session_id": "sess_bob",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "bob"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch("hermes_state.DEFAULT_DB_PATH", db_path), \
             patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            result = mirror_to_session(
                "telegram",
                "-1001",
                "Hello group!",
                source_label="cli",
                user_id="alice",
            )

        assert result is True
        mock_sqlite.assert_called_once()
        assert mock_sqlite.call_args[0][0] == "sess_alice"

    def test_no_matching_session(self, tmp_path):
        db_path = _setup_state_db(tmp_path, {})

        with patch("hermes_state.DEFAULT_DB_PATH", db_path):
            result = mirror_to_session("telegram", "99999", "Hello!")

        assert result is False


class TestAppendToSqlite:
    def test_connection_is_closed_after_use(self, tmp_path):
        """Verify _append_to_sqlite closes the SessionDB connection."""
        from gateway.mirror import _append_to_sqlite
        mock_db = MagicMock()

        with patch("hermes_state.SessionDB", return_value=mock_db):
            _append_to_sqlite("sess_1", {"role": "assistant", "content": "hello"})

        mock_db.append_message.assert_called_once()
        mock_db.close.assert_called_once()
