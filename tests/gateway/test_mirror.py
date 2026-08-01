"""Tests for gateway/mirror.py — session mirroring."""

from unittest.mock import patch, MagicMock

from gateway.mirror import (
    mirror_to_session,
    _find_session_id,
)


class TestFindSessionId:
    def test_finds_matching_session_from_state_db(self):
        mock_db = MagicMock()
        mock_db.find_session_by_origin.return_value = "sess_abc"

        with patch("hermes_state.SessionDB", return_value=mock_db):
            result = _find_session_id("telegram", "12345")

        assert result == "sess_abc"
        mock_db.find_session_by_origin.assert_called_once_with(
            platform="telegram",
            chat_id="12345",
            thread_id=None,
            user_id=None,
        )
        mock_db.close.assert_called_once()

    def test_does_not_fallback_to_sessions_json(self, tmp_path):
        import json
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True)
        index_file = sessions_dir / "sessions.json"
        index_file.write_text(json.dumps({
            "agent:main:telegram:dm": {
                "session_id": "sess_json",
                "origin": {"platform": "telegram", "chat_id": "12345"},
            }
        }))
        mock_db = MagicMock()
        mock_db.find_session_by_origin.return_value = None

        with patch("hermes_state.SessionDB", return_value=mock_db):
            result = _find_session_id("telegram", "12345")

        assert result is None

    def test_thread_id_passed_to_state_db(self):
        mock_db = MagicMock()
        mock_db.find_session_by_origin.return_value = "sess_topic_a"

        with patch("hermes_state.SessionDB", return_value=mock_db):
            result = _find_session_id("telegram", "-1001", thread_id="10")

        assert result == "sess_topic_a"
        assert mock_db.find_session_by_origin.call_args.kwargs["thread_id"] == "10"


class TestMirrorToSession:


    def test_successful_mirror_uses_user_id_for_group_session(self):
        mock_db = MagicMock()
        mock_db.find_session_by_origin.return_value = "sess_alice"

        with patch("hermes_state.SessionDB", return_value=mock_db), \
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

    def test_no_matching_session(self):
        mock_db = MagicMock()
        mock_db.find_session_by_origin.return_value = None

        with patch("hermes_state.SessionDB", return_value=mock_db):
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
