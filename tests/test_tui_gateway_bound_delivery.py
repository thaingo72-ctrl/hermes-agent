"""Desktop/TUI delivery obligations for gateway-bound sessions."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from gateway import delivery_ledger as dl
from tui_gateway import server


def _session_db(row, message_id):
    return SimpleNamespace(
        get_session=Mock(return_value=row),
        get_latest_message_id=Mock(return_value=message_id),
    )


def test_bound_telegram_reply_records_stable_external_obligation(monkeypatch):
    db = _session_db(
        {
            "session_key": "agent:main:telegram:dm:123",
            "source": "telegram",
            "chat_id": "123",
            "thread_id": None,
        },
        42,
    )
    compute = Mock(return_value="stable-obligation-id")
    record = Mock()
    monkeypatch.setattr(server, "_session_db", lambda _session: nullcontext(db))
    monkeypatch.setattr(dl, "ledger_enabled", lambda: True)
    monkeypatch.setattr(dl, "compute_obligation_id", compute)
    monkeypatch.setattr(dl, "record_obligation", record)

    server._record_bound_platform_delivery(
        {"session_key": "desktop-session-id"}, "  reply from desktop  "
    )

    compute.assert_called_once_with(
        "agent:main:telegram:dm:123", "42", "reply from desktop"
    )
    record.assert_called_once_with(
        obligation_id="stable-obligation-id",
        session_key="agent:main:telegram:dm:123",
        platform="telegram",
        chat_id="123",
        thread_id=None,
        content="reply from desktop",
        external_owner=True,
    )


def test_desktop_only_reply_creates_no_delivery_obligation(monkeypatch):
    db = _session_db(
        {
            "session_key": "desktop-session-id",
            "source": "desktop",
            "chat_id": None,
            "thread_id": None,
        },
        7,
    )
    record = Mock()
    monkeypatch.setattr(server, "_session_db", lambda _session: nullcontext(db))
    monkeypatch.setattr(dl, "ledger_enabled", lambda: True)
    monkeypatch.setattr(dl, "record_obligation", record)

    server._record_bound_platform_delivery(
        {"session_key": "desktop-session-id"}, "local reply"
    )

    record.assert_not_called()
