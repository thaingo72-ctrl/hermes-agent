"""Acceptance coverage for concurrent cross-surface SessionDB opens."""

from __future__ import annotations

import json
import multiprocessing
import sqlite3
from pathlib import Path

from hermes_state import SessionDB


def _open_and_write(
    db_path: str,
    start_event,
    ready_queue,
    result_queue,
    worker_id: int,
    source: str,
) -> None:
    """Initialize the shared schema and persist one session from a child process."""
    ready_queue.put(worker_id)
    if not start_event.wait(timeout=15):
        result_queue.put((worker_id, "start-timeout"))
        return
    db = None
    try:
        db = SessionDB(db_path=Path(db_path))
        session_id = f"concurrent-{source}-{worker_id}"
        db.create_session(
            session_id=session_id,
            source=source,
            model_config={"surface": source, "worker": worker_id},
        )
        db.append_message(session_id, role="user", content=f"message-{worker_id}")
        result_queue.put((worker_id, "ok"))
    except Exception as exc:  # pragma: no cover - only returned to parent
        result_queue.put((worker_id, f"{type(exc).__name__}:{exc}"))
    finally:
        if db is not None:
            db.close()


def test_concurrent_gateway_cli_desktop_schema_opens_are_consistent(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    db_path = tmp_path / "state.db"
    start_event = ctx.Event()
    ready_queue = ctx.Queue()
    result_queue = ctx.Queue()
    sources = ["gateway", "cli", "desktop", "gateway", "cli", "desktop"]
    workers = [
        ctx.Process(
            target=_open_and_write,
            args=(str(db_path), start_event, ready_queue, result_queue, index, source),
        )
        for index, source in enumerate(sources)
    ]

    for worker in workers:
        worker.start()
    assert sorted(ready_queue.get(timeout=15) for _ in workers) == list(range(len(workers)))
    start_event.set()

    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0
    results = sorted(result_queue.get(timeout=5) for _ in workers)
    assert results == [(index, "ok") for index in range(len(workers))]

    con = sqlite3.connect(db_path)
    try:
        assert con.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        sessions = con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        message_fts = con.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        metadata = [
            json.loads(row[0])
            for row in con.execute("SELECT model_config FROM sessions ORDER BY id")
        ]
        assert sessions == len(workers)
        assert messages == len(workers)
        assert message_fts == messages
        assert sorted(item["worker"] for item in metadata) == list(range(len(workers)))
        assert {item["surface"] for item in metadata} == {"gateway", "cli", "desktop"}
    finally:
        con.close()


def test_latest_message_id_reads_newest_row_beyond_first_page(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("long", source="desktop")
        latest_assistant = None
        for index in range(105):
            role = "assistant" if index % 2 else "user"
            message_id = db.append_message("long", role=role, content=str(index))
            if role == "assistant":
                latest_assistant = message_id

        assert db.get_latest_message_id("long", role="assistant") == latest_assistant
    finally:
        db.close()
