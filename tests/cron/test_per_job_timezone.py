from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("croniter")

from cron.jobs import (
    compute_next_run,
    create_job,
    get_due_jobs,
    load_jobs,
    parse_schedule,
    save_jobs,
)


def test_parse_cron_timezone_prefix():
    parsed = parse_schedule("TZ=Asia/Ho_Chi_Minh 30 6 * * 1-5")
    assert parsed["kind"] == "cron"
    assert parsed["expr"] == "30 6 * * 1-5"
    assert parsed["timezone"] == "Asia/Ho_Chi_Minh"
    alias = parse_schedule("CRON_TZ=America/New_York 0 9 * * 1-5")
    assert alias["timezone"] == "America/New_York"
    with pytest.raises(ValueError, match="Invalid cron timezone"):
        parse_schedule("TZ=Not/A_Zone 0 9 * * *")
    with pytest.raises(ValueError, match="only supported for cron"):
        parse_schedule("TZ=Asia/Ho_Chi_Minh every 2h")


def test_next_run_uses_job_timezone(monkeypatch):
    pacific = ZoneInfo("America/Los_Angeles")
    now = datetime(2026, 8, 3, 16, 0, tzinfo=pacific)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    schedule = parse_schedule("TZ=Asia/Ho_Chi_Minh 30 6 * * 1-5")
    next_run = compute_next_run(schedule)
    assert next_run is not None
    result = datetime.fromisoformat(next_run)
    assert result == datetime(2026, 8, 4, 6, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))


def test_due_check_preserves_per_job_wall_clock(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron/jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron/output")
    pacific = ZoneInfo("America/Los_Angeles")
    now = datetime(2026, 8, 4, 5, 30, tzinfo=pacific)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    job = create_job(prompt="support", schedule="TZ=Asia/Ho_Chi_Minh 0 19 * * 1-5")
    jobs = load_jobs()
    jobs[0]["next_run_at"] = "2026-08-04T19:00:00+07:00"
    save_jobs(jobs)
    assert any(item["id"] == job["id"] for item in get_due_jobs())
