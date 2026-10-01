"""Tests for the scorebug detect claim loop."""

from __future__ import annotations

import threading

from worker.detect_queue import pull_queue_enabled, run_detect_queue


def test_pull_queue_is_off_unless_explicitly_enabled(monkeypatch) -> None:
    monkeypatch.delenv("SCOREBUG_PULL_QUEUE", raising=False)
    assert pull_queue_enabled() is False
    monkeypatch.setenv("SCOREBUG_PULL_QUEUE", "0")
    assert pull_queue_enabled() is False
    monkeypatch.setenv("SCOREBUG_PULL_QUEUE", "1")
    assert pull_queue_enabled() is True


def test_loop_runs_a_claimed_id_then_stops_on_empty_queue() -> None:
    stop = threading.Event()
    claimed = ["vid-1"]
    ran: list[str] = []

    def claim() -> str | None:
        if not claimed:
            stop.set()
            return None
        return claimed.pop(0)

    def run_video(video_id: str) -> bool:
        ran.append(video_id)
        return True

    run_detect_queue(
        stop,
        is_busy=lambda: False,
        run_video=run_video,
        claim=claim,
        idle_seconds=0,
    )

    assert ran == ["vid-1"]


def test_loop_does_not_claim_while_a_job_is_running() -> None:
    stop = threading.Event()
    claims = {"count": 0}

    def claim() -> str | None:
        claims["count"] += 1
        return "vid-1"

    def is_busy() -> bool:
        stop.set()
        return True

    def run_video(video_id: str) -> bool:
        raise AssertionError(video_id)

    run_detect_queue(
        stop,
        is_busy=is_busy,
        run_video=run_video,
        claim=claim,
        idle_seconds=0,
    )

    assert claims["count"] == 0
