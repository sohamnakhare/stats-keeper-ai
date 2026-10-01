"""Claim queued scorebug videos and run one detect job at a time."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable

from .scorebug_api import claim_next_scorebug_video

logger = logging.getLogger(__name__)

_IDLE_SECONDS = 5
_BUSY_SECONDS = 1


def pull_queue_enabled() -> bool:
    """Return True only when ``SCOREBUG_PULL_QUEUE=1``.

    The claim loop is off by default so a normal service start does not call
    ``POST /api/scorebug-videos/next``.
    """
    return os.environ.get("SCOREBUG_PULL_QUEUE", "").strip() == "1"


def run_detect_queue(
    stop: threading.Event,
    *,
    is_busy: Callable[[], bool],
    run_video: Callable[[str], bool],
    claim: Callable[[], str | None] = claim_next_scorebug_video,
    idle_seconds: float = _IDLE_SECONDS,
) -> None:
    """Claim a video, run it, then claim the next one.

    A 204 claim waits ``idle_seconds`` so a video queued while this process
    is idle is still picked up. Nothing is claimed while ``is_busy`` is true.
    """
    while not stop.is_set():
        if is_busy():
            if stop.wait(_BUSY_SECONDS):
                return
            continue
        video_id = claim()
        if not video_id:
            if stop.wait(idle_seconds):
                return
            continue
        logger.info("claimed scorebug video %s", video_id)
        if not run_video(video_id):
            if stop.wait(_BUSY_SECONDS):
                return


def start_detect_queue(
    stop: threading.Event,
    *,
    is_busy: Callable[[], bool],
    run_video: Callable[[str], bool],
) -> threading.Thread | None:
    """Start the claim loop when ``SCOREBUG_PULL_QUEUE=1``."""
    if not pull_queue_enabled():
        return None
    thread = threading.Thread(
        target=run_detect_queue,
        args=(stop,),
        kwargs={"is_busy": is_busy, "run_video": run_video},
        name="scorebug-detect-queue",
        daemon=True,
    )
    thread.start()
    return thread
