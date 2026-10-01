"""macOS HTTP API that runs one Vision OCR job at a time."""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from mac import vision_ocr
from mac.track import build_track
from worker.detect_queue import start_detect_queue
from worker.scorebug_api import fetch_scorebug_video, score_timeline_url

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Show Vision read logs under uvicorn, which only configures its own logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    for name in (
        "mac.api",
        "mac.track",
        "mac.vision_ocr",
        "worker.detect_queue",
        "worker.scorebug_api",
    ):
        log = logging.getLogger(name)
        log.setLevel(logging.INFO)
        if not any(isinstance(item, logging.StreamHandler) for item in log.handlers):
            log.addHandler(handler)
        log.propagate = False


_configure_logging()

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PACKAGE_ROOT / "worker" / "output"


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    stop = threading.Event()
    thread = start_detect_queue(
        stop,
        is_busy=_detection_busy,
        run_video=_run_claimed_video,
    )
    yield
    stop.set()
    if thread is not None:
        thread.join(timeout=2)


app = FastAPI(title="PBP Video Mapper (Vision)", lifespan=_lifespan)

_lock = threading.Lock()
_jobs: dict[str, Job] = {}
_running_id: str | None = None


@dataclass
class Job:
    """One detection request and its outcome."""

    job_id: str
    video_id: str
    status: str = "running"
    progress: int = 0
    result: dict | None = None
    error: str | None = None
    webhook_error: str | None = None
    fps: float = 1.0
    debug: bool = False
    done: threading.Event = field(default_factory=threading.Event)


class DetectRequest(BaseModel):
    """Body for POST /detect."""

    video_id: str = Field(alias="id")
    fps: float = Field(default=1.0, gt=0)
    debug: bool = False

    model_config = {"populate_by_name": True}

    @field_validator("video_id")
    @classmethod
    def _require_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("id is required")
        return stripped


def reset_state() -> None:
    """Drop in-memory jobs and the cached recognizer. Used by tests."""
    global _running_id
    with _lock:
        _jobs.clear()
        _running_id = None
    vision_ocr.reset_recognizer()


def _job_body(job: Job) -> dict:
    body: dict = {
        "jobId": job.job_id,
        "videoId": job.video_id,
        "status": job.status,
        "progress": job.progress,
    }
    if job.result is not None:
        body["result"] = job.result
    if job.error is not None:
        body["error"] = job.error
    if job.webhook_error is not None:
        body["webhookError"] = job.webhook_error
    return body


def post_result_webhook(payload: dict) -> str | None:
    """POST a terminal payload to the scorebug score-timeline route.

    The URL is ``{SCOREBUG_API_BASE}/api/scorebug-videos/{videoId}/score-timeline``.
    """
    video_id = str(payload.get("videoId") or "").strip()
    if not video_id:
        return "Webhook failed: video id is missing"
    url = score_timeline_url(video_id)

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        message = f"Webhook failed ({exc.code})"
        logger.error("%s for job %s", message, payload.get("jobId"))
        return message
    except Exception as exc:
        message = str(exc)
        logger.error("Webhook failed for job %s: %s", payload.get("jobId"), message)
        return message
    return None


def _set_progress(job_id: str, progress: int) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None and job.status == "running":
            job.progress = progress


def _finish(job_id: str, *, status: str, result: dict | None, error: str | None) -> None:
    global _running_id
    with _lock:
        video_id = _jobs[job_id].video_id
    payload: dict = {
        "jobId": job_id,
        "videoId": video_id,
        "status": status,
    }
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error
    webhook_error = post_result_webhook(payload)
    with _lock:
        job = _jobs[job_id]
        job.status = status
        job.progress = 100 if status == "done" else job.progress
        job.result = result
        job.error = error
        job.webhook_error = webhook_error
        if _running_id == job_id:
            _running_id = None
        job.done.set()


def _run_job(job_id: str) -> None:
    with _lock:
        job = _jobs[job_id]
        video_id = job.video_id
        fps = job.fps
        debug = job.debug
    logger.info("detect start video=%s fps=%s", video_id, fps)
    try:
        video = fetch_scorebug_video(video_id)
        crop_dir = OUTPUT_DIR / "vision-crops" / job_id
        artifact = build_track(
            video_path=video.video_url,
            scorebug=video.scorebug,
            regions=video.fields,
            sample_fps=fps,
            include_raw_ocr=debug,
            on_progress=lambda pct: _set_progress(job_id, pct),
            recognizer=vision_ocr.get_recognizer(),
            crop_dir=crop_dir,
            video_url=video.video_url,
        )
        output = OUTPUT_DIR / f"{job_id}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            json.dump(artifact.model_dump(by_alias=True), handle, indent=2)
        _finish(
            job_id,
            status="done",
            result=artifact.model_dump(by_alias=True),
            error=None,
        )
    except Exception as exc:
        logger.exception("Vision detection failed for %s", video_id)
        _finish(job_id, status="failed", result=None, error=str(exc))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _terminal_body(job: Job) -> dict:
    """Same JSON posted to the result webhook."""
    body: dict = {
        "jobId": job.job_id,
        "videoId": job.video_id,
        "status": job.status,
    }
    if job.result is not None:
        body["result"] = job.result
    if job.error is not None:
        body["error"] = job.error
    return body


def _detection_busy() -> bool:
    with _lock:
        return _running_id is not None


def _register_job(video_id: str, *, fps: float = 1.0, debug: bool = False) -> Job | None:
    """Reserve the single running slot, or return None when one is taken."""
    global _running_id
    job = Job(
        job_id=uuid.uuid4().hex,
        video_id=video_id,
        fps=fps,
        debug=debug,
    )
    with _lock:
        if _running_id is not None:
            return None
        _jobs[job.job_id] = job
        _running_id = job.job_id
    return job


def _run_claimed_video(video_id: str) -> bool:
    """Run one claimed video on this thread. False when a job is already running."""
    job = _register_job(video_id)
    if job is None:
        return False
    _run_job(job.job_id)
    return True


@app.post("/detect")
def detect(body: DetectRequest) -> dict:
    """Run Vision OCR for a scorebug video id and return the webhook payload."""
    job = _register_job(body.video_id, fps=body.fps, debug=body.debug)
    if job is None:
        raise HTTPException(
            status_code=409,
            detail="A detection job is already running",
        )
    threading.Thread(target=_run_job, args=(job.job_id,), daemon=True).start()
    job.done.wait()
    with _lock:
        return _terminal_body(job)


@app.get("/detect/{job_id}")
def get_detect(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_body(job)
