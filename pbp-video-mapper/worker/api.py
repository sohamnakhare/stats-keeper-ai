"""HTTP API that runs one scorebug OCR job at a time."""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .download_video import download_video
from .ocr_reader import ScoreboardOCRReader
from .scoreboard_detector import ocr_scorebug_track, save_scoreboard_track
from .scorebug_api import fetch_scorebug_video

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
_WEBHOOK_ENV = "SCOREBUG_RESULT_WEBHOOK_URL"

app = FastAPI(title="PBP Video Mapper")

_lock = threading.Lock()
_jobs: dict[str, "Job"] = {}
_running_id: str | None = None
_reader: ScoreboardOCRReader | None = None
_reader_gpu: bool | None = None


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
    gpu: bool = True
    debug: bool = False


class DetectRequest(BaseModel):
    """Body for POST /detect."""

    video_id: str = Field(alias="id")
    fps: float = Field(default=1.0, gt=0)
    gpu: bool = True
    debug: bool = False

    model_config = {"populate_by_name": True}

    @field_validator("video_id")
    @classmethod
    def _require_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("id is required")
        return stripped


def result_webhook_url() -> str | None:
    """Return the result webhook, or None when it is unset."""
    configured = os.environ.get(_WEBHOOK_ENV, "").strip()
    return configured or None


def get_reader(gpu: bool) -> ScoreboardOCRReader:
    """Return the process-wide OCR reader, creating it on first use."""
    global _reader, _reader_gpu
    with _lock:
        if _reader is not None and _reader_gpu == gpu:
            return _reader
    created = ScoreboardOCRReader(gpu=gpu)
    with _lock:
        if _reader is None or _reader_gpu != gpu:
            _reader = created
            _reader_gpu = gpu
        return _reader


def reset_state() -> None:
    """Drop in-memory jobs and the cached reader. Used by tests."""
    global _running_id, _reader, _reader_gpu
    with _lock:
        _jobs.clear()
        _running_id = None
        _reader = None
        _reader_gpu = None


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
    """POST a terminal payload. Return an error string, or None on success.

    Returns None without calling the network when the webhook URL is unset.
    """
    url = result_webhook_url()
    if url is None:
        return None

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
        job = _jobs[job_id]
        video_id = job.video_id
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


def _run_job(job_id: str) -> None:
    with _lock:
        job = _jobs[job_id]
        video_id = job.video_id
        fps = job.fps
        gpu = job.gpu
        debug = job.debug
    try:
        video = fetch_scorebug_video(video_id)
        video_path = download_video(video.video_url)
        artifact = ocr_scorebug_track(
            video_path=video_path,
            scorebug=video.scorebug,
            regions=video.fields,
            sample_fps=fps,
            gpu=gpu,
            include_raw_ocr=debug,
            on_progress=lambda pct: _set_progress(job_id, pct),
            reader=get_reader(gpu),
        )
        output = OUTPUT_DIR / f"{job_id}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        save_scoreboard_track(artifact, output)
        _finish(
            job_id,
            status="done",
            result=artifact.model_dump(by_alias=True),
            error=None,
        )
    except Exception as exc:
        logger.exception("Detection failed for %s", video_id)
        _finish(job_id, status="failed", result=None, error=str(exc))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/detect", status_code=202)
def detect(body: DetectRequest) -> dict:
    """Start OCR for a scorebug video id."""
    global _running_id
    job = Job(
        job_id=uuid.uuid4().hex,
        video_id=body.video_id,
        fps=body.fps,
        gpu=body.gpu,
        debug=body.debug,
    )
    with _lock:
        if _running_id is not None:
            raise HTTPException(
                status_code=409,
                detail="A detection job is already running",
            )
        _jobs[job.job_id] = job
        _running_id = job.job_id
    threading.Thread(target=_run_job, args=(job.job_id,), daemon=True).start()
    return {"jobId": job.job_id, "status": "running"}


@app.get("/detect/{job_id}")
def get_detect(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_body(job)
