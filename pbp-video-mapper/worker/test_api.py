"""HTTP tests for the one-at-a-time detect API."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from worker.api import app, reset_state
from worker.schemas import ScoreboardRegion, ScoreboardTrackArtifact
from worker.scorebug_api import ScorebugVideo


@pytest.fixture(autouse=True)
def _clean_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reset_state()
    monkeypatch.setattr("worker.api.OUTPUT_DIR", tmp_path)
    monkeypatch.delenv("SCOREBUG_RESULT_WEBHOOK_URL", raising=False)
    yield
    reset_state()


def _region(x: float, y: float, w: float, h: float) -> ScoreboardRegion:
    return ScoreboardRegion(x=x, y=y, w=w, h=h)


def _video() -> ScorebugVideo:
    return ScorebugVideo(
        video_url="https://cdn.example.com/game.mp4",
        scorebug=_region(0.1, 0.8, 0.4, 0.15),
        fields={"game_clock": _region(0.05, 0.2, 0.2, 0.5)},
    )


def _artifact() -> ScoreboardTrackArtifact:
    return ScoreboardTrackArtifact(
        video_path="/tmp/game.mp4",
        sample_fps=1.0,
        total_frames=2,
        readings=[],
        game_clock_to_video=[],
    )


def _wait(client: TestClient, job_id: str) -> dict:
    deadline = time.time() + 2
    body: dict = {}
    while time.time() < deadline:
        body = client.get(f"/detect/{job_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.01)
    raise AssertionError(body)


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_detect_finishes_and_posts_webhook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    posted: list[dict] = []

    def fake_urlopen(request: object, timeout: int = 30):
        assert timeout == 30
        posted.append(json.loads(request.data.decode()))  # type: ignore[attr-defined]

        class _Response:
            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

            def read(self) -> bytes:
                return b""

        return _Response()

    monkeypatch.setenv(
        "SCOREBUG_RESULT_WEBHOOK_URL",
        "https://app.example.com/hooks/scorebug",
    )
    monkeypatch.setattr("worker.api.fetch_scorebug_video", lambda video_id: _video())
    monkeypatch.setattr("worker.api.download_video", lambda url: Path("/tmp/game.mp4"))
    monkeypatch.setattr("worker.api.get_reader", lambda gpu: object())
    monkeypatch.setattr(
        "worker.api.ocr_scorebug_track",
        lambda **kwargs: _artifact(),
    )
    monkeypatch.setattr("worker.api.urllib.request.urlopen", fake_urlopen)

    client = TestClient(app)
    started = client.post("/detect", json={"id": "vid-1", "fps": 2.0})
    assert started.status_code == 202
    job_id = started.json()["jobId"]
    assert started.json()["status"] == "running"

    body = _wait(client, job_id)
    assert body["status"] == "done"
    assert body["videoId"] == "vid-1"
    assert body["progress"] == 100
    assert body["result"]["videoPath"] == "/tmp/game.mp4"
    assert body["result"]["sampleFps"] == 1.0
    assert posted == [
        {
            "jobId": job_id,
            "videoId": "vid-1",
            "status": "done",
            "result": body["result"],
        }
    ]
    saved = json.loads((tmp_path / f"{job_id}.json").read_text())
    assert saved["videoPath"] == "/tmp/game.mp4"


def test_second_detect_is_rejected_while_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def block_ocr(**kwargs: object) -> ScoreboardTrackArtifact:
        started.set()
        release.wait(timeout=2)
        return _artifact()

    monkeypatch.setattr("worker.api.fetch_scorebug_video", lambda video_id: _video())
    monkeypatch.setattr("worker.api.download_video", lambda url: Path("/tmp/game.mp4"))
    monkeypatch.setattr("worker.api.get_reader", lambda gpu: object())
    monkeypatch.setattr("worker.api.ocr_scorebug_track", block_ocr)

    client = TestClient(app)
    first = client.post("/detect", json={"id": "vid-1"})
    assert first.status_code == 202
    assert started.wait(timeout=2)

    second = client.post("/detect", json={"id": "vid-2"})
    assert second.status_code == 409

    release.set()
    assert _wait(client, first.json()["jobId"])["status"] == "done"


def test_failed_job_posts_error_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[dict] = []

    def fake_urlopen(request: object, timeout: int = 30):
        posted.append(json.loads(request.data.decode()))  # type: ignore[attr-defined]

        class _Response:
            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

            def read(self) -> bytes:
                return b""

        return _Response()

    def fail_fetch(video_id: str) -> ScorebugVideo:
        raise RuntimeError(f"missing {video_id}")

    monkeypatch.setenv("SCOREBUG_RESULT_WEBHOOK_URL", "https://app.example.com/hook")
    monkeypatch.setattr("worker.api.fetch_scorebug_video", fail_fetch)
    monkeypatch.setattr("worker.api.urllib.request.urlopen", fake_urlopen)

    client = TestClient(app)
    started = client.post("/detect", json={"id": "vid-9"})
    body = _wait(client, started.json()["jobId"])

    assert body["status"] == "failed"
    assert body["error"] == "missing vid-9"
    assert posted == [
        {
            "jobId": started.json()["jobId"],
            "videoId": "vid-9",
            "status": "failed",
            "error": "missing vid-9",
        }
    ]


def test_skips_webhook_when_url_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_urlopen(request: object, timeout: int = 30):
        raise AssertionError("webhook should not be called")

    monkeypatch.setattr("worker.api.fetch_scorebug_video", lambda video_id: _video())
    monkeypatch.setattr("worker.api.download_video", lambda url: Path("/tmp/game.mp4"))
    monkeypatch.setattr("worker.api.get_reader", lambda gpu: object())
    monkeypatch.setattr("worker.api.ocr_scorebug_track", lambda **kwargs: _artifact())
    monkeypatch.setattr("worker.api.urllib.request.urlopen", fail_urlopen)

    client = TestClient(app)
    started = client.post("/detect", json={"id": "vid-1"})
    body = _wait(client, started.json()["jobId"])
    assert body["status"] == "done"
    assert "webhookError" not in body


def test_unknown_job_is_not_found() -> None:
    client = TestClient(app)
    response = client.get("/detect/missing")
    assert response.status_code == 404
