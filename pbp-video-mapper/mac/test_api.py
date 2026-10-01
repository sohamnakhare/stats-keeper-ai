"""HTTP tests for the macOS Vision detect API."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mac.api import app, reset_state
from worker.extract_frames import SampledFrame
from worker.schemas import ScoreboardRegion
from worker.scorebug_api import ScorebugVideo


@pytest.fixture(autouse=True)
def _clean_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    reset_state()
    monkeypatch.setattr("mac.api.OUTPUT_DIR", tmp_path)
    monkeypatch.setenv("SCOREBUG_API_BASE", "http://localhost:3000")
    monkeypatch.setenv("SCOREBUG_PULL_QUEUE", "0")
    monkeypatch.setattr("mac.api.urllib.request.urlopen", lambda request, timeout=30: _Response())
    yield
    reset_state()


def _region(x: float, y: float, w: float, h: float) -> ScoreboardRegion:
    return ScoreboardRegion(x=x, y=y, w=w, h=h)


def _video() -> ScorebugVideo:
    return ScorebugVideo(
        video_url="https://cdn.example.com/game.mp4",
        scorebug=_region(0.1, 0.8, 0.4, 0.15),
        fields={
            "game_clock": _region(0.05, 0.2, 0.2, 0.5),
            "home_score": _region(0.3, 0.2, 0.2, 0.5),
            "away_score": _region(0.55, 0.2, 0.2, 0.5),
        },
    )


def _frames(url: str, sample_fps: float = 1.0, scorebug: ScoreboardRegion | None = None):
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    yield SampledFrame(index=0, t_sec=1.0, bgr=image)


def _raw_regions(image: np.ndarray, regions: dict) -> dict[str, str]:
    return {"game_clock": "9:58", "home_score": "12", "away_score": "8"}


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return b""


def test_detect_finishes_and_posts_webhook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    posted: list[dict] = []

    def fake_urlopen(request: object, timeout: int = 30) -> _Response:
        assert timeout == 30
        posted.append(
            (
                request.full_url,  # type: ignore[attr-defined]
                json.loads(request.data.decode()),  # type: ignore[attr-defined]
            )
        )
        return _Response()

    monkeypatch.setattr("mac.api.fetch_scorebug_video", lambda video_id: _video())
    monkeypatch.setattr("mac.track.iter_remote_frames", _frames)
    monkeypatch.setattr("mac.track.remote_duration_seconds", lambda url: 1.0)
    monkeypatch.setattr("mac.vision_ocr.recognize_regions", _raw_regions)
    monkeypatch.setattr("mac.api.urllib.request.urlopen", fake_urlopen)

    client = TestClient(app)
    response = client.post("/detect", json={"id": "vid-1", "fps": 2.0})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"
    assert body["videoId"] == "vid-1"
    assert body["result"]["totalFrames"] == 1
    assert body["result"]["readings"][0]["homeScore"] == 12
    assert body["result"]["readings"][0]["gameClock"] == "9:58"
    assert body["result"]["gameClockToVideo"][0]["gameClock"] == "9:58"
    assert posted == [
        ("http://localhost:3000/api/scorebug-videos/vid-1/score-timeline", body)
    ]
    saved = json.loads((tmp_path / f"{body['jobId']}.json").read_text())
    assert saved["videoPath"] == "https://cdn.example.com/game.mp4"


def test_second_detect_is_rejected_while_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def block_regions(image: np.ndarray, regions: dict) -> dict[str, str]:
        started.set()
        release.wait(timeout=2)
        return _raw_regions(image, regions)

    monkeypatch.setattr("mac.api.fetch_scorebug_video", lambda video_id: _video())
    monkeypatch.setattr("mac.track.iter_remote_frames", _frames)
    monkeypatch.setattr("mac.track.remote_duration_seconds", lambda url: 1.0)
    monkeypatch.setattr("mac.vision_ocr.recognize_regions", block_regions)

    client = TestClient(app)
    finished: dict[str, object] = {}

    def first_call() -> None:
        finished["response"] = TestClient(app).post("/detect", json={"id": "vid-1"})

    thread = threading.Thread(target=first_call)
    thread.start()
    assert started.wait(timeout=2)

    second = client.post("/detect", json={"id": "vid-2"})
    assert second.status_code == 409

    release.set()
    thread.join(timeout=3)
    first = finished["response"]
    assert first.status_code == 200
    assert first.json()["status"] == "done"


def test_failed_job_posts_error_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[dict] = []

    def fake_urlopen(request: object, timeout: int = 30) -> _Response:
        posted.append(
            (
                request.full_url,  # type: ignore[attr-defined]
                json.loads(request.data.decode()),  # type: ignore[attr-defined]
            )
        )
        return _Response()

    def fail_fetch(video_id: str) -> ScorebugVideo:
        raise RuntimeError(f"missing {video_id}")

    monkeypatch.setattr("mac.api.fetch_scorebug_video", fail_fetch)
    monkeypatch.setattr("mac.api.urllib.request.urlopen", fake_urlopen)

    client = TestClient(app)
    response = client.post("/detect", json={"id": "vid-9"})
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "failed"
    assert body["error"] == "missing vid-9"
    assert posted == [
        ("http://localhost:3000/api/scorebug-videos/vid-9/score-timeline", body)
    ]


def test_choose_text_keeps_a_clock_instead_of_a_neighboring_score() -> None:
    from mac.vision_ocr import _choose_text, _to_scorebug, parse_score

    assert _choose_text("game_clock", [("12", 0.95), ("9:58", 0.4)]) == "9:58"
    assert parse_score("1O") == 10
    home = ScoreboardRegion(x=0.6, y=0.0, w=0.4, h=1.0)
    assert _to_scorebug((0.0, 0.0, 1.0, 1.0), home) == (0.6, 0.0, 0.4, 1.0)


def test_draw_debug_outlines_markings_and_vision_text() -> None:
    from mac.vision_ocr import draw_debug

    image = np.zeros((40, 200, 3), dtype=np.uint8)
    regions = {"home_score": ScoreboardRegion(x=0.1, y=0.25, w=0.2, h=0.5)}
    drawn = draw_debug(
        image,
        regions,
        [("4|5", 0.9, (0.5, 0.25, 0.2, 0.5))],
    )
    assert drawn.shape[0] == 160
    assert tuple(int(value) for value in drawn[80, 80]) == (0, 220, 0)
    assert tuple(int(value) for value in drawn[80, 400]) == (0, 140, 255)


def test_unknown_job_is_not_found() -> None:
    client = TestClient(app)
    response = client.get("/detect/missing")
    assert response.status_code == 404
