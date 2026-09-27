"""Unit tests for scorebug API marking conversion."""

from __future__ import annotations

import io
import json
import urllib.request

import pytest

from worker.scorebug_api import fetch_scorebug_video, parse_scorebug_video

SAMPLE = {
    "video": {
        "id": "vid-1",
        "videoUrl": "https://www.youtube.com/watch?v=abc",
        "videoType": "youtube",
        "title": "Home vs Away",
        "markings": {
            "scorebug": {"x": 0.1, "y": 0.8, "w": 0.4, "h": 0.15},
            "fields": {
                "homeScore": {"x": 0.05, "y": 0.2, "w": 0.2, "h": 0.5},
                "awayScore": {"x": 0.7, "y": 0.2, "w": 0.2, "h": 0.5},
            },
        },
    }
}


def test_parse_keeps_crop_relative_fields() -> None:
    video = parse_scorebug_video(SAMPLE)

    assert video.video_url == "https://www.youtube.com/watch?v=abc"
    assert video.scorebug.x == pytest.approx(0.1)
    assert set(video.fields) == {"home_score", "away_score"}
    assert video.fields["home_score"].x == pytest.approx(0.05)
    assert video.fields["away_score"].x == pytest.approx(0.7)


def test_parse_rejects_field_outside_crop() -> None:
    payload = {
        "video": {
            "videoUrl": "https://www.youtube.com/watch?v=abc",
            "markings": {
                "scorebug": {"x": 0.1, "y": 0.8, "w": 0.4, "h": 0.15},
                "fields": {
                    "homeScore": {"x": 0.9, "y": 0.2, "w": 0.2, "h": 0.5},
                },
            },
        }
    }

    with pytest.raises(ValueError, match="homeScore"):
        parse_scorebug_video(payload)


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _fake_urlopen(requested: list[str]):
    def fake_urlopen(
        request: urllib.request.Request,
        timeout: int = 30,
    ) -> _Response:
        assert timeout == 30
        requested.append(request.full_url)
        return _Response(json.dumps(SAMPLE).encode())

    return fake_urlopen


def test_fetch_uses_scorebug_api_base_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []
    monkeypatch.setenv("SCOREBUG_API_BASE", "https://app.example.com/")
    monkeypatch.setattr(
        "worker.scorebug_api.urllib.request.urlopen",
        _fake_urlopen(requested),
    )

    video = fetch_scorebug_video("vid-1")

    assert requested == ["https://app.example.com/api/scorebug-videos/vid-1"]
    assert video.video_url == "https://www.youtube.com/watch?v=abc"


def test_fetch_defaults_to_localhost_when_base_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []
    monkeypatch.delenv("SCOREBUG_API_BASE", raising=False)
    monkeypatch.setattr(
        "worker.scorebug_api.urllib.request.urlopen",
        _fake_urlopen(requested),
    )

    fetch_scorebug_video("vid-1")

    assert requested == ["http://localhost:3000/api/scorebug-videos/vid-1"]
