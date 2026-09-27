"""Unit tests for video URL classification and download reuse."""

from __future__ import annotations

import hashlib

import pytest

from worker.download_video import (
    UnsupportedVideoUrl,
    classify_video_url,
    download_video,
    extension_for_content_type,
)


def test_classify_youtube_links() -> None:
    assert classify_video_url("https://www.youtube.com/watch?v=abc") == "youtube"
    assert classify_video_url("https://youtu.be/abc") == "youtube"
    assert classify_video_url("https://www.youtube.com/shorts/abc") == "youtube"
    assert classify_video_url("https://m.youtube.com/watch?v=abc") == "youtube"


def test_classify_direct_video_links() -> None:
    assert classify_video_url("https://cdn.example.com/game.mp4") == "direct"
    assert classify_video_url("https://cdn.example.com/game.webm?token=1") == "direct"
    assert classify_video_url("http://cdn.example.com/a/game.MKV") == "direct"
    assert classify_video_url("https://cdn.example.com/game.mov") == "direct"


def test_classify_probe_and_rejected_links() -> None:
    assert classify_video_url("https://cdn.example.com/stream") == "probe"
    with pytest.raises(UnsupportedVideoUrl):
        classify_video_url("ftp://files.example.com/game.mp4")
    with pytest.raises(UnsupportedVideoUrl):
        classify_video_url("not-a-url")


def test_extension_for_content_type() -> None:
    assert extension_for_content_type("video/mp4; charset=binary") == ".mp4"
    assert extension_for_content_type("video/webm") == ".webm"
    assert extension_for_content_type("video/ogg") == ".mp4"
    assert extension_for_content_type("text/html") is None
    assert extension_for_content_type(None) is None


def test_download_reuses_existing_file(tmp_path) -> None:
    url = "https://cdn.example.com/game.mp4"
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    existing = tmp_path / f"{digest}.mp4"
    existing.write_bytes(b"video")

    assert download_video(url, tmp_path) == existing
