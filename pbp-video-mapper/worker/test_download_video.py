"""Unit tests for video URL classification and download reuse."""

from __future__ import annotations

import hashlib

import pytest

from worker.download_video import (
    UnsupportedVideoUrl,
    aria2c_command,
    classify_video_url,
    download_video,
    extension_for_content_type,
    youtube_format,
    youtube_stream_command,
)
from worker.extract_frames import ffmpeg_url_command, iter_png_frames, scorebug_crop_filter
from worker.schemas import ScoreboardRegion


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


def test_youtube_format_is_video_only_and_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCOREBUG_MAX_HEIGHT", raising=False)
    capped = youtube_format()
    assert capped == (
        "bv*[height<=720][ext=mp4]/b[height<=720][ext=mp4]/"
        "bv*[height<=720]/b[height<=720]"
    )
    assert "bestaudio" not in capped
    monkeypatch.setenv("SCOREBUG_MAX_HEIGHT", "480")
    assert "height<=480" in youtube_format()
    assert "bestaudio" not in youtube_format()


def test_youtube_stream_fetches_fragments_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCOREBUG_CONCURRENT_FRAGMENTS", raising=False)
    monkeypatch.delenv("SCOREBUG_MAX_HEIGHT", raising=False)
    command = youtube_stream_command("https://youtu.be/abc")
    fragments = command.index("--concurrent-fragments")
    assert command[fragments + 1] == "16"
    assert "height<=720" in command[command.index("-f") + 1]
    assert command[command.index("-o") + 1] == "-"


def test_aria2c_uses_the_same_connection_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("SCOREBUG_CONCURRENT_FRAGMENTS", raising=False)
    dest = tmp_path / "video.mp4"
    command = aria2c_command("https://cdn.example.com/game.mp4", dest)
    assert command[command.index("-x") + 1] == "16"
    assert command[command.index("-s") + 1] == "16"


def test_ffmpeg_crop_uses_the_scorebug_box() -> None:
    region = ScoreboardRegion(x=0.1, y=0.8, w=0.4, h=0.15)
    filt = scorebug_crop_filter(1.0, region)
    assert filt.startswith("fps=1,")
    assert "crop=iw*0.4:ih*0.15:iw*0.1:ih*0.8" in filt
    command = ffmpeg_url_command("https://cdn.example.com/game.mp4", 1.0, region)
    assert command[0] == "ffmpeg"
    assert filt in command


def test_png_stream_yields_frames() -> None:
    import io

    import numpy as np

    first = np.zeros((8, 10, 3), dtype=np.uint8)
    second = np.full((6, 6, 3), 9, dtype=np.uint8)
    encoded = [_png(first), _png(second)]
    frames = list(iter_png_frames(io.BytesIO(b"".join(encoded))))
    assert [frame.shape[:2] for frame in frames] == [(8, 10), (6, 6)]


def _png(image) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_download_reuses_existing_file(tmp_path) -> None:
    url = "https://cdn.example.com/game.mp4"
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    existing = tmp_path / f"{digest}.mp4"
    existing.write_bytes(b"video")

    assert download_video(url, tmp_path) == existing
