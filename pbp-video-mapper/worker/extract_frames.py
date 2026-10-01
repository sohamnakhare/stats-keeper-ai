"""Sample video frames at a target FPS."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from .download_video import (
    classify_video_url,
    download_for_fallback,
    youtube_format,
    youtube_stream_command,
    ytdlp_argv,
)
from .schemas import ScoreboardRegion

logger = logging.getLogger(__name__)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class SampledFrame:
    index: int
    t_sec: float
    bgr: object  # numpy ndarray; typed loosely to avoid import in stubs


def iter_sampled_frames(
    video_path: str,
    sample_fps: float = 10.0,
    max_width: int = 1280,
) -> Iterator[SampledFrame]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if native_fps <= 0:
        native_fps = 30.0

    frame_interval = max(1, int(round(native_fps / max(sample_fps, 0.1))))
    sample_fps_effective = native_fps / frame_interval

    frame_idx = 0
    sample_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_interval == 0:
                height, width = frame.shape[:2]
                if width > max_width:
                    scale = max_width / width
                    frame = cv2.resize(
                        frame,
                        (max_width, int(height * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                yield SampledFrame(
                    index=sample_idx,
                    t_sec=sample_idx / sample_fps_effective,
                    bgr=frame,
                )
                sample_idx += 1

            frame_idx += 1
    finally:
        cap.release()


class StreamStartError(RuntimeError):
    """ffmpeg or yt-dlp exited before producing a frame."""


def crop_normalized(image: np.ndarray, region: ScoreboardRegion) -> np.ndarray:
    """Crop a normalized top-left box out of a BGR image."""
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(region.x * width)))
    y1 = max(0, min(height, int(region.y * height)))
    x2 = max(0, min(width, int((region.x + region.w) * width)))
    y2 = max(0, min(height, int((region.y + region.h) * height)))
    return image[y1:y2, x1:x2]


def scorebug_crop_filter(sample_fps: float, region: ScoreboardRegion) -> str:
    """ffmpeg filter that emits ``sample_fps`` scorebug crops."""
    fps = max(sample_fps, 0.1)
    return (
        f"fps={fps:g},"
        f"crop=iw*{region.w}:ih*{region.h}:iw*{region.x}:ih*{region.y}"
    )


def ffmpeg_pipe_command(sample_fps: float, region: ScoreboardRegion) -> list[str]:
    """Decode a video on stdin into PNG scorebug crops on stdout."""
    return _ffmpeg_command(["-i", "pipe:0"], sample_fps, region)


def ffmpeg_url_command(
    url: str,
    sample_fps: float,
    region: ScoreboardRegion,
) -> list[str]:
    """Decode a remote video into PNG scorebug crops on stdout."""
    return _ffmpeg_command(["-i", url], sample_fps, region)


def _ffmpeg_command(
    input_args: list[str],
    sample_fps: float,
    region: ScoreboardRegion,
) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        *input_args,
        "-an",
        "-vf",
        scorebug_crop_filter(sample_fps, region),
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]


def iter_png_bytes(stream: object) -> Iterator[bytes]:
    """Yield each PNG document from a concatenated stream."""
    pending = b""
    read = getattr(stream, "read")
    while True:
        block = read(1 << 16)
        if not block:
            break
        pending += block
        while True:
            start = pending.find(_PNG_SIGNATURE)
            if start < 0:
                pending = pending[-7:]
                break
            if start:
                pending = pending[start:]
            end = pending.find(b"IEND", 8)
            if end < 0 or len(pending) < end + 8:
                break
            yield pending[: end + 8]
            pending = pending[end + 8 :]


def iter_png_frames(stream: object) -> Iterator[np.ndarray]:
    """Decode each PNG on ``stream`` to a BGR image."""
    for payload in iter_png_bytes(stream):
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            yield image


def remote_duration_seconds(url: str) -> float | None:
    """Return the media duration in seconds, or None when it cannot be read."""
    kind = classify_video_url(url)
    if kind == "youtube":
        command = [
            *ytdlp_argv(),
            "--no-download",
            "--no-playlist",
            "-f",
            youtube_format(),
            "--print",
            "duration",
            url,
        ]
    else:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            url,
        ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    line = result.stdout.strip().splitlines()
    if not line:
        return None
    try:
        duration = float(line[-1])
    except ValueError:
        return None
    return duration if duration > 0 else None


def iter_remote_frames(
    url: str,
    sample_fps: float,
    scorebug: ScoreboardRegion,
) -> Iterator[SampledFrame]:
    """Sample scorebug crops while the video is still downloading.

    Each yielded frame is already the scorebug crop. When the stream cannot
    start, the video is downloaded and then sampled locally.
    """
    stream = _iter_piped_frames(url, sample_fps, scorebug)
    try:
        first = next(stream)
    except StopIteration:
        yield from _iter_fallback_frames(url, sample_fps, scorebug)
        return
    except StreamStartError as exc:
        logger.warning("stream did not start (%s); downloading %s", exc, url)
        yield from _iter_fallback_frames(url, sample_fps, scorebug)
        return
    yield first
    yield from stream


def _iter_fallback_frames(
    url: str,
    sample_fps: float,
    scorebug: ScoreboardRegion,
) -> Iterator[SampledFrame]:
    path = download_for_fallback(url)
    try:
        for frame in iter_sampled_frames(str(path), sample_fps=sample_fps):
            yield SampledFrame(
                index=frame.index,
                t_sec=frame.t_sec,
                bgr=crop_normalized(frame.bgr, scorebug),
            )
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)


def _iter_piped_frames(
    url: str,
    sample_fps: float,
    scorebug: ScoreboardRegion,
) -> Iterator[SampledFrame]:
    kind = classify_video_url(url)
    ytdlp: subprocess.Popen[bytes] | None = None
    ffmpeg: subprocess.Popen[bytes] | None = None
    ytdlp_err: list[bytes] = []
    ffmpeg_err: list[bytes] = []
    drainers: list[threading.Thread] = []
    produced = 0
    fps = max(sample_fps, 0.1)
    try:
        if kind == "youtube":
            ytdlp = subprocess.Popen(
                youtube_stream_command(url),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert ytdlp.stderr is not None and ytdlp.stdout is not None
            ytdlp_drain = threading.Thread(
                target=_drain,
                args=(ytdlp.stderr, ytdlp_err),
                daemon=True,
            )
            ytdlp_drain.start()
            drainers.append(ytdlp_drain)
            ffmpeg = subprocess.Popen(
                ffmpeg_pipe_command(fps, scorebug),
                stdin=ytdlp.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            ytdlp.stdout.close()
        else:
            ffmpeg = subprocess.Popen(
                ffmpeg_url_command(url, fps, scorebug),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        assert ffmpeg.stderr is not None and ffmpeg.stdout is not None
        ffmpeg_drain = threading.Thread(
            target=_drain,
            args=(ffmpeg.stderr, ffmpeg_err),
            daemon=True,
        )
        ffmpeg_drain.start()
        drainers.append(ffmpeg_drain)
        for image in iter_png_frames(ffmpeg.stdout):
            yield SampledFrame(index=produced, t_sec=produced / fps, bgr=image)
            produced += 1
    except FileNotFoundError as exc:
        raise StreamStartError(str(exc)) from exc
    finally:
        _stop(ffmpeg)
        _stop(ytdlp)
        for drainer in drainers:
            drainer.join(timeout=1)
    if produced == 0:
        detail = _stderr_text(ytdlp_err, ffmpeg_err) or "ffmpeg produced no frames"
        raise StreamStartError(detail)
    if ffmpeg is not None and ffmpeg.returncode not in (0, None):
        detail = _stderr_text(ytdlp_err, ffmpeg_err)
        raise RuntimeError(detail or f"ffmpeg exited {ffmpeg.returncode}")


def _drain(pipe: object, chunks: list[bytes]) -> None:
    read = getattr(pipe, "read")
    try:
        while True:
            block = read(4096)
            if not block:
                break
            chunks.append(block)
    except Exception:
        return


def _stop(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.kill()
    process.wait()


def _stderr_text(ytdlp_err: list[bytes], ffmpeg_err: list[bytes]) -> str:
    text = b"".join(ytdlp_err + ffmpeg_err).decode("utf-8", errors="replace").strip()
    return text[-500:]
