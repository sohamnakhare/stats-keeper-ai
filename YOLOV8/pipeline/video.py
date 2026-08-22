"""OpenCV video I/O and frame sampling."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SampledFrame:
    index: int
    source_frame: int
    t_sec: float
    bgr: np.ndarray
    native_fps: float
    sample_fps: float


def open_capture(video_path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def video_native_fps(cap: cv2.VideoCapture) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if fps > 1e-3 else 30.0


def iter_sampled_frames(
    video_path: str | Path,
    sample_fps: float = 10.0,
    max_width: int = 1280,
) -> Iterator[SampledFrame]:
    """Yield frames at approximately `sample_fps`, optionally downscaled."""
    cap = open_capture(video_path)
    try:
        native_fps = video_native_fps(cap)
        frame_interval = max(1, int(round(native_fps / max(sample_fps, 0.1))))
        effective_fps = native_fps / frame_interval

        frame_idx = 0
        sample_idx = 0
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
                    source_frame=frame_idx,
                    t_sec=sample_idx / effective_fps,
                    bgr=frame,
                    native_fps=native_fps,
                    sample_fps=effective_fps,
                )
                sample_idx += 1
            frame_idx += 1
    finally:
        cap.release()


def make_video_writer(
    path: str | Path,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer
