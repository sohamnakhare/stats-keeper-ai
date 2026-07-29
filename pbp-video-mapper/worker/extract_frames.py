"""Sample video frames at a target FPS."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2


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
