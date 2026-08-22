"""Pixel -> court-meter homography from court-marker.html calibration."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from player_config import HomographyConfig


def _mean_quad_drift(a: np.ndarray, b: np.ndarray) -> float:
    """Mean vertex distance between two (4, 2) quads."""
    da = np.asarray(a, dtype=np.float64).reshape(4, 2)
    db = np.asarray(b, dtype=np.float64).reshape(4, 2)
    delta = da - db
    return float(np.mean(np.hypot(delta[:, 0], delta[:, 1])))


def _lerp_quad(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear blend between two (4, 2) paint quads."""
    return (1.0 - t) * np.asarray(a, dtype=np.float64) + t * np.asarray(b, dtype=np.float64)


def _build_pose_paint_samples(
    frames: list[dict],
    order: tuple[int, ...],
    preset: dict[str, float | str],
    paint_pts: np.ndarray,
    config: HomographyConfig,
    min_conf: float,
) -> list[tuple[float, np.ndarray]]:
    """Smooth paint quads with hysteresis + ramped homography adoption."""
    from court_overlay import _score_homography, keypoints_confident

    samples: list[tuple[float, np.ndarray]] = []
    last_pixels: np.ndarray | None = None
    ema_pixels: np.ndarray | None = None
    alpha = float(config.paint_ema_alpha)
    adopt = float(config.paint_adopt_thresh)
    hysteresis = max(1, int(config.paint_adopt_hysteresis))
    ramp_frames = max(1, int(config.paint_ramp_frames))
    fast_mult = max(1.0, float(config.paint_fast_adopt_mult))

    pending_high = 0
    ramp_start: np.ndarray | None = None
    ramp_target: np.ndarray | None = None
    ramp_step = 0
    ramp_total = 0

    def append_sample(t_sec: float) -> None:
        if last_pixels is not None:
            samples.append((t_sec, last_pixels.copy()))

    def advance_ramp() -> None:
        nonlocal last_pixels, ramp_step, ramp_start, ramp_target, ramp_total
        if ramp_start is None or ramp_target is None or ramp_total <= 0:
            return
        ramp_step += 1
        frac = min(1.0, ramp_step / ramp_total)
        last_pixels = _lerp_quad(ramp_start, ramp_target, frac)
        if ramp_step >= ramp_total:
            ramp_start = None
            ramp_target = None
            ramp_step = 0
            ramp_total = 0

    def start_ramp(target: np.ndarray, *, frames: int) -> None:
        nonlocal last_pixels, ramp_start, ramp_target, ramp_step, ramp_total, pending_high
        if last_pixels is None:
            last_pixels = target.copy()
            pending_high = 0
            return
        ramp_start = last_pixels.copy()
        ramp_target = target.copy()
        ramp_step = 0
        ramp_total = max(1, frames)
        pending_high = 0

    for frame in frames:
        t_sec = float(frame.get("t_sec", 0.0))
        kpts = frame.get("keypoints") or []
        from_flow = str(frame.get("source") or "") == "flow" or any(
            str(k.get("source") or "") == "flow" for k in kpts
        )
        usable = False
        pixels: np.ndarray | None = None
        if not from_flow and len(kpts) == 4:
            confs = [float(k.get("conf", 0.0)) for k in kpts]
            if keypoints_confident(confs, min_conf):
                pixels = np.array(
                    [[float(kpts[i]["x"]), float(kpts[i]["y"])] for i in order],
                    dtype=np.float64,
                )
                usable = _normalized_homography_ok(
                    pixels, paint_pts, preset, _score_homography
                )

        if ramp_total > 0:
            advance_ramp()
            append_sample(t_sec)
            if usable and pixels is not None:
                if ema_pixels is None:
                    ema_pixels = pixels.copy()
                else:
                    ema_pixels = (1.0 - alpha) * ema_pixels + alpha * pixels
            continue

        if usable and pixels is not None:
            if ema_pixels is None or last_pixels is None:
                ema_pixels = pixels.copy()
                last_pixels = pixels.copy()
            else:
                ema_pixels = (1.0 - alpha) * ema_pixels + alpha * pixels
                drift = _mean_quad_drift(ema_pixels, last_pixels)
                if drift >= adopt * fast_mult:
                    start_ramp(ema_pixels, frames=min(2, ramp_frames))
                    advance_ramp()
                elif drift >= adopt:
                    pending_high += 1
                    if pending_high >= hysteresis:
                        start_ramp(ema_pixels, frames=ramp_frames)
                        advance_ramp()
                else:
                    pending_high = 0

        append_sample(t_sec)

    return samples


def _in_court_or_none(
    x: float,
    y: float,
    court: CourtDimensions,
    config: HomographyConfig,
) -> tuple[float, float] | None:
    """Drop far-outside points. Never snap onto the baseline (that pulses Y)."""
    margin = config.court_margin
    if x < -margin or x > court.length + margin:
        return None
    if y < -margin or y > court.width + margin:
        return None
    if config.clamp_to_court:
        if x < 0.0 or x > court.length or y < 0.0 or y > court.width:
            return None
    return x, y


def _normalized_homography_ok(
    pixels: np.ndarray,
    paint_pts: np.ndarray,
    preset: dict[str, float | str],
    score_fn,
) -> bool:
    """True when pixel->court H inverts to a plausible court->unit-square overlay."""
    matrix, _ = cv2.findHomography(pixels, paint_pts, method=0)
    if matrix is None:
        return False
    try:
        overlay_h = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return False
    return float(score_fn(overlay_h, preset, paint_pts, 1, 1)) >= 0.0


@dataclass(frozen=True)
class CourtDimensions:
    length: float  # meters, along x
    width: float  # meters, along y
    unit: str = "m"
    preset: str | None = None
    layout: str = "full"  # "full" | "half"


class CourtProjector:
    """Projects normalized pixel coordinates to court meters.

    Calibration pixels are normalized (0-1), so the homography maps
    normalized-pixel space directly to court space; this is consistent
    regardless of the resolution frames were sampled at.
    """

    def __init__(
        self,
        court: CourtDimensions,
        homography: np.ndarray,
        config: HomographyConfig,
    ) -> None:
        self.court = court
        self.matrix = homography
        self.config = config

    @classmethod
    def from_calibration_file(
        cls, path: Path | str, config: HomographyConfig | None = None
    ) -> CourtProjector:
        config = config or HomographyConfig()
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        points = data.get("points") or []
        if len(points) < 4:
            raise ValueError(
                f"Calibration file {path} has {len(points)} points; at least 4 are required."
            )

        court_info = data.get("court") or {}
        length = float(court_info.get("length", 28.0))
        layout = court_info.get("layout")
        if layout not in ("full", "half"):
            # Legacy calibrations without layout: short courts were half/3x3.
            layout = "half" if length < 20 else "full"
        court = CourtDimensions(
            length=length,
            width=float(court_info.get("width", 15.0)),
            unit=str(court_info.get("unit", "m")),
            preset=court_info.get("preset"),
            layout=layout,
        )

        pixel_pts = np.array(
            [[float(p["pixel"]["x"]), float(p["pixel"]["y"])] for p in points],
            dtype=np.float64,
        )
        court_pts = np.array(
            [[float(p["court"]["x"]), float(p["court"]["y"])] for p in points],
            dtype=np.float64,
        )

        method = cv2.RANSAC if len(points) > 4 else 0
        matrix, _ = cv2.findHomography(
            pixel_pts, court_pts, method, config.ransac_reproj_threshold
        )
        if matrix is None:
            raise ValueError(f"Could not compute homography from {path}.")

        return cls(court=court, homography=matrix, config=config)

    def project(self, px: float, py: float, t_sec: float | None = None) -> tuple[float, float] | None:
        """Project a normalized pixel point to court meters.

        Returns None when the point lands further than `court_margin` outside
        the court (typically a bench player, spectator, or bad detection).
        `t_sec` is ignored for a static calibration.
        """
        src = np.array([[[px, py]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.matrix)
        x, y = float(dst[0, 0, 0]), float(dst[0, 0, 1])
        return _in_court_or_none(x, y, self.court, self.config)


class PoseCourtProjector:
    """Per-frame pixel -> court meters from court_detect.json D keypoints.

    Zoom changes the image size of the paint; using that frame's keypoints
    (holding the last confident set) keeps player meters aligned with the overlay.
    """

    def __init__(
        self,
        court: CourtDimensions,
        paint_pts: np.ndarray,
        samples: list[tuple[float, np.ndarray]],
        config: HomographyConfig,
        paint_index: int = 0,
    ) -> None:
        self.court = court
        self._paint_pts = paint_pts
        self._times = [s[0] for s in samples]
        self._pixels = [s[1] for s in samples]
        self.config = config
        self.paint_index = paint_index
        self._h_cache: dict[int, np.ndarray] = {}

    @classmethod
    def from_detect_file(
        cls,
        path: Path | str,
        config: HomographyConfig | None = None,
        min_keypoint_conf: float | None = None,
    ) -> PoseCourtProjector:
        from court_overlay import (
            DEFAULT_KEYPOINT_ORDER,
            MIN_KEYPOINT_CONF,
            PRESETS,
            paint_quads,
        )

        config = config or HomographyConfig()
        min_conf = MIN_KEYPOINT_CONF if min_keypoint_conf is None else min_keypoint_conf
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        preset_name = str(data.get("preset") or "fibaHalf")
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown court detect preset {preset_name!r} in {path}")
        preset = PRESETS[preset_name]
        order_raw = data.get("keypointOrder") or list(DEFAULT_KEYPOINT_ORDER)
        order = tuple(int(i) for i in order_raw)
        if sorted(order) != [0, 1, 2, 3]:
            order = DEFAULT_KEYPOINT_ORDER

        quads = paint_quads(preset)
        paint_index = int(data.get("paintIndex") or 0)
        if paint_index < 0 or paint_index >= len(quads):
            paint_index = 0
        paint_pts = quads[paint_index]

        samples = _build_pose_paint_samples(
            data.get("frames") or [],
            order,
            preset,
            paint_pts,
            config,
            min_conf,
        )

        if not samples:
            raise ValueError(f"No confident court keypoints in {path}")

        court = CourtDimensions(
            length=float(preset["length"]),
            width=float(preset["width"]),
            unit="m",
            preset=preset_name,
            layout=str(preset["layout"]),
        )
        return cls(
            court=court,
            paint_pts=paint_pts,
            samples=samples,
            config=config,
            paint_index=paint_index,
        )

    def _index_for_t(self, t_sec: float) -> int:
        idx = bisect.bisect_right(self._times, t_sec) - 1
        return max(0, idx)

    def _matrix_at(self, t_sec: float) -> np.ndarray:
        idx = self._index_for_t(t_sec)
        cached = self._h_cache.get(idx)
        if cached is not None:
            return cached
        matrix, _ = cv2.findHomography(self._pixels[idx], self._paint_pts, method=0)
        if matrix is None:
            raise ValueError(f"Could not compute pose homography at t={t_sec:.3f}s")
        self._h_cache[idx] = matrix
        return matrix

    def project(self, px: float, py: float, t_sec: float | None = None) -> tuple[float, float] | None:
        t = 0.0 if t_sec is None else t_sec
        matrix = self._matrix_at(t)
        src = np.array([[[px, py]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, matrix)
        x, y = float(dst[0, 0, 0]), float(dst[0, 0, 1])
        return _in_court_or_none(x, y, self.court, self.config)
