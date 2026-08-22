"""Tests for pose-based (zoom-aware) court projection."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

WORKER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_ROOT))

from court_overlay import PRESETS, paint_quads  # noqa: E402
from court_projection import PoseCourtProjector, _build_pose_paint_samples, _mean_quad_drift  # noqa: E402
from player_config import HomographyConfig  # noqa: E402


def _detect_payload(t_a: float, t_b: float, scale_a: float, scale_b: float) -> dict:
    paint = paint_quads(PRESETS["fibaHalf"])[0]
    # Map court meters to fake normalized pixels: origin + scale * court_xy.
    # Larger scale = zoomed in.

    def kpts(scale: float) -> list[dict]:
        ox, oy = 0.2, 0.3
        out = []
        for x, y in paint:
            out.append({"x": ox + scale * x, "y": oy + scale * y, "conf": 0.99})
        return out

    return {
        "preset": "fibaHalf",
        "keypointOrder": [0, 1, 2, 3],
        "frames": [
            {"index": 0, "t_sec": t_a, "keypoints": kpts(scale_a)},
            {"index": 1, "t_sec": t_b, "keypoints": kpts(scale_b)},
        ],
    }


def test_pose_projector_zoom_changes_meters() -> None:
    payload = _detect_payload(0.0, 2.0, scale_a=0.02, scale_b=0.04)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "court_detect.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        projector = PoseCourtProjector.from_detect_file(
            path, HomographyConfig(clamp_to_court=False)
        )

        paint = paint_quads(PRESETS["fibaHalf"])[0]
        cx, cy = float(np.mean(paint[:, 0])), float(np.mean(paint[:, 1]))
        px0 = 0.2 + 0.02 * cx
        py0 = 0.3 + 0.02 * cy
        a = projector.project(px0, py0, t_sec=0.0)
        assert a is not None
        assert abs(a[0] - cx) < 0.15
        assert abs(a[1] - cy) < 0.15

        # Same image pixel after zoom-in is a different court location.
        b = projector.project(px0, py0, t_sec=2.0)
        assert b is not None
        assert abs(b[0] - a[0]) + abs(b[1] - a[1]) > 0.5


def test_pose_projector_holds_last_good() -> None:
    payload = _detect_payload(0.0, 1.0, 0.02, 0.02)
    payload["frames"].insert(
        1,
        {
            "index": 1,
            "t_sec": 0.5,
            "keypoints": [
                {"x": 0.1, "y": 0.1, "conf": 0.01},
                {"x": 0.2, "y": 0.1, "conf": 0.01},
                {"x": 0.2, "y": 0.2, "conf": 0.01},
                {"x": 0.1, "y": 0.2, "conf": 0.01},
            ],
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "court_detect.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        projector = PoseCourtProjector.from_detect_file(path)
        assert projector._index_for_t(0.5) >= 0
        paint = paint_quads(PRESETS["fibaHalf"])[0]
        cx, cy = float(paint[0][0]), float(paint[0][1])
        px = 0.2 + 0.02 * cx
        py = 0.3 + 0.02 * cy
        a = projector.project(px, py, t_sec=0.0)
        b = projector.project(px, py, t_sec=0.5)
        assert a is not None and b is not None
        assert abs(a[0] - b[0]) < 0.05
        assert abs(a[1] - b[1]) < 0.05


def test_pose_projector_uses_paint_index_1() -> None:
    paint = paint_quads(PRESETS["fiba"])[1]
    ox, oy, scale = 0.15, 0.2, 0.02

    def kpts() -> list[dict]:
        return [{"x": ox + scale * x, "y": oy + scale * y, "conf": 0.99} for x, y in paint]

    payload = {
        "preset": "fiba",
        "keypointOrder": [0, 1, 2, 3],
        "paintIndex": 1,
        "frames": [{"index": 0, "t_sec": 0.0, "keypoints": kpts()}],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "court_detect.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        projector = PoseCourtProjector.from_detect_file(
            path, HomographyConfig(clamp_to_court=False)
        )
        assert projector.paint_index == 1
        cx, cy = float(np.mean(paint[:, 0])), float(np.mean(paint[:, 1]))
        px, py = ox + scale * cx, oy + scale * cy
        got = projector.project(px, py, t_sec=0.0)
        assert got is not None
        assert abs(got[0] - cx) < 0.2
        assert abs(got[1] - cy) < 0.2
        left = paint_quads(PRESETS["fiba"])[0]
        left_cx = float(np.mean(left[:, 0]))
        assert abs(got[0] - left_cx) > 5.0


def test_pose_projector_skips_low_score_sample() -> None:
    payload = _detect_payload(0.0, 1.0, 0.02, 0.02)
    payload["frames"].insert(
        1,
        {
            "index": 1,
            "t_sec": 0.5,
            "keypoints": [
                {"x": 0.5, "y": 0.5, "conf": 0.99},
                {"x": 0.5, "y": 0.5, "conf": 0.99},
                {"x": 0.5, "y": 0.5, "conf": 0.99},
                {"x": 0.5, "y": 0.5, "conf": 0.99},
            ],
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "court_detect.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        projector = PoseCourtProjector.from_detect_file(
            path, HomographyConfig(clamp_to_court=False)
        )
        paint = paint_quads(PRESETS["fibaHalf"])[0]
        cx, cy = float(paint[0][0]), float(paint[0][1])
        px = 0.2 + 0.02 * cx
        py = 0.3 + 0.02 * cy
        a = projector.project(px, py, t_sec=0.0)
        b = projector.project(px, py, t_sec=0.5)
        assert a is not None and b is not None
        assert abs(a[0] - b[0]) < 0.05
        assert abs(a[1] - b[1]) < 0.05


def test_pose_projector_ignores_flow_and_holds_last() -> None:
    payload = _detect_payload(0.0, 1.0, 0.02, 0.04)
    payload["frames"][1]["source"] = "flow"
    for kp in payload["frames"][1]["keypoints"]:
        kp["conf"] = 0.15
        kp["source"] = "flow"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "court_detect.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        projector = PoseCourtProjector.from_detect_file(
            path, HomographyConfig(clamp_to_court=False)
        )
        paint = paint_quads(PRESETS["fibaHalf"])[0]
        cx, cy = float(np.mean(paint[:, 0])), float(np.mean(paint[:, 1]))
        px0 = 0.2 + 0.02 * cx
        py0 = 0.3 + 0.02 * cy
        a = projector.project(px0, py0, t_sec=0.0)
        b = projector.project(px0, py0, t_sec=1.0)
        assert a is not None and b is not None
        assert abs(a[0] - b[0]) < 0.05
        assert abs(a[1] - b[1]) < 0.05


def test_pose_projector_holds_through_keypoint_jitter() -> None:
    paint = paint_quads(PRESETS["fibaHalf"])[0]
    ox, oy, scale = 0.2, 0.3, 0.02

    def kpts(nudge: float) -> list[dict]:
        out = []
        for i, (x, y) in enumerate(paint):
            dx = nudge if i % 2 == 0 else -nudge
            out.append({"x": ox + scale * x + dx, "y": oy + scale * y, "conf": 0.99})
        return out

    payload = {
        "preset": "fibaHalf",
        "keypointOrder": [0, 1, 2, 3],
        "frames": [
            {"index": i, "t_sec": i * 0.1, "keypoints": kpts(0.0015 if i % 2 else 0.0)}
            for i in range(12)
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "court_detect.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        projector = PoseCourtProjector.from_detect_file(
            path, HomographyConfig(clamp_to_court=False)
        )
        cx, cy = float(np.mean(paint[:, 0])), float(np.mean(paint[:, 1]))
        px, py = ox + scale * cx, oy + scale * cy
        a = projector.project(px, py, t_sec=0.0)
        b = projector.project(px, py, t_sec=1.1)
        assert a is not None and b is not None
        assert abs(a[0] - b[0]) < 0.12
        assert abs(a[1] - b[1]) < 0.12


def test_paint_hysteresis_ignores_single_spike() -> None:
    paint = paint_quads(PRESETS["fibaHalf"])[0]
    preset = PRESETS["fibaHalf"]
    ox, oy, scale = 0.2, 0.3, 0.02
    order = (0, 1, 2, 3)

    def kpts(nudge: float) -> list[dict]:
        return [
            {"x": ox + scale * x + nudge, "y": oy + scale * y, "conf": 0.99}
            for x, y in paint
        ]

    cfg = HomographyConfig(
        clamp_to_court=False,
        paint_ema_alpha=1.0,
        paint_adopt_thresh=0.012,
        paint_adopt_hysteresis=3,
        paint_ramp_frames=4,
        paint_fast_adopt_mult=10.0,
    )
    frames = [
        {"t_sec": 0.0, "keypoints": kpts(0.0)},
        {"t_sec": 0.1, "keypoints": kpts(0.02)},
        {"t_sec": 0.2, "keypoints": kpts(0.0)},
    ]
    samples = _build_pose_paint_samples(frames, order, preset, paint, cfg, 0.5)
    assert len(samples) == 3
    assert np.allclose(samples[0][1], samples[1][1])
    assert np.allclose(samples[0][1], samples[2][1])


def test_paint_ramp_blends_over_multiple_frames() -> None:
    paint = paint_quads(PRESETS["fibaHalf"])[0]
    preset = PRESETS["fibaHalf"]
    ox, oy, scale = 0.2, 0.3, 0.02
    order = (0, 1, 2, 3)

    def kpts(nudge: float) -> list[dict]:
        return [
            {"x": ox + scale * x + nudge, "y": oy + scale * y, "conf": 0.99}
            for x, y in paint
        ]

    cfg = HomographyConfig(
        clamp_to_court=False,
        paint_ema_alpha=1.0,
        paint_adopt_thresh=0.001,
        paint_adopt_hysteresis=1,
        paint_ramp_frames=4,
        paint_fast_adopt_mult=10.0,
    )
    frames = [{"t_sec": i * 0.1, "keypoints": kpts(0.0 if i == 0 else 0.05)} for i in range(6)]
    samples = _build_pose_paint_samples(frames, order, preset, paint, cfg, 0.5)
    quads = [s[1] for s in samples]
    unique = {tuple(q.reshape(-1)) for q in quads}
    assert len(unique) >= 3
    assert not np.allclose(quads[0], quads[-1])
    for i in range(1, len(quads)):
        step = _mean_quad_drift(quads[i - 1], quads[i])
        assert step < 0.03


def test_in_court_does_not_snap_to_edge() -> None:
    from court_projection import CourtDimensions, _in_court_or_none

    court = CourtDimensions(length=15.0, width=14.0)
    strict = HomographyConfig(clamp_to_court=True, court_margin=0.3)
    loose = HomographyConfig(clamp_to_court=False, court_margin=0.3)
    assert _in_court_or_none(-0.1, 5.0, court, strict) is None
    assert _in_court_or_none(-0.1, 5.0, court, loose) == (-0.1, 5.0)
    assert _in_court_or_none(-0.5, 5.0, court, loose) is None
    assert _in_court_or_none(7.0, 7.0, court, strict) == (7.0, 7.0)
