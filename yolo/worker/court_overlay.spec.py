"""Tests for court overlay geometry and keypoint-order parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WORKER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_ROOT))

from court_overlay import (  # noqa: E402
    DEFAULT_KEYPOINT_ORDER,
    PRESETS,
    clip_segment_to_rect,
    fit_overlay,
    keypoints_confident,
    paint_quads,
    parse_keypoint_order,
    snap_point_to_gradient,
    warp_keypoints_lk,
)


def test_default_keypoint_order_is_bl_br_tr_tl() -> None:
    assert DEFAULT_KEYPOINT_ORDER == (0, 1, 2, 3)
    assert parse_keypoint_order("0,1,2,3") == DEFAULT_KEYPOINT_ORDER


def test_parse_keypoint_order_accepts_permutation() -> None:
    assert parse_keypoint_order("3,0,1,2") == (3, 0, 1, 2)


def test_parse_keypoint_order_auto() -> None:
    assert parse_keypoint_order("auto") is None
    assert parse_keypoint_order(None) is None
    assert parse_keypoint_order("") is None


def test_parse_keypoint_order_rejects_duplicates() -> None:
    try:
        parse_keypoint_order("0,0,1,2")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_fiba_half_paint_quad() -> None:
    quad = paint_quads(PRESETS["fibaHalf"])[0]
    assert quad.shape == (4, 2)
    width = float(np.linalg.norm(quad[1] - quad[0]))
    depth = float(np.linalg.norm(quad[3] - quad[0]))
    assert abs(width - 4.9) < 1e-6
    assert abs(depth - 5.8) < 1e-6


def test_fit_overlay_recovers_order() -> None:
    import cv2

    from court_overlay import court_to_pixels

    preset = PRESETS["fibaHalf"]
    outline = np.array([[0.0, 0.0], [15.0, 0.0], [15.0, 14.0], [0.0, 14.0]], dtype=np.float64)
    image = np.array([[200.0, 60.0], [440.0, 70.0], [620.0, 340.0], [30.0, 330.0]], dtype=np.float64)
    H, _ = cv2.findHomography(outline, image, 0)
    assert H is not None
    pixels = court_to_pixels(H, paint_quads(preset)[0])
    shuffled = [tuple(pixels[2]), tuple(pixels[0]), tuple(pixels[3]), tuple(pixels[1])]
    fit = fit_overlay(shuffled, preset, frame_w=640, frame_h=360)
    assert fit is not None
    ordered = np.array([shuffled[i] for i in fit.order])
    projected = court_to_pixels(fit.homography, paint_quads(preset)[0])
    assert np.allclose(projected, ordered, atol=1.5)
    assert fit.score > 0


def test_clip_segment_keeps_in_frame_part() -> None:
    clipped = clip_segment_to_rect((-20.0, 50.0), (100.0, 50.0), width=80, height=60)
    assert clipped is not None
    (x0, y0), (x1, y1) = clipped
    assert abs(x0 - 0.0) < 1e-6
    assert abs(x1 - 79.0) < 1e-6
    assert abs(y0 - 50.0) < 1e-6


def test_clip_segment_rejects_outside() -> None:
    assert clip_segment_to_rect((-10.0, -10.0), (-5.0, -5.0), 80, 60) is None


def test_keypoints_confident() -> None:
    assert keypoints_confident([0.9, 0.8, 0.7, 0.6])
    assert not keypoints_confident([0.9, 0.01, 0.7, 0.6])


def test_snap_point_moves_to_bright_edge() -> None:
    mag = np.zeros((40, 40), dtype=np.float32)
    mag[20, 28] = 1.0
    x, y = snap_point_to_gradient(mag, 20.0, 20.0, radius=12)
    assert (int(x), int(y)) == (28, 20)


def test_warp_keypoints_lk_tracks_translation() -> None:
    import cv2

    pts = [(40.0, 30.0), (120.0, 30.0), (120.0, 90.0), (40.0, 90.0)]
    prev = np.zeros((120, 160), dtype=np.uint8)
    for x, y in pts:
        cv2.circle(prev, (int(x), int(y)), 7, 255, -1)
    dx, dy = 6.0, -4.0
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    gray = cv2.warpAffine(prev, matrix, (160, 120), borderMode=cv2.BORDER_CONSTANT)
    warped = warp_keypoints_lk(prev, gray, pts)
    assert warped is not None
    expected = np.array(pts) + np.array([dx, dy])
    assert np.allclose(warped, expected, atol=2.0)


def test_warp_keypoints_lk_rejects_lost_points() -> None:
    prev = np.full((80, 80), 128, dtype=np.uint8)
    gray = np.full((80, 80), 128, dtype=np.uint8)
    outside = [(-10.0, -10.0), (200.0, -10.0), (200.0, 200.0), (-10.0, 200.0)]
    assert warp_keypoints_lk(prev, gray, outside) is None
