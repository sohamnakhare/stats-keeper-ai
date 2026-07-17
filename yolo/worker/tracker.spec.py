"""Unit tests for ball Kalman tracker."""

from __future__ import annotations

from detector import BallDetection
from schemas import BallTrackerConfig, HoopRoi
from tracker import BallTracker

HOOP = HoopRoi(x=0.5, y=0.25, w=0.05, h=0.09)


def test_tracker_resets_after_default_predict_window() -> None:
    cfg = BallTrackerConfig(max_predict_frames=1, max_predict_frames_shot=20)
    tracker = BallTracker(sample_fps=20.0, config=cfg)
    det = BallDetection(x=0.3, y=0.7, w=0.02, h=0.02, conf=0.9)
    tracker.step(0, det, hoop=HOOP)
    tracker.step(1, det, hoop=HOOP)
    p2 = tracker.step(2, None, hoop=HOOP)
    p3 = tracker.step(3, None, hoop=HOOP)
    assert p2.predicted
    assert p3.x == 0.0 and p3.y == 0.0


def test_tracker_extends_predict_during_shot_flight() -> None:
    cfg = BallTrackerConfig(
        max_predict_frames=2,
        max_predict_frames_shot=10,
        min_upward_velocity=0.001,
    )
    tracker = BallTracker(sample_fps=20.0, config=cfg)
    tracker.step(0, BallDetection(0.5, 0.55, 0.02, 0.02, 0.9), hoop=HOOP)
    tracker.step(1, BallDetection(0.5, 0.48, 0.02, 0.02, 0.9), hoop=HOOP)
    last = tracker.points[-1]
    for i in range(2, 11):
        last = tracker.step(i, None, hoop=HOOP)
    assert last.predicted
    assert last.x != 0.0
