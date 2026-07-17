"""Kalman filter ball tracker with gap-fill."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter

from detector import BallDetection
from schemas import BallTrackPoint, BallTrackerConfig, HoopRoi

MAX_PREDICT_FRAMES = 8
MAX_PREDICT_FRAMES_SHOT = 20


@dataclass
class _TrackState:
    kalman: KalmanFilter
    frames_since_detection: int = 0
    initialized: bool = False


class BallTracker:
    """Track normalized ball center; predict through short occlusions."""

    def __init__(
        self,
        sample_fps: float,
        config: BallTrackerConfig | None = None,
    ) -> None:
        self.sample_fps = sample_fps
        self.config = config or BallTrackerConfig()
        self._state: _TrackState | None = None
        self._points: list[BallTrackPoint] = []

    def reset(self) -> None:
        self._state = None
        self._points = []

    def _new_kalman(self) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0 / max(self.sample_fps, 1.0)
        kf.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        kf.R *= 0.02
        kf.P *= 10.0
        kf.Q = np.eye(4) * 0.001
        return kf

    def _velocity_y(self) -> float:
        if self._state is None:
            return 0.0
        return float(self._state.kalman.x[3])

    def _in_shot_flight(self, hoop: HoopRoi | None) -> bool:
        """Extend Kalman predict when ball appears to be heading toward the basket."""
        if self._state is None or hoop is None:
            return False

        ball_x = float(self._state.kalman.x[0])
        ball_y = float(self._state.kalman.x[1])
        vy = self._velocity_y()

        cfg = self.config
        near_hoop_x = abs(ball_x - hoop.x) <= hoop.w * cfg.shot_hoop_x_band
        rising = vy <= -cfg.min_upward_velocity
        above_rim_zone = ball_y <= hoop.y + hoop.h

        return near_hoop_x and (rising or above_rim_zone)

    def _max_predict_frames(self, hoop: HoopRoi | None) -> int:
        if self._in_shot_flight(hoop):
            return self.config.max_predict_frames_shot
        return self.config.max_predict_frames

    def step(
        self,
        frame_index: int,
        detection: BallDetection | None,
        hoop: HoopRoi | None = None,
    ) -> BallTrackPoint:
        t_sec = frame_index / max(self.sample_fps, 1.0)

        if detection is not None:
            if self._state is None:
                kf = self._new_kalman()
                kf.x = np.array([detection.x, detection.y, 0.0, 0.0], dtype=float)
                self._state = _TrackState(kalman=kf, initialized=True, frames_since_detection=0)
            else:
                assert self._state is not None
                self._state.kalman.predict()
                self._state.kalman.update(np.array([detection.x, detection.y], dtype=float))
                self._state.frames_since_detection = 0

            point = BallTrackPoint(
                frame=frame_index,
                tSec=t_sec,
                x=detection.x,
                y=detection.y,
                w=detection.w,
                h=detection.h,
                detected=True,
                predicted=False,
                conf=detection.conf,
            )
            self._points.append(point)
            return point

        if self._state is None or not self._state.initialized:
            point = BallTrackPoint(
                frame=frame_index,
                tSec=t_sec,
                x=0.0,
                y=0.0,
                w=None,
                h=None,
                detected=False,
                predicted=False,
                conf=None,
            )
            self._points.append(point)
            return point

        self._state.frames_since_detection += 1
        max_predict = self._max_predict_frames(hoop)
        if self._state.frames_since_detection > max_predict:
            self._state = None
            point = BallTrackPoint(
                frame=frame_index,
                tSec=t_sec,
                x=0.0,
                y=0.0,
                w=None,
                h=None,
                detected=False,
                predicted=False,
                conf=None,
            )
            self._points.append(point)
            return point

        self._state.kalman.predict()
        px = float(self._state.kalman.x[0])
        py = float(self._state.kalman.x[1])
        point = BallTrackPoint(
            frame=frame_index,
            tSec=t_sec,
            x=px,
            y=py,
            w=None,
            h=None,
            detected=False,
            predicted=True,
            conf=None,
        )
        self._points.append(point)
        return point

    @property
    def points(self) -> list[BallTrackPoint]:
        return self._points

    def detection_rate(self) -> float:
        if not self._points:
            return 0.0
        detected = sum(1 for p in self._points if p.detected)
        return detected / len(self._points)
