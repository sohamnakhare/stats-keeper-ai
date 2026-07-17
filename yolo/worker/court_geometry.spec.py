"""Unit tests for court polygon geometry."""

from __future__ import annotations

from dataclasses import dataclass

from court_geometry import ball_center_in_court, point_in_polygon


@dataclass
class _Ball:
    x: float
    y: float
    w: float = 0.02
    h: float = 0.02
    conf: float = 0.9


SQUARE = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]


def test_point_inside_square() -> None:
    assert point_in_polygon(0.5, 0.5, SQUARE)


def test_point_outside_square() -> None:
    assert not point_in_polygon(0.1, 0.5, SQUARE)


def test_ball_center_in_court_disabled_when_no_polygon() -> None:
    ball = _Ball(0.1, 0.1)
    assert ball_center_in_court(ball, None)
    assert ball_center_in_court(ball, [])


def test_ball_center_rejected_outside_court() -> None:
    ball = _Ball(0.1, 0.5)
    assert not ball_center_in_court(ball, SQUARE)
