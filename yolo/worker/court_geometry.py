"""Court boundary helpers (normalized image coordinates)."""

from __future__ import annotations

from typing import Protocol


class _BallCenter(Protocol):
    x: float
    y: float


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test. polygon vertices in order (closed implicitly)."""
    n = len(polygon)
    if n < 3:
        return True

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def ball_center_in_court(
    ball: _BallCenter,
    polygon: list[tuple[float, float]] | None,
) -> bool:
    if not polygon or len(polygon) < 3:
        return True
    return point_in_polygon(ball.x, ball.y, polygon)


def normalize_court_polygon(
    points: list[tuple[float, float] | dict[str, float]],
) -> list[tuple[float, float]]:
    """Parse courtPolygon from JSON into (x, y) tuples."""
    normalized: list[tuple[float, float]] = []
    for point in points:
        if isinstance(point, dict):
            normalized.append((float(point["x"]), float(point["y"])))
        else:
            normalized.append((float(point[0]), float(point[1])))
    return normalized
