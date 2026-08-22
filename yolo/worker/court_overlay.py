"""Project a full court diagram from 4 paint (D) keypoints."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Sequence

import cv2
import numpy as np

PRESETS: dict[str, dict[str, float | str]] = {
    "fiba": {
        "layout": "full",
        "length": 28.0,
        "width": 15.0,
        "paint_width": 4.9,
        "free_throw": 5.8,
        "rim_dist": 1.575,
        "arc_radius": 6.75,
        "ft_circle": 1.8,
        "corner": 0.9,
    },
    "nba": {
        "layout": "full",
        "length": 28.65,
        "width": 15.24,
        "paint_width": 4.88,
        "free_throw": 5.79,
        "rim_dist": 1.575,
        "arc_radius": 7.24,
        "ft_circle": 1.8,
        "corner": 0.91,
    },
    "fibaHalf": {
        "layout": "half",
        "length": 15.0,
        "width": 14.0,
        "paint_width": 4.9,
        "free_throw": 5.8,
        "rim_dist": 1.575,
        "arc_radius": 6.75,
        "ft_circle": 1.8,
        "corner": 0.9,
    },
    "nbaHalf": {
        "layout": "half",
        "length": 15.24,
        "width": 14.325,
        "paint_width": 4.88,
        "free_throw": 5.79,
        "rim_dist": 1.575,
        "arc_radius": 7.24,
        "ft_circle": 1.8,
        "corner": 0.91,
    },
    "fiba3x3": {
        "layout": "half",
        "length": 15.0,
        "width": 11.0,
        "paint_width": 4.9,
        "free_throw": 5.8,
        "rim_dist": 1.575,
        "arc_radius": 6.75,
        "ft_circle": 1.8,
        "corner": 0.9,
    },
}

# Model skeleton (image / D quad): 0 bottom-left, 1 bottom-right, 2 top-right, 3 top-left.
# Mapped to paint meters as baseL, baseR, ftR, ftL.
DEFAULT_KEYPOINT_ORDER: tuple[int, int, int, int] = (0, 1, 2, 3)
MIN_KEYPOINT_CONF = 0.20
SNAP_RADIUS = 14
EMA_ALPHA = 0.40
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
FLOW_KEYPOINT_CONF = 0.15
_SCORE_REF_AREA = 640.0 * 360.0

BOUNDARY_COLOR = (80, 200, 80)  # green BGR
PAINT_COLOR = (240, 80, 240)  # magenta BGR
ARC_COLOR = (240, 200, 56)  # cyan-ish BGR
RIM_COLOR = (0, 220, 255)  # yellow BGR


@dataclass(frozen=True)
class OverlayFit:
    homography: np.ndarray  # court meters -> image pixels
    order: tuple[int, int, int, int]
    score: float
    paint_index: int  # 0 = primary (half / left), 1 = opposite full-court paint


def parse_keypoint_order(raw: str | None) -> tuple[int, int, int, int] | None:
    """Parse '0,1,2,3' as model keypoint indices for baseL, baseR, ftR, ftL.

    Returns None for auto-search when raw is empty or 'auto'.
    """
    if raw is None or not raw.strip() or raw.strip().lower() == "auto":
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--keypoint-order must be four comma-separated indices, e.g. 0,1,2,3")
    try:
        order = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"Invalid --keypoint-order: {raw}") from exc
    if sorted(order) != [0, 1, 2, 3]:
        raise ValueError("--keypoint-order must be a permutation of 0,1,2,3")
    return order  # type: ignore[return-value]


def paint_quads(preset: dict[str, float | str]) -> list[np.ndarray]:
    """Paint corner sets in court meters: baseL, baseR, ftR, ftL."""
    length = float(preset["length"])
    width = float(preset["width"])
    paint_w = float(preset["paint_width"])
    ft = float(preset["free_throw"])
    layout = str(preset["layout"])

    if layout == "half":
        x_left = length / 2 - paint_w / 2
        x_right = length / 2 + paint_w / 2
        return [
            np.array(
                [[x_left, 0.0], [x_right, 0.0], [x_right, ft], [x_left, ft]],
                dtype=np.float64,
            )
        ]

    y_top = width / 2 - paint_w / 2
    y_bot = width / 2 + paint_w / 2
    left = np.array(
        [[0.0, y_top], [0.0, y_bot], [ft, y_bot], [ft, y_top]],
        dtype=np.float64,
    )
    right = np.array(
        [[length, y_top], [length, y_bot], [length - ft, y_bot], [length - ft, y_top]],
        dtype=np.float64,
    )
    return [left, right]


def court_to_pixels(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    src = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, homography)
    return dst.reshape(-1, 2)


def gradient_magnitude(frame_bgr: np.ndarray) -> np.ndarray:
    """Normalized Sobel magnitude in [0, 1] for line snapping."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    peak = float(np.max(mag))
    if peak <= 1e-6:
        return mag
    return mag / peak


def snap_point_to_gradient(
    mag: np.ndarray,
    x: float,
    y: float,
    radius: int = SNAP_RADIUS,
    outward_from: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Move (x, y) to the strongest nearby edge, preferring outside the D."""
    height, width = mag.shape[:2]
    ix, iy = int(round(x)), int(round(y))
    if not (1 <= ix < width - 1 and 1 <= iy < height - 1):
        return x, y

    best_x, best_y = float(ix), float(iy)
    best_s = float(mag[iy, ix])
    orig_dist = 0.0
    if outward_from is not None:
        orig_dist = float(np.hypot(ix - outward_from[0], iy - outward_from[1]))

    for dy in range(-radius, radius + 1):
        ny = iy + dy
        if ny < 1 or ny >= height - 1:
            continue
        for dx in range(-radius, radius + 1):
            nx = ix + dx
            if nx < 1 or nx >= width - 1:
                continue
            score = float(mag[ny, nx])
            if outward_from is not None:
                dist = float(np.hypot(nx - outward_from[0], ny - outward_from[1]))
                if dist > orig_dist:
                    score *= 1.08
            if score > best_s:
                best_s = score
                best_x, best_y = float(nx), float(ny)
    return best_x, best_y


def snap_keypoints(
    pixel_pts: Sequence[tuple[float, float]],
    mag: np.ndarray,
    radius: int = SNAP_RADIUS,
) -> list[tuple[float, float]]:
    if len(pixel_pts) != 4:
        return list(pixel_pts)
    cx = sum(p[0] for p in pixel_pts) / 4.0
    cy = sum(p[1] for p in pixel_pts) / 4.0
    return [snap_point_to_gradient(mag, x, y, radius, (cx, cy)) for x, y in pixel_pts]


def warp_keypoints_lk(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    prev_pts: Sequence[tuple[float, float]] | np.ndarray,
) -> np.ndarray | None:
    """Track 4 paint points with pyramidal Lucas-Kanade.

    Returns an (4, 2) float64 array, or None if any point is lost or the
    quad collapses.
    """
    pts = np.asarray(prev_pts, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) != 4 or prev_gray.size == 0 or gray.size == 0:
        return None
    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        pts,
        None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_pts is None or status is None:
        return None
    if not np.all(status.reshape(-1) == 1):
        return None
    warped = next_pts.reshape(4, 2).astype(np.float64)
    height, width = gray.shape[:2]
    if np.any(warped[:, 0] < 0) or np.any(warped[:, 0] > width - 1):
        return None
    if np.any(warped[:, 1] < 0) or np.any(warped[:, 1] > height - 1):
        return None
    area = abs(cv2.contourArea(warped.astype(np.float32)))
    if area < 8.0:
        return None
    return warped


def keypoints_confident(
    confs: Sequence[float],
    min_conf: float = MIN_KEYPOINT_CONF,
) -> bool:
    return len(confs) == 4 and all(c >= min_conf for c in confs)


def clip_segment_to_rect(
    p0: tuple[float, float] | np.ndarray,
    p1: tuple[float, float] | np.ndarray,
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Liang-Barsky clip of a segment to the image rectangle."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    xmin, xmax = 0.0, float(width - 1)
    ymin, ymax = 0.0, float(height - 1)
    p_vals = (-dx, dx, -dy, dy)
    q_vals = (x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
    u1, u2 = 0.0, 1.0
    for p, q in zip(p_vals, q_vals):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            u1 = max(u1, t)
        else:
            u2 = min(u2, t)
        if u1 > u2:
            return None
    return (x0 + u1 * dx, y0 + u1 * dy), (x0 + u2 * dx, y0 + u2 * dy)


def _sample_court_segment(a: np.ndarray, b: np.ndarray, steps: int = 24) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    return (1.0 - t)[:, None] * a + t[:, None] * b


def _signed_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _score_homography(
    homography: np.ndarray,
    preset: dict[str, float | str],
    paint_pts: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> float:
    length = float(preset["length"])
    width = float(preset["width"])
    outline = np.array([[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]], dtype=np.float64)
    corners = court_to_pixels(homography, outline)
    pad_x, pad_y = 0.25 * frame_w, 0.25 * frame_h
    inside = sum(
        1
        for x, y in corners
        if -pad_x < x < frame_w + pad_x and -pad_y < y < frame_h + pad_y
    )
    if inside < 3:
        return -1e9

    court_area = abs(cv2.contourArea(corners.astype(np.float32)))
    paint = court_to_pixels(homography, paint_pts)
    paint_area = abs(cv2.contourArea(paint.astype(np.float32)))
    frame_area = float(frame_w * frame_h)
    min_paint = 8.0 * frame_area / _SCORE_REF_AREA
    if paint_area < min_paint or court_area < paint_area * 1.15:
        return -1e9

    score = float(inside) * 10.0 + min(court_area / frame_area, 2.0) * 5.0 + court_area / paint_area
    court_sign = np.sign(_signed_area(paint_pts))
    image_sign = np.sign(_signed_area(paint))
    if court_sign != 0 and image_sign == court_sign:
        score += 20.0
    return score


def fit_overlay(
    pixel_pts: Sequence[tuple[float, float]],
    preset: dict[str, float | str],
    frame_w: int,
    frame_h: int,
    order: tuple[int, int, int, int] | None = None,
) -> OverlayFit | None:
    """Find court-meters -> pixels homography from 4 D keypoints."""
    if len(pixel_pts) != 4:
        return None
    pixels = np.array(pixel_pts, dtype=np.float64)
    quads = paint_quads(preset)
    orders: list[tuple[int, int, int, int]]
    if order is not None:
        orders = [order]
    else:
        orders = list(permutations(range(4)))  # type: ignore[assignment]

    best: OverlayFit | None = None
    for paint_index, court_pts in enumerate(quads):
        for candidate in orders:
            dst = np.array([pixels[i] for i in candidate], dtype=np.float64)
            matrix, _ = cv2.findHomography(court_pts, dst, method=0)
            if matrix is None:
                continue
            score = _score_homography(matrix, preset, court_pts, frame_w, frame_h)
            if best is None or score > best.score:
                best = OverlayFit(
                    homography=matrix,
                    order=candidate,
                    score=score,
                    paint_index=paint_index,
                )
    if best is None or best.score < 0:
        return None
    return best


def _polyline(
    frame: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    closed: bool,
) -> None:
    if len(pts) < 2:
        return
    height, width = frame.shape[:2]
    count = len(pts)
    edges = count if closed else count - 1
    for i in range(edges):
        a = pts[i]
        b = pts[(i + 1) % count] if closed else pts[i + 1]
        clipped = clip_segment_to_rect(a, b, width, height)
        if clipped is None:
            continue
        c0, c1 = clipped
        cv2.line(
            frame,
            (int(round(c0[0])), int(round(c0[1]))),
            (int(round(c1[0])), int(round(c1[1]))),
            color,
            2,
            cv2.LINE_AA,
        )


def _project_and_draw_segment(
    frame: np.ndarray,
    homography: np.ndarray,
    a: Sequence[float],
    b: Sequence[float],
    color: tuple[int, int, int],
    steps: int = 24,
) -> None:
    """Project a court-meter segment densely so off-screen ends still draw in-frame."""
    samples = _sample_court_segment(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), steps)
    pixels = court_to_pixels(homography, samples)
    _polyline(frame, pixels, color, closed=False)


def draw_court_diagram(
    frame: np.ndarray,
    homography: np.ndarray,
    preset: dict[str, float | str],
) -> None:
    """Draw boundary, paint, 3pt/2pt arc, and rim from a court->pixel homography."""
    length = float(preset["length"])
    width = float(preset["width"])
    paint_w = float(preset["paint_width"])
    ft = float(preset["free_throw"])
    rim_dist = float(preset["rim_dist"])
    arc_r = float(preset["arc_radius"])
    ft_circle = float(preset["ft_circle"])
    corner = float(preset["corner"])
    layout = str(preset["layout"])

    outline = [
        ([0.0, 0.0], [length, 0.0]),
        ([length, 0.0], [length, width]),
        ([length, width], [0.0, width]),
        ([0.0, width], [0.0, 0.0]),
    ]
    for a, b in outline:
        _project_and_draw_segment(frame, homography, a, b, BOUNDARY_COLOR)

    if layout == "half":
        mid_x = length / 2
        x_left = mid_x - paint_w / 2
        x_right = mid_x + paint_w / 2
        paint_edges = [
            ([x_left, 0.0], [x_right, 0.0]),
            ([x_right, 0.0], [x_right, ft]),
            ([x_right, ft], [x_left, ft]),
            ([x_left, ft], [x_left, 0.0]),
        ]
        for a, b in paint_edges:
            _project_and_draw_segment(frame, homography, a, b, PAINT_COLOR)

        ft_arc = np.array(
            [
                [
                    mid_x + ft_circle * np.cos(np.deg2rad(deg)),
                    ft + ft_circle * np.sin(np.deg2rad(deg)),
                ]
                for deg in range(0, 181, 4)
            ],
            dtype=np.float64,
        )
        _polyline(frame, court_to_pixels(homography, ft_arc), PAINT_COLOR, closed=False)

        cos_end = min(1.0, (mid_x - corner) / arc_r)
        end_angle = float(np.arccos(cos_end))
        arc = np.array(
            [
                [
                    mid_x + arc_r * np.cos(a),
                    rim_dist + arc_r * np.sin(a),
                ]
                for a in np.linspace(end_angle, np.pi - end_angle, 48)
            ],
            dtype=np.float64,
        )
        _polyline(frame, court_to_pixels(homography, arc), ARC_COLOR, closed=False)
        y_end = rim_dist + arc_r * np.sin(end_angle)
        for cx in (corner, length - corner):
            _project_and_draw_segment(
                frame,
                homography,
                [cx, 0.0],
                [cx, min(y_end, width)],
                ARC_COLOR,
            )

        rim = court_to_pixels(homography, np.array([[mid_x, rim_dist]], dtype=np.float64))[0]
        cv2.circle(frame, (int(round(rim[0])), int(round(rim[1]))), 5, RIM_COLOR, 2, cv2.LINE_AA)
        return

    mid_x = length / 2
    _project_and_draw_segment(frame, homography, [mid_x, 0.0], [mid_x, width], BOUNDARY_COLOR)

    y_top = width / 2 - paint_w / 2
    y_bot = width / 2 + paint_w / 2
    for x0, x1 in ((0.0, ft), (length, length - ft)):
        paint_edges = [
            ([x0, y_top], [x1, y_top]),
            ([x1, y_top], [x1, y_bot]),
            ([x1, y_bot], [x0, y_bot]),
            ([x0, y_bot], [x0, y_top]),
        ]
        for a, b in paint_edges:
            _project_and_draw_segment(frame, homography, a, b, PAINT_COLOR)

    for basket_x, sign in ((rim_dist, 1.0), (length - rim_dist, -1.0)):
        rim = court_to_pixels(homography, np.array([[basket_x, width / 2]], dtype=np.float64))[0]
        cv2.circle(frame, (int(round(rim[0])), int(round(rim[1]))), 5, RIM_COLOR, 2, cv2.LINE_AA)
        arc = np.array(
            [
                [
                    basket_x + sign * arc_r * np.sin(a),
                    width / 2 + arc_r * np.cos(a),
                ]
                for a in np.linspace(-np.pi / 2, np.pi / 2, 40)
            ],
            dtype=np.float64,
        )
        _polyline(frame, court_to_pixels(homography, arc), ARC_COLOR, closed=False)


def outline_polygon_normalized(
    homography: np.ndarray,
    preset: dict[str, float | str],
    frame_w: int,
    frame_h: int,
) -> list[dict[str, float]]:
    length = float(preset["length"])
    width = float(preset["width"])
    outline = np.array([[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]], dtype=np.float64)
    pixels = court_to_pixels(homography, outline)
    return [
        {"x": round(float(x) / frame_w, 6), "y": round(float(y) / frame_h, 6)}
        for x, y in pixels
    ]
