"""Wheelchair full-court shot attempt detection from ball track + hoop ROI."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from detector import (
    DEFAULT_MODEL_PATH,
    BallDetection,
    ObjectDetection,
    create_ball_hoop_detector,
)
from extract_frames import iter_sampled_frames
from schemas import (
    AttemptDetectionConfig,
    AttemptDedupeConfig,
    AttemptsArtifact,
    BallPosition,
    BallTrackPoint,
    DetectorConfig,
    FrameHoopSnapshot,
    HoopRoi,
    HoopTrackingConfig,
    ManualHoopRoiFile,
    RimInteractionConfig,
    ShotAttempt,
)
from tracker import BallTracker

# ---------------------------------------------------------------------------
# Frame & detection state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FrameState:
    frame: int
    t_sec: float
    ball_x: float
    ball_y: float
    ball_detected: bool
    hoop_detected: bool
    hoop: HoopRoi | None = None


@dataclass
class _AttemptFrameContext:
    """Read-only view of one frame inside the attempt-detection loop."""

    frames: list[_FrameState]
    active_hoops: list[HoopRoi]
    index: int
    config: AttemptDetectionConfig

    @property
    def frame(self) -> _FrameState:
        return self.frames[self.index]

    @property
    def prev(self) -> _FrameState:
        return self.frames[self.index - 1]

    @property
    def hoop(self) -> HoopRoi:
        return self.active_hoops[self.index]

    def threshold_at(self, frame_index: int | None = None) -> float:
        idx = self.index if frame_index is None else frame_index
        return rim_threshold(self.active_hoops[idx], self.config.ring_level_margin)


@dataclass
class _DetectionState:
    """Mutable state carried across frames (re-arm + cooldown)."""

    armed: bool = True
    last_attempt_t: float = field(default_factory=lambda: -1.0)


class _AttemptGate(Protocol):
    def __call__(self, ctx: _AttemptFrameContext, state: _DetectionState) -> bool: ...


@dataclass(frozen=True)
class _TrackResult:
    frame_states: list[_FrameState]
    hoop_detections: list[ObjectDetection]
    ball_track: list[BallTrackPoint]
    ball_detected_count: int
    hoop_detected_count: int
    total_frames: int


# ---------------------------------------------------------------------------
# Hoop calibration (full-court)
# ---------------------------------------------------------------------------


def _detection_to_hoop_roi(obj: ObjectDetection) -> HoopRoi:
    return HoopRoi(x=obj.x, y=obj.y, w=obj.w, h=obj.h)


def _hoop_roi_from_detections(detections: list[ObjectDetection]) -> HoopRoi:
    return HoopRoi(
        x=float(statistics.median([h.x for h in detections])),
        y=float(statistics.median([h.y for h in detections])),
        w=float(statistics.median([h.w for h in detections])),
        h=float(statistics.median([h.h for h in detections])),
    )


def select_hoop_for_ball(
    hoops: list[ObjectDetection],
    ball: ObjectDetection | None,
) -> ObjectDetection | None:
    if not hoops:
        return None
    if ball is None or len(hoops) == 1:
        return hoops[0]
    return min(hoops, key=lambda hoop: abs(hoop.x - ball.x))


def cluster_court_hoops(
    hoops: list[ObjectDetection],
    min_conf: float = 0.0,
    min_separation: float = 0.12,
) -> list[HoopRoi]:
    """
    Cluster hoop detections into 1 or 2 basket locations for full-court video.

    Uses the largest x-axis gap between detections; if the gap is wide enough,
    treats the clip as full-court with left/right baskets.
    """
    filtered = [h for h in hoops if h.conf >= min_conf]
    if not filtered:
        filtered = list(hoops)
    if not filtered:
        raise ValueError(
            "No hoop detections found in video. Ensure the basket is visible in the clip."
        )
    if len(filtered) == 1:
        return [_detection_to_hoop_roi(filtered[0])]

    ordered = sorted(filtered, key=lambda h: h.x)
    xs = [h.x for h in ordered]
    max_gap = 0.0
    split_idx = 1
    for i in range(1, len(xs)):
        gap = xs[i] - xs[i - 1]
        if gap > max_gap:
            max_gap = gap
            split_idx = i

    if max_gap < min_separation:
        return [_hoop_roi_from_detections(filtered)]

    left = ordered[:split_idx]
    right = ordered[split_idx:]
    if not left or not right:
        return [_hoop_roi_from_detections(filtered)]

    return [_hoop_roi_from_detections(left), _hoop_roi_from_detections(right)]


def _nearest_court_index(hoop: HoopRoi, court_hoops: list[HoopRoi]) -> int:
    return min(range(len(court_hoops)), key=lambda i: abs(hoop.x - court_hoops[i].x))


def _hoop_x_jump(from_hoop: HoopRoi, to_hoop: HoopRoi) -> float:
    return abs(to_hoop.x - from_hoop.x)


def _court_midline(court_hoops: list[HoopRoi]) -> float:
    if len(court_hoops) > 1:
        return (court_hoops[0].x + court_hoops[-1].x) / 2
    return court_hoops[0].x


def _side_label(x: float, midline: float) -> str:
    return "left" if x < midline else "right"


def _ball_nearest_court_index(ball_x: float, court_hoops: list[HoopRoi]) -> int:
    return min(range(len(court_hoops)), key=lambda i: abs(ball_x - court_hoops[i].x))


def _ball_aligned_with_hoop(ball_x: float, hoop: HoopRoi, max_sep: float) -> bool:
    return abs(ball_x - hoop.x) <= max_sep


def resolve_active_hoops(
    frames: list[_FrameState],
    court_hoops: list[HoopRoi],
    config: HoopTrackingConfig | None = None,
) -> list[HoopRoi]:
    """
    Pick the shooting-court hoop per frame with temporal stabilization.

    Holds the last accepted detection per court side (left/right of midline) so a
    swish dropout on the right basket does not bleed into left-wing possessions.
    """
    if not court_hoops:
        raise ValueError("court_hoops must not be empty")

    cfg = config or HoopTrackingConfig()
    midline = _court_midline(court_hoops)
    active: list[HoopRoi] = []
    last_index = 0
    last_good_by_side: dict[str, HoopRoi] = {}
    hold_frames_by_side: dict[str, int] = {"left": 0, "right": 0}

    for frame in frames:
        candidate = frame.hoop
        accepted: HoopRoi | None = None

        if candidate is not None:
            side = _side_label(candidate.x, midline)
            same_side_anchor = last_good_by_side.get(side)
            if (
                cfg.reject_hoop_jump
                and same_side_anchor is not None
                and _hoop_x_jump(same_side_anchor, candidate) > cfg.max_hoop_jump_x
            ):
                accepted = None
            else:
                accepted = candidate

        if accepted is not None:
            side = _side_label(accepted.x, midline)
            last_good_by_side[side] = accepted
            last_index = _nearest_court_index(accepted, court_hoops)
            hold_frames_by_side[side] = 0
            active.append(accepted)
            continue

        if cfg.hold_last_good_hoop:
            ball_side: str | None = None
            if is_valid_ball_position(frame):
                ball_side = _side_label(frame.ball_x, midline)

            if ball_side is not None:
                side_hoop = last_good_by_side.get(ball_side)
                if side_hoop is not None:
                    aligned = True
                    if cfg.ball_proximity_side_lock:
                        if len(court_hoops) > 1:
                            ball_court = _ball_nearest_court_index(frame.ball_x, court_hoops)
                            hoop_court = _nearest_court_index(side_hoop, court_hoops)
                            aligned = ball_court == hoop_court
                        else:
                            aligned = _ball_aligned_with_hoop(
                                frame.ball_x,
                                side_hoop,
                                cfg.side_switch_ball_x_sep,
                            )
                    if aligned:
                        hold_limit = cfg.max_hoop_hold_frames
                        held = hold_frames_by_side[ball_side]
                        if hold_limit <= 0 or held < hold_limit:
                            hold_frames_by_side[ball_side] = held + 1
                            active.append(side_hoop)
                            continue
                        hold_frames_by_side[ball_side] = 0

        if is_valid_ball_position(frame):
            last_index = _ball_nearest_court_index(frame.ball_x, court_hoops)

        active.append(court_hoops[last_index])

    return active


def load_manual_hoop_roi_file(hoop_roi_path: str | Path) -> ManualHoopRoiFile:
    path = Path(hoop_roi_path)
    if not path.exists():
        raise FileNotFoundError(f"Hoop ROI file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ManualHoopRoiFile.model_validate(payload)


def load_court_polygon(hoop_roi_path: str | Path) -> list[tuple[float, float]] | None:
    """Load courtPolygon from hoop_roi.json (≥3 points required to filter)."""
    parsed = load_manual_hoop_roi_file(hoop_roi_path)
    if not parsed.court_polygon or len(parsed.court_polygon) < 3:
        return None
    return [(point.x, point.y) for point in parsed.court_polygon]


def load_manual_court_hoops(hoop_roi_path: str | Path) -> list[HoopRoi] | None:
    """Load manual basket ROIs from hoop_roi.json, or None if only courtPolygon is set."""
    parsed = load_manual_hoop_roi_file(hoop_roi_path)

    if parsed.court_hoops:
        return sorted(parsed.court_hoops, key=lambda hoop: hoop.x)

    hoops: list[HoopRoi] = []
    if parsed.left_hoop is not None:
        hoops.append(parsed.left_hoop)
    if parsed.right_hoop is not None:
        hoops.append(parsed.right_hoop)
    if hoops:
        return sorted(hoops, key=lambda hoop: hoop.x)

    if parsed.hoop_roi is not None:
        return [parsed.hoop_roi]

    return None


def resolve_active_hoops_manual(
    frames: list[_FrameState],
    court_hoops: list[HoopRoi],
) -> list[HoopRoi]:
    """
    Pick active basket from user-marked court hoops using ball x proximity.

    Sticky last_index when ball track is lost so swish dropouts keep the same end.
    """
    if not court_hoops:
        raise ValueError("court_hoops must not be empty")
    if len(court_hoops) == 1:
        return [court_hoops[0]] * len(frames)

    last_index = 0
    active: list[HoopRoi] = []
    for frame in frames:
        if is_valid_ball_position(frame):
            last_index = _ball_nearest_court_index(frame.ball_x, court_hoops)
        active.append(court_hoops[last_index])
    return active


def median_hoop_roi(hoops: list[ObjectDetection], min_conf: float = 0.0) -> HoopRoi:
    filtered = [h for h in hoops if h.conf >= min_conf]
    if not filtered:
        filtered = hoops
    if not filtered:
        raise ValueError(
            "No hoop detections found in video. Ensure the basket is visible in the clip."
        )
    return _hoop_roi_from_detections(filtered)


# ---------------------------------------------------------------------------
# Rim geometry
# ---------------------------------------------------------------------------


def rim_level_y(hoop: HoopRoi) -> float:
    """Normalized y of the ring plane (top of hoop bbox)."""
    return hoop.y - hoop.h / 2


def rim_threshold(hoop: HoopRoi, margin: float) -> float:
    return rim_level_y(hoop) + margin


def _rim_threshold(hoop: HoopRoi, margin: float) -> float:
    """Alias kept for tests and backward compatibility."""
    return rim_threshold(hoop, margin)


def rim_crossing_band(hoop: HoopRoi, horizontal_expand: float) -> HoopRoi:
    """Horizontal band at rim height for visualization."""
    ry = rim_level_y(hoop)
    return HoopRoi(
        x=hoop.x,
        y=ry,
        w=hoop.w * horizontal_expand,
        h=max(hoop.h * 0.15, 0.02),
    )


def is_below_rim(ball_y: float, threshold: float) -> bool:
    return ball_y > threshold


def is_at_or_above_rim(ball_y: float, threshold: float) -> bool:
    return ball_y <= threshold


def is_in_rim_approach_zone(ball_y: float, threshold: float, approach_depth: float) -> bool:
    """Just below the rim plane — ball center may never cross strict threshold on layups."""
    return threshold < ball_y <= threshold + approach_depth


def is_valid_ball_position(frame: _FrameState) -> bool:
    if frame.ball_detected:
        return True
    return not (frame.ball_x == 0.0 and frame.ball_y == 0.0)


def _deepest_below_rim(ctx: _AttemptFrameContext) -> float | None:
    margin = ctx.config.ring_level_margin
    lookback = ctx.config.lookback_frames
    threshold_curr = ctx.threshold_at()
    deepest = threshold_curr
    found = False
    start = max(0, ctx.index - lookback)

    for j in range(start, ctx.index):
        frame = ctx.frames[j]
        if not is_valid_ball_position(frame):
            continue
        threshold_j = ctx.threshold_at(j)
        if is_below_rim(frame.ball_y, threshold_j):
            found = True
            deepest = max(deepest, frame.ball_y)

    return deepest if found else None


RimSignalKind = Literal["crossed_up", "crossed_down", "rim_tube", "approach", "resume"]
_AT_RIM_SIGNALS: frozenset[RimSignalKind] = frozenset({"crossed_up", "crossed_down", "rim_tube"})


# ---------------------------------------------------------------------------
# Vertical attempt signals (any one triggers rim attempt)
# ---------------------------------------------------------------------------


def crossed_rim_downward(prev_y: float, curr_y: float, threshold: float) -> bool:
    """Ball track crossed the rim plane downward (into net / after swish)."""
    return is_at_or_above_rim(prev_y, threshold) and is_below_rim(curr_y, threshold)


def crossed_rim_downward_with_lookback(ctx: _AttemptFrameContext) -> bool:
    """Detect descent through rim plane — common when ascent frames are occluded."""
    cfg = ctx.config
    if not cfg.rim_interaction.enable_crossed_rim_downward:
        return False
    if ctx.index < 1 or cfg.lookback_frames < 1:
        return False

    threshold = ctx.threshold_at()
    curr = ctx.frame
    if not is_below_rim(curr.ball_y, threshold):
        return False

    prev = ctx.prev
    if is_valid_ball_position(prev) and crossed_rim_downward(prev.ball_y, curr.ball_y, threshold):
        return True

    min_drop = cfg.rim_interaction.min_rim_drop
    start = max(0, ctx.index - cfg.lookback_frames)
    for j in range(ctx.index - 1, start - 1, -1):
        frame = ctx.frames[j]
        if not is_valid_ball_position(frame):
            continue
        threshold_j = ctx.threshold_at(j)
        if is_at_or_above_rim(frame.ball_y, threshold_j):
            return curr.ball_y >= frame.ball_y + min_drop
    return False


def is_in_rim_tube(
    ball_x: float,
    ball_y: float,
    hoop: HoopRoi,
    threshold: float,
    horizontal_expand: float,
    above_depth: float,
    below_depth: float,
) -> bool:
    """2D rim cylinder: near hoop in x and y straddling the ring plane."""
    if not near_rim_horizontally(ball_x, hoop, horizontal_expand):
        return False
    tube_top = threshold - above_depth
    tube_bottom = threshold + below_depth
    return tube_top <= ball_y <= tube_bottom


def entered_rim_tube_with_lookback(ctx: _AttemptFrameContext) -> bool:
    """Ball entered the rim tube after being outside it — catches brief swish transits."""
    cfg = ctx.config
    rim_cfg = cfg.rim_interaction
    if not rim_cfg.enable_entered_rim_tube:
        return False
    if ctx.index < 1 or cfg.lookback_frames < 1:
        return False

    threshold = ctx.threshold_at()
    curr = ctx.frame
    if not is_in_rim_tube(
        curr.ball_x,
        curr.ball_y,
        ctx.hoop,
        threshold,
        cfg.horizontal_expand,
        rim_cfg.rim_tube_above_depth,
        rim_cfg.rim_tube_below_depth,
    ):
        return False

    start = max(0, ctx.index - cfg.lookback_frames)
    for j in range(start, ctx.index):
        frame = ctx.frames[j]
        if not is_valid_ball_position(frame):
            continue
        threshold_j = ctx.threshold_at(j)
        hoop_j = ctx.active_hoops[j]
        if not is_in_rim_tube(
            frame.ball_x,
            frame.ball_y,
            hoop_j,
            threshold_j,
            cfg.horizontal_expand,
            rim_cfg.rim_tube_above_depth,
            rim_cfg.rim_tube_below_depth,
        ):
            return True
    return False


def get_rim_signal_kind(ctx: _AttemptFrameContext) -> RimSignalKind | None:
    """First matching rim interaction signal for this frame (priority order)."""
    cfg = ctx.config
    margin = cfg.ring_level_margin

    if crossed_rim_downward_with_lookback(ctx):
        return "crossed_down"
    if crossed_rim_with_lookback(
        ctx.frames,
        ctx.active_hoops,
        ctx.index,
        margin,
        cfg.lookback_frames,
        cfg.min_rise,
    ):
        return "crossed_up"
    if entered_rim_tube_with_lookback(ctx):
        return "rim_tube"
    if approaching_rim_with_lookback(
        ctx.frames,
        ctx.active_hoops,
        ctx.index,
        margin,
        cfg.lookback_frames,
        cfg.min_rise,
        cfg.rim_approach_depth,
        cfg.min_deep_below,
    ):
        return "approach"
    if resumed_attempt_near_rim(
        ctx.frames,
        ctx.active_hoops,
        ctx.index,
        margin,
        cfg.lookback_frames,
        cfg.min_rise,
        cfg.rim_approach_depth,
        cfg.min_deep_below,
    ):
        return "resume"
    return None


def _uses_relaxed_confirm_gates(kind: RimSignalKind | None, rim_cfg: RimInteractionConfig) -> bool:
    return rim_cfg.relax_confirm_gates_at_rim and kind is not None and kind in _AT_RIM_SIGNALS


# ---------------------------------------------------------------------------
# Legacy vertical signals (kept for direct unit tests)
# ---------------------------------------------------------------------------


def crossed_rim_upward(prev_y: float, curr_y: float, threshold: float) -> bool:
    """Ball track crossed the rim plane from below (screen y decreases upward)."""
    return is_below_rim(prev_y, threshold) and is_at_or_above_rim(curr_y, threshold)


def crossed_rim_with_lookback(
    frames: list[_FrameState],
    active_hoops: list[HoopRoi],
    index: int,
    margin: float,
    lookback: int,
    min_rise: float,
) -> bool:
    ctx = _AttemptFrameContext(
        frames,
        active_hoops,
        index,
        AttemptDetectionConfig(
            ring_level_margin=margin,
            lookback_frames=lookback,
            min_rise=min_rise,
        ),
    )
    if index < 1 or lookback < 1:
        return False

    threshold = ctx.threshold_at()
    curr = ctx.frame
    if not is_at_or_above_rim(curr.ball_y, threshold):
        return False

    deepest_below = _deepest_below_rim(ctx)
    if deepest_below is None:
        return False

    return curr.ball_y <= deepest_below - min_rise


def approaching_rim_with_lookback(
    frames: list[_FrameState],
    active_hoops: list[HoopRoi],
    index: int,
    margin: float,
    lookback: int,
    min_rise: float,
    approach_depth: float,
    min_deep_below: float,
) -> bool:
    ctx = _AttemptFrameContext(
        frames,
        active_hoops,
        index,
        AttemptDetectionConfig(
            ring_level_margin=margin,
            lookback_frames=lookback,
            min_rise=min_rise,
            rim_approach_depth=approach_depth,
            min_deep_below=min_deep_below,
        ),
    )
    if index < 1 or lookback < 1:
        return False

    threshold = ctx.threshold_at()
    curr = ctx.frame
    if not is_in_rim_approach_zone(curr.ball_y, threshold, approach_depth):
        return False

    deepest_below = _deepest_below_rim(ctx)
    if deepest_below is None or deepest_below < threshold + min_deep_below:
        return False

    return curr.ball_y <= deepest_below - min_rise


def resumed_attempt_near_rim(
    frames: list[_FrameState],
    active_hoops: list[HoopRoi],
    index: int,
    margin: float,
    lookback: int,
    min_rise: float,
    approach_depth: float,
    min_deep_below: float,
) -> bool:
    ctx = _AttemptFrameContext(
        frames,
        active_hoops,
        index,
        AttemptDetectionConfig(
            ring_level_margin=margin,
            lookback_frames=lookback,
            min_rise=min_rise,
            rim_approach_depth=approach_depth,
            min_deep_below=min_deep_below,
        ),
    )
    if index < 2 or lookback < 2:
        return False

    threshold = ctx.threshold_at()
    curr = ctx.frame
    if not curr.ball_detected:
        return False

    near_rim_y = is_at_or_above_rim(curr.ball_y, threshold) or is_in_rim_approach_zone(
        curr.ball_y, threshold, approach_depth
    )
    if not near_rim_y:
        return False

    recent = frames[max(0, index - 3) : index]
    had_gap = any(not frame.ball_detected for frame in recent if is_valid_ball_position(frame))
    if not had_gap:
        prev = frames[index - 1]
        had_gap = not prev.ball_detected or not is_valid_ball_position(prev)
    if not had_gap:
        return False

    deepest_below = _deepest_below_rim(ctx)
    if deepest_below is None or deepest_below < threshold + min_deep_below:
        return False

    return curr.ball_y <= deepest_below - min_rise


def has_rim_vertical_signal(ctx: _AttemptFrameContext) -> bool:
    """Any rim interaction: cross (up/down), tube entry, approach, or resume."""
    return get_rim_signal_kind(ctx) is not None


def rim_attempt_with_lookback(
    frames: list[_FrameState],
    active_hoops: list[HoopRoi],
    index: int,
    config: AttemptDetectionConfig,
) -> bool:
    ctx = _AttemptFrameContext(frames, active_hoops, index, config)
    return has_rim_vertical_signal(ctx)


# ---------------------------------------------------------------------------
# Horizontal gates
# ---------------------------------------------------------------------------


def near_rim_horizontally(ball_x: float, hoop: HoopRoi, horizontal_expand: float) -> bool:
    half_w = (hoop.w / 2) * horizontal_expand
    return abs(ball_x - hoop.x) <= half_w


def is_vertically_near_rim(ball_y: float, threshold: float, approach_depth: float) -> bool:
    """At/above rim plane or in the just-below approach band."""
    return is_at_or_above_rim(ball_y, threshold) or is_in_rim_approach_zone(
        ball_y, threshold, approach_depth
    )


def near_hoop_horizontally_at_frame(
    ball_x: float,
    ball_y: float,
    hoop: HoopRoi,
    horizontal_expand: float,
    ring_level_margin: float,
    horizontal_vertical_depth: float,
) -> bool:
    """
    Current frame must align with the hoop in x and be vertically near the rim.

    Uses a tighter vertical band than rim_approach_depth so early sideline-arc
    x-alignments in the wide layup approach zone do not satisfy this gate.
    """
    threshold = rim_threshold(hoop, ring_level_margin)
    if not near_rim_horizontally(ball_x, hoop, horizontal_expand):
        return False
    return is_vertically_near_rim(ball_y, threshold, horizontal_vertical_depth)


def _horizontal_distance(ball_x: float, hoop: HoopRoi) -> float:
    return abs(ball_x - hoop.x)


def horizontal_convergence_with_lookback(
    frames: list[_FrameState],
    active_hoops: list[HoopRoi],
    index: int,
    horizontal_expand: float,
    lookback: int,
    min_convergence: float,
) -> bool:
    """
    Ball approached the active hoop horizontally — rejects passes exiting the paint.

    Passes if:
      - already tight on the rim and not moving away, or
      - horizontal distance to hoop decreased over the lookback window.
    """
    if index < 1 or lookback < 1:
        return False

    frame = frames[index]
    if not is_valid_ball_position(frame):
        return False

    hoop = active_hoops[index]
    dist_end = _horizontal_distance(frame.ball_x, hoop)
    half_band = (hoop.w / 2) * horizontal_expand

    if is_valid_ball_position(frames[index - 1]):
        dist_prev = _horizontal_distance(frames[index - 1].ball_x, active_hoops[index - 1])
        if dist_end > dist_prev + min_convergence:
            return False

    if dist_end <= half_band * 0.4:
        return True

    start = max(0, index - lookback)
    earliest_idx: int | None = None
    for j in range(start, index):
        if is_valid_ball_position(frames[j]):
            earliest_idx = j
            break

    if earliest_idx is None:
        return False

    dist_start = _horizontal_distance(frames[earliest_idx].ball_x, active_hoops[earliest_idx])
    return dist_end < dist_start - min_convergence


# ---------------------------------------------------------------------------
# Declarative attempt gates (applied in order each frame)
# ---------------------------------------------------------------------------


def _gate_valid_ball_position(ctx: _AttemptFrameContext, _state: _DetectionState) -> bool:
    return is_valid_ball_position(ctx.frame)


def _gate_shot_rearmed(ctx: _AttemptFrameContext, state: _DetectionState) -> bool:
    if state.armed:
        return True
    exit_depth = ctx.config.shot_exit_depth
    if ctx.frame.ball_y > ctx.threshold_at() + exit_depth:
        state.armed = True
    return False


def _gate_rim_vertical_signal(ctx: _AttemptFrameContext, _state: _DetectionState) -> bool:
    return has_rim_vertical_signal(ctx)


def _gate_near_hoop_horizontally(ctx: _AttemptFrameContext, _state: _DetectionState) -> bool:
    cfg = ctx.config
    frame = ctx.frame
    kind = get_rim_signal_kind(ctx)
    vertical_depth = cfg.horizontal_vertical_depth
    if _uses_relaxed_confirm_gates(kind, cfg.rim_interaction):
        vertical_depth = cfg.rim_approach_depth
    return near_hoop_horizontally_at_frame(
        frame.ball_x,
        frame.ball_y,
        ctx.hoop,
        cfg.horizontal_expand,
        cfg.ring_level_margin,
        vertical_depth,
    )


def _gate_horizontal_convergence(ctx: _AttemptFrameContext, _state: _DetectionState) -> bool:
    cfg = ctx.config
    kind = get_rim_signal_kind(ctx)
    if _uses_relaxed_confirm_gates(kind, cfg.rim_interaction):
        hoop = ctx.hoop
        dist_end = _horizontal_distance(ctx.frame.ball_x, hoop)
        half_band = (hoop.w / 2) * cfg.horizontal_expand
        if dist_end <= half_band:
            return True
    return horizontal_convergence_with_lookback(
        ctx.frames,
        ctx.active_hoops,
        ctx.index,
        cfg.horizontal_expand,
        cfg.lookback_frames,
        cfg.min_horizontal_convergence,
    )


def _gate_cooldown_elapsed(ctx: _AttemptFrameContext, state: _DetectionState) -> bool:
    return ctx.frame.t_sec - state.last_attempt_t >= ctx.config.cooldown_sec


ATTEMPT_DETECTION_GATES: tuple[tuple[str, _AttemptGate], ...] = (
    ("valid_ball_position", _gate_valid_ball_position),
    ("shot_rearmed", _gate_shot_rearmed),
    ("rim_vertical_signal", _gate_rim_vertical_signal),
    ("near_hoop_horizontally", _gate_near_hoop_horizontally),
    ("horizontal_convergence", _gate_horizontal_convergence),
    ("cooldown_elapsed", _gate_cooldown_elapsed),
)


def passes_attempt_gates(ctx: _AttemptFrameContext, state: _DetectionState) -> bool:
    return all(gate(ctx, state) for _, gate in ATTEMPT_DETECTION_GATES)


# ---------------------------------------------------------------------------
# Attempt scoring & emission
# ---------------------------------------------------------------------------


def _velocity(frames: list[_FrameState], index: int, window: int) -> tuple[float, float]:
    if index < window:
        return 0.0, 0.0
    prev = frames[index - window]
    curr = frames[index]
    dt = float(window)
    return (curr.ball_x - prev.ball_x) / dt, (curr.ball_y - prev.ball_y) / dt


def _confidence(
    entry_speed: float,
    upward: bool,
    near_horizontally: bool,
    ball_detected: bool,
    hoop_detected: bool,
) -> float:
    """Soft score only — does not gate attempts."""
    speed_score = min(entry_speed / 0.08, 1.0) if entry_speed > 0 else 0.0
    signal_score = sum(
        [
            0.3 if upward else 0.0,
            0.3 if near_horizontally else 0.0,
            0.2 if ball_detected else 0.0,
            0.2 if hoop_detected else 0.0,
        ]
    )
    return round(0.3 * speed_score + 0.7 * signal_score, 3)


def _build_shot_attempt(ctx: _AttemptFrameContext) -> ShotAttempt:
    cfg = ctx.config
    vx, vy = _velocity(ctx.frames, ctx.index, cfg.velocity_window)
    speed = math.hypot(vx, vy)
    upward = ctx.frame.ball_y < ctx.prev.ball_y
    near_x = near_rim_horizontally(ctx.frame.ball_x, ctx.hoop, cfg.horizontal_expand)
    signal_kind = get_rim_signal_kind(ctx)

    return ShotAttempt(
        frame=ctx.frame.frame,
        tSec=ctx.frame.t_sec,
        confidence=_confidence(
            entry_speed=speed,
            upward=upward,
            near_horizontally=near_x,
            ball_detected=ctx.frame.ball_detected,
            hoop_detected=ctx.frame.hoop_detected,
        ),
        entrySpeed=round(speed, 4),
        ballPos=BallPosition(x=round(ctx.frame.ball_x, 4), y=round(ctx.frame.ball_y, 4)),
        signalKind=signal_kind,
    )


def _mark_attempt_triggered(ctx: _AttemptFrameContext, state: _DetectionState) -> None:
    state.last_attempt_t = ctx.frame.t_sec
    state.armed = False


# ---------------------------------------------------------------------------
# Post-process: dedupe same-arc double triggers
# ---------------------------------------------------------------------------


def _attempts_in_same_cluster(
    earlier: ShotAttempt,
    later: ShotAttempt,
    config: AttemptDedupeConfig,
) -> bool:
    dt = later.t_sec - earlier.t_sec
    if dt <= 0 or dt > config.cluster_window_sec:
        return False
    dx = abs(earlier.ball_pos.x - later.ball_pos.x)
    return dx <= config.max_ball_x_sep


def _pick_primary_attempt(cluster: list[ShotAttempt]) -> ShotAttempt:
    """Prefer rim-cross (lower y), then later frame, then higher confidence."""
    return min(
        cluster,
        key=lambda attempt: (
            attempt.ball_pos.y,
            -attempt.t_sec,
            -attempt.confidence,
        ),
    )


def dedupe_shot_attempts(
    attempts: list[ShotAttempt],
    config: AttemptDedupeConfig | None = None,
) -> list[ShotAttempt]:
    """
    Mark likely duplicate triggers from one physical shot as deduped.

    All attempts are kept in the output for manual review; deduped entries
    point at primaryFrame for the canonical event in the cluster.
    """
    if not attempts:
        return []

    cfg = config or AttemptDedupeConfig()
    if not cfg.enabled:
        return list(attempts)

    ordered = sorted(attempts, key=lambda attempt: attempt.t_sec)
    clusters: list[list[ShotAttempt]] = [[ordered[0]]]
    for attempt in ordered[1:]:
        if _attempts_in_same_cluster(clusters[-1][-1], attempt, cfg):
            clusters[-1].append(attempt)
        else:
            clusters.append([attempt])

    marked: list[ShotAttempt] = []
    for cluster in clusters:
        primary = _pick_primary_attempt(cluster)
        for attempt in cluster:
            if attempt.frame == primary.frame:
                marked.append(attempt.model_copy())
                continue
            marked.append(
                attempt.model_copy(
                    update={"deduped": True, "primary_frame": primary.frame},
                )
            )

    return sorted(marked, key=lambda attempt: attempt.t_sec)


# ---------------------------------------------------------------------------
# Frame-sequence attempt detection
# ---------------------------------------------------------------------------


def detect_attempts_from_frames(
    frames: list[_FrameState],
    active_hoops: list[HoopRoi],
    config: AttemptDetectionConfig | None = None,
) -> list[ShotAttempt]:
    """
    Detect shot attempts by applying ATTEMPT_DETECTION_GATES to each frame.

    Gate order:
      1. valid_ball_position   — skip lost-track placeholders
      2. shot_rearmed          — one attempt per ascent; re-arm after exiting rim zone
      3. rim_vertical_signal   — cross (up/down), tube entry, approach, or resume
      4. near_hoop_horizontally — current frame near hoop x (relaxed band at rim)
      5. horizontal_convergence — toward rim (waived when already on rim)
      6. cooldown_elapsed      — min seconds since last attempt
    """
    if len(active_hoops) != len(frames):
        raise ValueError("active_hoops must align one-to-one with frames")

    cfg = config or AttemptDetectionConfig()
    state = _DetectionState(last_attempt_t=-cfg.cooldown_sec)
    attempts: list[ShotAttempt] = []

    for index in range(1, len(frames)):
        ctx = _AttemptFrameContext(frames, active_hoops, index, cfg)
        if not passes_attempt_gates(ctx, state):
            continue
        attempts.append(_build_shot_attempt(ctx))
        _mark_attempt_triggered(ctx, state)

    return attempts


# ---------------------------------------------------------------------------
# Video pipeline
# ---------------------------------------------------------------------------


def _to_ball_detection(obj: ObjectDetection) -> BallDetection:
    return BallDetection(x=obj.x, y=obj.y, w=obj.w, h=obj.h, conf=obj.conf)


def _region_anchor_hoop(
    manual_hoops: list[HoopRoi] | None,
    last_detected_hoop: HoopRoi | None,
    last_ball_x: float | None,
) -> HoopRoi | None:
    """Pick hoop ROI used for hoop-region ball detection this frame."""
    if manual_hoops:
        if len(manual_hoops) == 1:
            return manual_hoops[0]
        if last_ball_x is not None:
            return min(manual_hoops, key=lambda hoop: abs(hoop.x - last_ball_x))
        return manual_hoops[0]
    return last_detected_hoop


class _BallHoopDetector(Protocol):
    def detect_ball_with_hoop_fallback(
        self,
        frame_bgr,
        hoop: HoopRoi | None,
        expand: float = 3.5,
        detect_hoop_classes: list[str] | None = None,
    ) -> tuple[BallDetection | None, dict[str, ObjectDetection | None]]: ...

    def detect_hoops(self, frame_bgr) -> list[ObjectDetection]: ...


def _wheelchair_default_config() -> AttemptDetectionConfig:
    """
    Defaults tuned for wheelchair full-court clips with a left-side camera.

    Attempt rules stay unchanged; only hoop-tracking stabilization defaults are
    nudged so side-locking is more stable for long full-court pans.
    """
    return AttemptDetectionConfig(
        hoop_tracking=HoopTrackingConfig(
            hold_last_good_hoop=True,
            reject_hoop_jump=True,
            max_hoop_jump_x=0.12,
            max_hoop_hold_frames=0,
            ball_proximity_side_lock=True,
            side_switch_ball_x_sep=0.16,
        )
    )


def _track_video_frames(
    video_path: str | Path,
    detector: _BallHoopDetector,
    tracker: BallTracker,
    sample_fps: float,
    on_progress: Callable[[int], None] | None,
    detect_hoop: bool,
    manual_hoops: list[HoopRoi] | None,
    ball_detection_cfg: AttemptDetectionConfig,
) -> _TrackResult:
    cfg = ball_detection_cfg
    ball_cfg = cfg.ball_detection
    sampled_frames = list(iter_sampled_frames(str(video_path), sample_fps=sample_fps))
    hoop_detections: list[ObjectDetection] = []
    frame_states: list[_FrameState] = []
    ball_detected_count = 0
    hoop_detected_count = 0
    total = len(sampled_frames)
    last_detected_hoop: HoopRoi | None = None
    last_ball_x: float | None = None
    class_names = ["basketball", "hoop"] if detect_hoop else ["basketball"]

    for i, sampled in enumerate(sampled_frames):
        region_anchor = _region_anchor_hoop(manual_hoops, last_detected_hoop, last_ball_x)
        hoop_obj: ObjectDetection | None = None

        if ball_cfg.enable_hoop_region:
            _, detections = detector.detect_ball_with_hoop_fallback(
                sampled.bgr,
                region_anchor,
                expand=ball_cfg.hoop_region_expand,
                detect_hoop_classes=class_names,
            )
        else:
            detections = detector.detect_objects(sampled.bgr, class_names)

        ball_obj = detections.get("basketball")

        if detect_hoop:
            hoop_candidates = detector.detect_hoops(sampled.bgr)
            hoop_obj = select_hoop_for_ball(hoop_candidates, ball_obj)

            if hoop_obj is not None:
                hoop_detections.append(hoop_obj)
                hoop_detected_count += 1
                last_detected_hoop = _detection_to_hoop_roi(hoop_obj)

        ball_det = _to_ball_detection(ball_obj) if ball_obj is not None else None
        track_hoop = region_anchor or last_detected_hoop
        track_point = tracker.step(sampled.index, ball_det, hoop=track_hoop)

        if ball_obj is not None:
            ball_detected_count += 1
            last_ball_x = track_point.x
        elif track_point.predicted:
            last_ball_x = track_point.x

        frame_hoop = _detection_to_hoop_roi(hoop_obj) if hoop_obj is not None else None
        frame_states.append(
            _FrameState(
                frame=sampled.index,
                t_sec=sampled.t_sec,
                ball_x=track_point.x,
                ball_y=track_point.y,
                ball_detected=track_point.detected,
                hoop_detected=hoop_obj is not None if detect_hoop else False,
                hoop=frame_hoop,
            )
        )

        if on_progress and total > 0:
            on_progress(int((i + 1) / total * 100))

    return _TrackResult(
        frame_states=frame_states,
        hoop_detections=hoop_detections,
        ball_track=list(tracker.points),
        ball_detected_count=ball_detected_count,
        hoop_detected_count=hoop_detected_count,
        total_frames=total,
    )


def _build_active_hoop_track(
    frame_states: list[_FrameState],
    active_hoops: list[HoopRoi],
) -> list[FrameHoopSnapshot]:
    return [
        FrameHoopSnapshot(frame=frame.frame, hoop=hoop, rimY=rim_level_y(hoop))
        for frame, hoop in zip(frame_states, active_hoops, strict=True)
    ]


def process_video_to_attempts(
    video_path: str | Path,
    sample_fps: float = 20.0,
    ball_conf: float = 0.05,
    hoop_conf: float = 0.60,
    model_path: Path = DEFAULT_MODEL_PATH,
    ball_model_path: Path | None = None,
    config: AttemptDetectionConfig | None = None,
    hoop_roi_path: str | Path | None = None,
    device: str | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> AttemptsArtifact:
    cfg = config or _wheelchair_default_config()
    manual_hoops = load_manual_court_hoops(hoop_roi_path) if hoop_roi_path is not None else None
    court_polygon = load_court_polygon(hoop_roi_path) if hoop_roi_path is not None else None
    hoop_source: Literal["detected", "manual"] = "manual" if manual_hoops else "detected"

    detector_config = DetectorConfig(ball_conf=ball_conf, hoop_conf=hoop_conf)
    if device is not None:
        detector_config = detector_config.model_copy(update={"device": device})

    detector = create_ball_hoop_detector(
        model_path=model_path,
        config=detector_config,
        court_polygon=court_polygon,
        ball_model_path=ball_model_path,
    )
    tracker = BallTracker(sample_fps=sample_fps, config=cfg.ball_tracking)

    # 1. Detect & track ball (and hoop unless manual ROI is provided)
    track = _track_video_frames(
        video_path,
        detector,
        tracker,
        sample_fps,
        on_progress,
        detect_hoop=manual_hoops is None,
        manual_hoops=manual_hoops,
        ball_detection_cfg=cfg,
    )

    # 2. Court basket location(s)
    if manual_hoops is not None:
        court_hoops = manual_hoops
        full_court = len(court_hoops) > 1
    else:
        court_hoops = cluster_court_hoops(track.hoop_detections, min_conf=hoop_conf * 0.95)
        full_court = len(court_hoops) > 1

    # 3. Active shooting hoop per frame
    if manual_hoops is not None:
        active_hoops = resolve_active_hoops_manual(track.frame_states, court_hoops)
    else:
        active_hoops = resolve_active_hoops(track.frame_states, court_hoops, cfg.hoop_tracking)
    active_hoop_track = _build_active_hoop_track(track.frame_states, active_hoops)

    # 4. Run declarative attempt gates on the frame sequence
    raw_attempts = detect_attempts_from_frames(track.frame_states, active_hoops, cfg)
    attempts = dedupe_shot_attempts(raw_attempts, cfg.dedupe)

    primary_hoop = court_hoops[0]
    effective_fps = sample_fps
    if track.frame_states:
        last_t = track.frame_states[-1].t_sec
        if last_t > 0:
            effective_fps = (track.frame_states[-1].frame + 1) / last_t

    if manual_hoops is not None:
        hoop_detection_rate = 1.0
    else:
        hoop_detection_rate = (
            track.hoop_detected_count / track.total_frames if track.total_frames else 0.0
        )

    return AttemptsArtifact(
        video=str(video_path),
        sampleFps=effective_fps,
        hoopRoi=primary_hoop,
        rimY=rim_level_y(primary_hoop),
        attemptZone=rim_crossing_band(primary_hoop, cfg.horizontal_expand),
        courtHoops=court_hoops,
        fullCourt=full_court,
        activeHoopTrack=active_hoop_track,
        hoopSource=hoop_source,
        hoopDetectionRate=hoop_detection_rate,
        ballDetectionRate=track.ball_detected_count / track.total_frames if track.total_frames else 0.0,
        ballTrack=track.ball_track,
        attempts=attempts,
    )
