"""Unit tests for attempt detection rules."""

from __future__ import annotations

from attempt_detector import (
    _FrameState,
    approaching_rim_with_lookback,
    cluster_court_hoops,
    crossed_rim_downward,
    crossed_rim_upward,
    crossed_rim_with_lookback,
    dedupe_shot_attempts,
    detect_attempts_from_frames,
    entered_rim_tube_with_lookback,
    horizontal_convergence_with_lookback,
    is_vertically_near_rim,
    load_manual_court_hoops,
    load_court_polygon,
    near_hoop_horizontally_at_frame,
    resolve_active_hoops,
    resolve_active_hoops_manual,
    resumed_attempt_near_rim,
    rim_attempt_with_lookback,
    _region_anchor_hoop,
    _AttemptFrameContext,
    _rim_threshold,
)
from detector import ObjectDetection, hoop_search_region
from schemas import (
    AttemptDedupeConfig,
    AttemptDetectionConfig,
    BallPosition,
    HoopRoi,
    HoopTrackingConfig,
    RimInteractionConfig,
    ShotAttempt,
)


def _frame(i: int, x: float, y: float, hoop: HoopRoi | None = None) -> _FrameState:
    return _FrameState(
        frame=i,
        t_sec=float(i),
        ball_x=x,
        ball_y=y,
        ball_detected=True,
        hoop_detected=hoop is not None,
        hoop=hoop,
    )


def _hoops(active: HoopRoi, count: int) -> list[HoopRoi]:
    return [active] * count


HOOP = HoopRoi(x=0.65, y=0.27, w=0.05, h=0.09)
LEFT = HoopRoi(x=0.25, y=0.27, w=0.05, h=0.09)
RIGHT = HoopRoi(x=0.75, y=0.27, w=0.05, h=0.09)
CFG = AttemptDetectionConfig(cooldown_sec=0.0)
THRESHOLD = _rim_threshold(HOOP, CFG.ring_level_margin)


def test_cluster_court_hoops_splits_full_court() -> None:
    detections = [
        ObjectDetection("hoop", 0.24, 0.27, 0.05, 0.09, 0.9),
        ObjectDetection("hoop", 0.26, 0.27, 0.05, 0.09, 0.85),
        ObjectDetection("hoop", 0.74, 0.27, 0.05, 0.09, 0.92),
        ObjectDetection("hoop", 0.76, 0.27, 0.05, 0.09, 0.88),
    ]
    clusters = cluster_court_hoops(detections)
    assert len(clusters) == 2
    assert clusters[0].x < clusters[1].x


def test_resolve_active_hoop_follows_ball_end() -> None:
    frames = [
        _frame(0, 0.24, 0.40),
        _frame(1, 0.76, 0.40),
    ]
    active = resolve_active_hoops(frames, [LEFT, RIGHT])
    assert active[0].x == LEFT.x
    assert active[1].x == RIGHT.x


def test_hold_last_good_hoop_on_detection_miss() -> None:
    """Swish dropout: keep last per-frame hoop instead of court median fallback."""
    right_hoop = HoopRoi(x=0.86, y=0.27, w=0.05, h=0.09)
    court_median = HoopRoi(x=0.47, y=0.26, w=0.05, h=0.09)
    frames = [
        _FrameState(0, 0.0, 0.82, 0.24, True, True, right_hoop),
        _FrameState(1, 0.05, 0.83, 0.23, True, False, None),
        _FrameState(2, 0.10, 0.84, 0.24, True, False, None),
    ]
    active = resolve_active_hoops(frames, [court_median])
    assert active[0].x == right_hoop.x
    assert active[1].x == right_hoop.x
    assert active[2].x == right_hoop.x


def test_reject_spurious_hoop_jump() -> None:
    right_hoop = HoopRoi(x=0.86, y=0.27, w=0.05, h=0.09)
    spurious_center = HoopRoi(x=0.47, y=0.26, w=0.05, h=0.09)
    court_median = spurious_center
    frames = [
        _FrameState(0, 0.0, 0.82, 0.24, True, True, right_hoop),
        _FrameState(1, 0.05, 0.83, 0.23, True, True, spurious_center),
    ]
    active = resolve_active_hoops(frames, [court_median])
    assert active[1].x == right_hoop.x


def test_side_lock_uses_left_anchor_not_right_hold() -> None:
    """Right-side hold must not apply when ball is on the left wing."""
    right_hoop = HoopRoi(x=0.86, y=0.27, w=0.05, h=0.09)
    left_hoop = HoopRoi(x=0.35, y=0.27, w=0.05, h=0.09)
    court_median = HoopRoi(x=0.47, y=0.26, w=0.05, h=0.09)
    frames = [
        _FrameState(0, 0.0, 0.82, 0.24, True, True, right_hoop),
        _FrameState(1, 0.05, 0.83, 0.23, True, False, None),
        _FrameState(2, 0.10, 0.30, 0.24, True, True, left_hoop),
        _FrameState(3, 0.15, 0.28, 0.22, True, False, None),
    ]
    active = resolve_active_hoops(frames, [court_median])
    assert active[1].x == right_hoop.x
    assert active[2].x == left_hoop.x
    assert active[3].x == left_hoop.x


def test_hoop_hold_disabled_falls_back_to_court_median() -> None:
    right_hoop = HoopRoi(x=0.86, y=0.27, w=0.05, h=0.09)
    court_median = HoopRoi(x=0.47, y=0.26, w=0.05, h=0.09)
    frames = [
        _FrameState(0, 0.0, 0.82, 0.24, True, True, right_hoop),
        _FrameState(1, 0.05, 0.83, 0.23, True, False, None),
    ]
    cfg = HoopTrackingConfig(hold_last_good_hoop=False)
    active = resolve_active_hoops(frames, [court_median], cfg)
    assert active[1].x == court_median.x


def test_resolve_active_hoops_manual_picks_nearest_end() -> None:
    left = HoopRoi(x=0.31, y=0.27, w=0.05, h=0.09)
    right = HoopRoi(x=0.83, y=0.27, w=0.05, h=0.09)
    frames = [
        _frame(0, 0.30, 0.40),
        _frame(1, 0.82, 0.40),
        _FrameState(2, 2.0, 0.82, 0.38, False, False, None),
    ]
    active = resolve_active_hoops_manual(frames, [left, right])
    assert active[0].x == left.x
    assert active[1].x == right.x
    assert active[2].x == right.x


def test_load_manual_court_hoops_from_json() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hoop_roi.json"
        path.write_text(
            """
            {
              "leftHoop": {"x": 0.31, "y": 0.25, "w": 0.06, "h": 0.10},
              "rightHoop": {"x": 0.83, "y": 0.28, "w": 0.06, "h": 0.11}
            }
            """,
            encoding="utf-8",
        )
        hoops = load_manual_court_hoops(path)
        assert len(hoops) == 2
        assert hoops[0].x == 0.31
        assert hoops[1].x == 0.83


def test_load_court_polygon_from_json() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hoop_roi.json"
        path.write_text(
            """
            {
              "hoopRoi": {"x": 0.5, "y": 0.25, "w": 0.06, "h": 0.10},
              "courtPolygon": [
                {"x": 0.1, "y": 0.2},
                {"x": 0.9, "y": 0.2},
                {"x": 0.9, "y": 0.9},
                {"x": 0.1, "y": 0.9}
              ]
            }
            """,
            encoding="utf-8",
        )
        polygon = load_court_polygon(path)
        assert polygon is not None
        assert len(polygon) == 4
        assert polygon[0] == (0.1, 0.2)


def test_load_manual_court_hoops_returns_none_for_court_polygon_only() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hoop_roi.json"
        path.write_text(
            """
            {
              "leftHoop": null,
              "rightHoop": null,
              "courtHoops": [],
              "courtPolygon": [
                {"x": 0.1, "y": 0.2},
                {"x": 0.9, "y": 0.2},
                {"x": 0.9, "y": 0.9},
                {"x": 0.1, "y": 0.9}
              ]
            }
            """,
            encoding="utf-8",
        )
        assert load_manual_court_hoops(path) is None
        polygon = load_court_polygon(path)
        assert polygon is not None
        assert len(polygon) == 4


def test_crossed_rim_upward_requires_immediate_pair() -> None:
    assert crossed_rim_upward(0.26, 0.23, THRESHOLD)
    assert not crossed_rim_upward(0.23, 0.23, THRESHOLD)


def test_crossed_rim_downward_requires_immediate_pair() -> None:
    assert crossed_rim_downward(THRESHOLD - 0.01, THRESHOLD + 0.02, THRESHOLD)
    assert not crossed_rim_downward(THRESHOLD + 0.02, THRESHOLD + 0.03, THRESHOLD)


def test_swish_downward_cross_detects_with_relaxed_confirm() -> None:
    """First track point at rim then net exit — upward cross may lack lookback."""
    frames = [
        _frame(0, 0.65, THRESHOLD - 0.01),
        _frame(1, 0.68, THRESHOLD + 0.04),
    ]
    active = _hoops(HOOP, len(frames))
    assert crossed_rim_downward(frames[0].ball_y, frames[1].ball_y, THRESHOLD)
    assert not horizontal_convergence_with_lookback(
        frames, active, 1, CFG.horizontal_expand, CFG.lookback_frames, CFG.min_horizontal_convergence
    )

    no_relax = AttemptDetectionConfig(
        cooldown_sec=0.0,
        rim_interaction=RimInteractionConfig(relax_confirm_gates_at_rim=False),
    )
    assert len(detect_attempts_from_frames(frames, active, no_relax)) == 0

    attempts = detect_attempts_from_frames(frames, active, CFG)
    assert len(attempts) == 1
    assert attempts[0].frame == 1
    assert attempts[0].signal_kind == "crossed_down"


def test_entered_rim_tube_detects_swish_transit() -> None:
    frames = [
        _frame(0, 0.72, THRESHOLD + 0.10),
        _frame(1, 0.65, THRESHOLD + 0.02),
    ]
    active = _hoops(HOOP, len(frames))
    ctx = _AttemptFrameContext(frames, active, 1, CFG)
    assert entered_rim_tube_with_lookback(ctx)
    attempts = detect_attempts_from_frames(frames, active, CFG)
    assert len(attempts) == 1
    assert attempts[0].signal_kind == "rim_tube"


def test_lookback_catches_gap_before_rim() -> None:
    frames = [
        _frame(0, 0.64, 0.30),
        _frame(1, 0.64, 0.23),
        _frame(2, 0.64, 0.22),
    ]
    active = _hoops(HOOP, len(frames))
    assert not crossed_rim_upward(frames[1].ball_y, frames[2].ball_y, THRESHOLD)
    assert crossed_rim_with_lookback(
        frames, active, 2, CFG.ring_level_margin, CFG.lookback_frames, CFG.min_rise
    )


def test_approach_zone_catches_underbasket_finish() -> None:
    frames = [
        _frame(0, 0.65, 0.32),
        _frame(1, 0.65, 0.28),
        _frame(2, 0.65, THRESHOLD + 0.03),
    ]
    active = _hoops(HOOP, len(frames))
    assert not crossed_rim_with_lookback(
        frames, active, 2, CFG.ring_level_margin, CFG.lookback_frames, CFG.min_rise
    )
    assert approaching_rim_with_lookback(
        frames,
        active,
        2,
        CFG.ring_level_margin,
        CFG.lookback_frames,
        CFG.min_rise,
        CFG.rim_approach_depth,
        CFG.min_deep_below,
    )
    assert rim_attempt_with_lookback(frames, active, 2, CFG)


def test_resume_after_occlusion_near_rim() -> None:
    frames = [
        _frame(0, 0.65, THRESHOLD + 0.06),
        _frame(1, 0.65, THRESHOLD + 0.04),
        _FrameState(2, 2.0, 0.65, THRESHOLD + 0.02, False, True),
        _frame(3, 0.65, THRESHOLD + 0.01),
    ]
    active = _hoops(HOOP, len(frames))
    assert resumed_attempt_near_rim(
        frames,
        active,
        3,
        CFG.ring_level_margin,
        CFG.lookback_frames,
        CFG.min_rise,
        CFG.rim_approach_depth,
        CFG.min_deep_below,
    )


def test_horizontal_gate_requires_current_frame_near_x() -> None:
    frames = [_frame(0, 0.64, THRESHOLD + 0.01), _frame(1, 0.71, THRESHOLD - 0.01)]
    active = _hoops(HOOP, len(frames))
    assert near_hoop_horizontally_at_frame(
        frames[0].ball_x,
        frames[0].ball_y,
        active[0],
        CFG.horizontal_expand,
        CFG.ring_level_margin,
        CFG.horizontal_vertical_depth,
    )
    assert not near_hoop_horizontally_at_frame(
        frames[1].ball_x,
        frames[1].ball_y,
        active[1],
        CFG.horizontal_expand,
        CFG.ring_level_margin,
        CFG.horizontal_vertical_depth,
    )


def test_horizontal_gate_rejects_lookback_only_alignment() -> None:
    """Old lookback could pass when only an earlier frame aligned in x."""
    frames = [_frame(0, 0.64, 0.30), _frame(1, 0.71, 0.24)]
    active = _hoops(HOOP, len(frames))
    assert not near_hoop_horizontally_at_frame(
        frames[1].ball_x,
        frames[1].ball_y,
        active[1],
        CFG.horizontal_expand,
        CFG.ring_level_margin,
        CFG.horizontal_vertical_depth,
    )


def test_horizontal_gate_rejects_wide_approach_band_alignment() -> None:
    """Approach-zone y with x near hoop must not pass the tighter horizontal band."""
    y = THRESHOLD + 0.05
    assert not near_hoop_horizontally_at_frame(
        0.65,
        y,
        HOOP,
        CFG.horizontal_expand,
        CFG.ring_level_margin,
        CFG.horizontal_vertical_depth,
    )
    assert is_vertically_near_rim(y, THRESHOLD, CFG.rim_approach_depth)


def test_left_three_pointer_single_attempt() -> None:
    """Ball x aligns with hoop twice on arc; only rim-height pass should count."""
    threshold = THRESHOLD
    frames = [
        _frame(0, 0.65, threshold + 0.14),  # early x-alignment, far below rim
        _frame(1, 0.72, threshold + 0.10),
        _frame(2, 0.68, threshold + 0.06),  # x near but outside tight vertical band
        _frame(3, 0.64, threshold - 0.01),  # real rim cross
        _frame(4, 0.65, threshold + 0.02),
    ]
    active = _hoops(HOOP, len(frames))
    attempts = detect_attempts_from_frames(frames, active, CFG)
    assert len(attempts) == 1
    assert attempts[0].frame == 3


def test_approach_vertical_without_horizontal_does_not_double() -> None:
    """Early approach-zone vertical signal must not emit when x/y fail horizontal gate."""
    threshold = THRESHOLD
    frames = [
        _frame(0, 0.72, threshold + 0.06),
        _frame(1, 0.68, threshold + 0.06),  # approach zone but outside rim tube
        _frame(2, 0.64, threshold - 0.01),
    ]
    active = _hoops(HOOP, len(frames))
    attempts = detect_attempts_from_frames(frames, active, CFG)
    assert len(attempts) == 1
    assert attempts[0].frame == 2


def test_horizontal_convergence_rejects_pass_away() -> None:
    frames = [
        _frame(0, 0.64, THRESHOLD + 0.08),
        _frame(1, 0.64, THRESHOLD + 0.04),
        _frame(2, 0.72, THRESHOLD + 0.01),
    ]
    active = _hoops(HOOP, len(frames))
    assert not horizontal_convergence_with_lookback(
        frames, active, 2, CFG.horizontal_expand, CFG.lookback_frames, CFG.min_horizontal_convergence
    )


def test_horizontal_convergence_allows_shot_toward_rim() -> None:
    frames = [
        _frame(0, 0.72, THRESHOLD + 0.08),
        _frame(1, 0.68, THRESHOLD + 0.04),
        _frame(2, 0.64, THRESHOLD - 0.01),
    ]
    active = _hoops(HOOP, len(frames))
    assert horizontal_convergence_with_lookback(
        frames, active, 2, CFG.horizontal_expand, CFG.lookback_frames, CFG.min_horizontal_convergence
    )


def test_detect_finish_after_occlusion_gap() -> None:
    frames = [
        _frame(0, 0.65, THRESHOLD + 0.07),
        _frame(1, 0.65, THRESHOLD + 0.085),  # below rim but outside approach band
        _FrameState(2, 2.0, 0.0, 0.0, False, False, None),
        _frame(3, 0.65, THRESHOLD - 0.01),
    ]
    active = _hoops(HOOP, len(frames))
    attempts = detect_attempts_from_frames(frames, active, CFG)
    assert len(attempts) == 1
    assert attempts[0].frame == 3


def _attempt(frame: int, t_sec: float, x: float, y: float) -> ShotAttempt:
    return ShotAttempt(
        frame=frame,
        tSec=t_sec,
        confidence=0.9,
        entrySpeed=0.1,
        ballPos=BallPosition(x=x, y=y),
    )


def test_dedupe_marks_same_arc_cluster() -> None:
    attempts = [
        _attempt(1104, 55.22, 0.378, 0.166),
        _attempt(1131, 56.58, 0.384, 0.1485),
    ]
    result = dedupe_shot_attempts(attempts, AttemptDedupeConfig())
    assert len(result) == 2
    primary = next(a for a in result if not a.deduped)
    duplicate = next(a for a in result if a.deduped)
    assert primary.frame == 1131
    assert duplicate.frame == 1104
    assert duplicate.primary_frame == 1131


def test_dedupe_keeps_separate_shots() -> None:
    attempts = [
        _attempt(100, 10.0, 0.38, 0.20),
        _attempt(200, 20.0, 0.39, 0.19),
    ]
    result = dedupe_shot_attempts(attempts, AttemptDedupeConfig())
    assert all(not a.deduped for a in result)


def test_dedupe_keeps_far_apart_same_time_window() -> None:
    attempts = [
        _attempt(100, 10.0, 0.20, 0.20),
        _attempt(110, 10.8, 0.75, 0.19),
    ]
    result = dedupe_shot_attempts(attempts, AttemptDedupeConfig())
    assert all(not a.deduped for a in result)


def test_hoop_search_region_clamps_to_frame() -> None:
    hoop = HoopRoi(x=0.02, y=0.02, w=0.05, h=0.09)
    x1, y1, x2, y2 = hoop_search_region(hoop, expand=3.5)
    assert x1 >= 0.0
    assert y1 >= 0.0
    assert x2 <= 1.0
    assert y2 <= 1.0
    assert x2 > x1
    assert y2 > y1


def test_region_anchor_hoop_manual_single() -> None:
    hoop = HoopRoi(x=0.5, y=0.25, w=0.05, h=0.09)
    anchor = _region_anchor_hoop([hoop], None, None)
    assert anchor is not None
    assert anchor.x == hoop.x


def test_region_anchor_hoop_manual_dual_follows_ball() -> None:
    left = HoopRoi(x=0.25, y=0.27, w=0.05, h=0.09)
    right = HoopRoi(x=0.75, y=0.27, w=0.05, h=0.09)
    anchor = _region_anchor_hoop([left, right], None, 0.72)
    assert anchor is not None
    assert anchor.x == right.x
