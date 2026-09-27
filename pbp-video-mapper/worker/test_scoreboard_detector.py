"""Unit tests for scorebug crop remapping."""

from __future__ import annotations

import pytest

from worker.schemas import ScoreboardRegion
from worker.scoreboard_detector import (
    prepare_scorebug_regions,
    remap_region_into_scorebug,
)


def _region(x: float, y: float, w: float, h: float) -> ScoreboardRegion:
    return ScoreboardRegion(x=x, y=y, w=w, h=h)


def test_remap_region_into_scorebug() -> None:
    scorebug = _region(0.1, 0.7, 0.4, 0.2)
    clock = _region(0.12, 0.72, 0.08, 0.05)

    remapped = remap_region_into_scorebug(clock, scorebug)

    assert remapped.x == pytest.approx(0.05)
    assert remapped.y == pytest.approx(0.1)
    assert remapped.w == pytest.approx(0.2)
    assert remapped.h == pytest.approx(0.25)


def test_prepare_keeps_inner_regions_and_drops_scorebug() -> None:
    regions = {
        "scorebug": _region(0.1, 0.7, 0.4, 0.2),
        "game_clock": _region(0.12, 0.72, 0.08, 0.05),
        "home_score": _region(0.22, 0.72, 0.06, 0.05),
    }

    scorebug, inner = prepare_scorebug_regions(regions)

    assert scorebug == regions["scorebug"]
    assert "scorebug" not in inner
    assert set(inner) == {"game_clock", "home_score"}
    assert inner["game_clock"].x == pytest.approx(0.05)


def test_prepare_rejects_missing_scorebug() -> None:
    with pytest.raises(ValueError, match="scorebug"):
        prepare_scorebug_regions({"game_clock": _region(0.1, 0.1, 0.2, 0.1)})


def test_prepare_rejects_region_outside_scorebug() -> None:
    regions = {
        "scorebug": _region(0.1, 0.7, 0.4, 0.2),
        "game_clock": _region(0.05, 0.72, 0.08, 0.05),
    }

    with pytest.raises(ValueError, match="game_clock"):
        prepare_scorebug_regions(regions)
