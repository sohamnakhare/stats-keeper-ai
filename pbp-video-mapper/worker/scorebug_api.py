"""Load a scorebug video URL and crop-relative field boxes from the local API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import quote

from .schemas import ScoreboardRegion

_DEFAULT_API_BASE = "http://localhost:3000"

_FIELD_NAMES = {
    "homeScore": "home_score",
    "awayScore": "away_score",
    "gameClock": "game_clock",
    "shotClock": "shot_clock",
}


@dataclass(frozen=True)
class ScorebugVideo:
    """Video URL plus the scorebug crop and OCR fields."""

    video_url: str
    scorebug: ScoreboardRegion
    fields: dict[str, ScoreboardRegion]


def scorebug_api_base() -> str:
    """Return the scorebug API origin.

    ``SCOREBUG_API_BASE`` is set when the image is built. Local runs fall
    back to localhost when it is unset or blank.
    """
    configured = os.environ.get("SCOREBUG_API_BASE", "").strip()
    base = configured or _DEFAULT_API_BASE
    return base.rstrip("/")


def fetch_scorebug_video(video_id: str) -> ScorebugVideo:
    """GET /api/scorebug-videos/{id} and return the URL and markings."""
    video_id = video_id.strip()
    if not video_id:
        raise ValueError("Video id is required")

    url = f"{scorebug_api_base()}/api/scorebug-videos/{quote(video_id, safe='')}"
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Scorebug API failed ({exc.code}) for id {video_id}"
        ) from exc

    return parse_scorebug_video(payload)


def parse_scorebug_video(payload: dict) -> ScorebugVideo:
    """Turn an API payload into a scorebug crop and crop-relative fields."""
    video = payload.get("video")
    if not isinstance(video, dict):
        raise ValueError("Scorebug API response must include 'video'")

    video_url = video.get("videoUrl")
    if not isinstance(video_url, str) or not video_url.strip():
        raise ValueError("Scorebug API response must include 'videoUrl'")

    markings = video.get("markings")
    if not isinstance(markings, dict):
        raise ValueError("Scorebug API response must include 'markings'")

    scorebug = _region_from_box(markings.get("scorebug"), "scorebug")
    if scorebug.w <= 0 or scorebug.h <= 0:
        raise ValueError("scorebug region must have positive width and height")

    raw_fields = markings.get("fields")
    if not isinstance(raw_fields, dict) or not raw_fields:
        raise ValueError("Scorebug API markings must include at least one field")

    fields: dict[str, ScoreboardRegion] = {}
    for name, box in raw_fields.items():
        ocr_name = _FIELD_NAMES.get(str(name), str(name))
        if ocr_name in fields:
            raise ValueError(f"Duplicate field '{name}'")
        region = _region_from_box(box, str(name))
        if region.w <= 0 or region.h <= 0:
            raise ValueError(f"Field '{name}' must have positive width and height")
        if not _inside_crop(region):
            raise ValueError(f"Field '{name}' must lie inside the scorebug crop")
        fields[ocr_name] = region

    return ScorebugVideo(
        video_url=video_url.strip(),
        scorebug=scorebug,
        fields=fields,
    )


def _region_from_box(box: object, name: str) -> ScoreboardRegion:
    if not isinstance(box, dict):
        raise ValueError(f"'{name}' must be a box with x, y, w, and h")
    try:
        return ScoreboardRegion(
            x=float(box["x"]),
            y=float(box["y"]),
            w=float(box["w"]),
            h=float(box["h"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' must be a box with x, y, w, and h") from exc


def _inside_crop(region: ScoreboardRegion, eps: float = 1e-6) -> bool:
    return (
        region.x >= -eps
        and region.y >= -eps
        and region.x + region.w <= 1 + eps
        and region.y + region.h <= 1 + eps
    )
