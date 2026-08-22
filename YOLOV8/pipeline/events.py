"""Stage 5: event detection (stub).

Not implemented in the MVP. Downstream renderers should treat `events` as empty.

Next-step heuristics (once trajectories are reliable):

Passes
    The ball trajectory leaves player A's proximity zone (e.g. 4–6 ft) and
    enters player B's zone within a short window, without a bounce pattern
    that looks like a dribble. Tag `{kind: "pass", t, fromId, toId}`.
    A dashed line on the diagram is the usual playbook glyph.

Dribbles
    Ball y (or speed) oscillates while remaining near one player. Playbook
    glyph is a squiggly / zigzag polyline along that segment.

Screens
    Player S becomes nearly stationary (speed below ~1.5 ft/s) while teammate
    C's path passes within ~3 ft of S, and a defender D's path is deflected
    or delayed near the same timestamp. Tag `{kind: "screen", t, screenerId,
    userId}` and draw a filled circle / T-bar on S's location.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.schemas import FORMAT_EVENTS


def detect_events(_trajectories: dict | None = None) -> list[dict]:
    """Return detected play events. MVP: always empty."""
    return []


def run_events(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    traj_path = output_dir / "trajectories.json"
    trajectories = None
    if traj_path.exists():
        trajectories = json.loads(traj_path.read_text(encoding="utf-8"))
    events = detect_events(trajectories)
    artifact = {
        "format": FORMAT_EVENTS,
        "events": events,
        "status": "stub",
        "note": (
            "Pass / screen / dribble detection is not implemented. "
            "See pipeline/events.py for the intended heuristics."
        ),
    }
    out_json = output_dir / "events.json"
    out_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Stage 5 events: stub ({len(events)} events) → {out_json}")
    return artifact
