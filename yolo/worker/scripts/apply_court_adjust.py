#!/usr/bin/env python3
"""Apply a play-renderer court_adjust.json to a court_calibration.json.

The adjust transform maps stored player court coords -> corrected court coords
(court diagram fixed). We compose it with the existing pixel->court homography:

    H_new = T @ H_old

then rewrite each landmark so court meters stay the same (true paint corners etc.)
and pixel locations are updated to match H_new:

    pixel' = inv(H_new) @ court

Re-run track_players.py with the new calibration file afterward.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

WORKER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKER_ROOT))

from court_projection import CourtProjector  # noqa: E402
from player_config import HomographyConfig  # noqa: E402


def similarity_matrix(
    *,
    tx: float,
    ty: float,
    rotation_deg: float,
    scale: float,
    flip_x: bool,
    pivot_x: float,
    pivot_y: float,
) -> np.ndarray:
    """3x3 homography for center-pivoted flip/scale/rotate + translation."""
    c, s = np.cos(np.radians(rotation_deg)), np.sin(np.radians(rotation_deg))
    to_origin = np.array(
        [[1.0, 0.0, -pivot_x], [0.0, 1.0, -pivot_y], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    flip = np.array(
        [[-1.0 if flip_x else 1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rot_scale = np.array(
        [[scale * c, -scale * s, 0.0], [scale * s, scale * c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    from_origin = np.array(
        [[1.0, 0.0, pivot_x], [0.0, 1.0, pivot_y], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    translate = np.array(
        [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return translate @ from_origin @ rot_scale @ flip @ to_origin


def apply_h(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    pts = np.array([[[x, y]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pts, matrix)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update court_calibration.json from play-renderer court_adjust.json"
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Original court_calibration.json from court-marker.html",
    )
    parser.add_argument(
        "--adjust",
        type=Path,
        required=True,
        help="court_adjust.json exported from play-renderer.html",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output calibration path (default: <calibration>.adjusted.json)",
    )
    args = parser.parse_args()

    if not args.calibration.exists():
        print(f"Calibration not found: {args.calibration}", file=sys.stderr)
        sys.exit(1)
    if not args.adjust.exists():
        print(f"Adjust file not found: {args.adjust}", file=sys.stderr)
        sys.exit(1)

    calib = json.loads(args.calibration.read_text(encoding="utf-8"))
    adjust_doc = json.loads(args.adjust.read_text(encoding="utf-8"))
    if adjust_doc.get("format") != "court-adjust-v1":
        print(
            f"Unexpected adjust format: {adjust_doc.get('format')!r} (expected court-adjust-v1)",
            file=sys.stderr,
        )
        sys.exit(1)

    transform = adjust_doc.get("transform") or {}
    court_info = calib.get("court") or {}
    length = float(court_info.get("length", adjust_doc.get("court", {}).get("length", 28)))
    width = float(court_info.get("width", adjust_doc.get("court", {}).get("width", 15)))

    projector = CourtProjector.from_calibration_file(args.calibration, HomographyConfig())
    t_mat = similarity_matrix(
        tx=float(transform.get("tx", 0.0)),
        ty=float(transform.get("ty", 0.0)),
        rotation_deg=float(transform.get("rotationDeg", 0.0)),
        scale=float(transform.get("scale", 1.0)),
        flip_x=bool(transform.get("flipX", False)),
        pivot_x=length / 2.0,
        pivot_y=width / 2.0,
    )
    h_new = t_mat @ projector.matrix
    h_inv = np.linalg.inv(h_new)

    new_points = []
    for pt in calib.get("points") or []:
        court = pt["court"]
        cx, cy = float(court["x"]), float(court["y"])
        px, py = apply_h(h_inv, cx, cy)
        new_points.append(
            {
                **pt,
                "court": {"x": cx, "y": cy},
                "pixel": {"x": round(px, 6), "y": round(py, 6)},
            }
        )

    output = {
        **calib,
        "source": "play-renderer-adjust",
        "updatedAt": adjust_doc.get("updatedAt"),
        "adjust": {
            "from": str(args.adjust.name),
            "transform": transform,
        },
        "points": new_points,
    }

    out_path = args.output
    if out_path is None:
        out_path = args.calibration.with_name(args.calibration.stem + ".adjusted.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Landmarks updated: {len(new_points)}")
    print("Re-run tracking with:")
    print(
        f"  python worker/scripts/track_players.py <video> "
        f"--calibration {out_path} --output worker/output/player_movement.json"
    )


if __name__ == "__main__":
    main()
