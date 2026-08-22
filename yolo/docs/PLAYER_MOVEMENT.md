# Player Movement Pipeline

End-to-end documentation for turning game video into court-meter player tracks in `player_movement.json`. This pipeline powers [`play-renderer.html`](../play-renderer.html) and [`play-diagram.html`](../play-diagram.html).

## Table of contents

1. [Overview](#overview)
2. [Quick start](#quick-start)
3. [Prerequisites](#prerequisites)
4. [Architecture](#architecture)
5. [Step 1: Court detection](#step-1-court-detection)
6. [Step 2: Player tracking](#step-2-player-tracking)
7. [Per-point algorithm](#per-point-algorithm)
8. [Court projection](#court-projection)
9. [Homography smoothing](#homography-smoothing)
10. [Team classification & jersey OCR](#team-classification--jersey-ocr)
11. [Track merging & gap fill](#track-merging--gap-fill)
12. [Output format](#output-format)
13. [Configuration reference](#configuration-reference)
14. [Tuning & troubleshooting](#tuning--troubleshooting)
15. [Source files & tests](#source-files--tests)

---

## Overview

The player movement pipeline answers: **where is each player on the court over time?**

It runs in two CLI steps:

| Step | Script | Output |
|------|--------|--------|
| 1 | `detect_court.py` | `court_detect.json` — per-frame paint (D) keypoints + optional overlay video |
| 2 | `track_players.py` | `player_movement.json` — tracked players with court `(x, y)` in meters |

**Why two steps?** Court detection is expensive and shared; player tracking can be re-run with different smoothing, team, or merge settings without re-detecting the court.

**Coordinate systems:**

- **Image space** `(px, py)` — normalized 0–1, origin top-left. Foot position in the frame.
- **Court space** `(x, y)` — meters on a court preset (e.g. FIBA half = 15×14 m). `x` runs along court length, `y` across width.

---

## Quick start

```bash
cd yolo && source .venv/bin/activate

# 1. Court detect (once per clip)
python worker/scripts/detect_court.py new.mp4 \
  --json worker/output/court_detect.json \
  --preset fibaHalf \
  --device mps

# 2. Player movement
python worker/scripts/track_players.py new.mp4 \
  --court-detect worker/output/court_detect.json \
  --kit-book kit_book.json \
  --device mps
```

Open [`play-renderer.html`](../play-renderer.html) in a browser and load `worker/output/player_movement.json` to replay tracks on a top-down court.

**Important:** Use the same `--max-width` (default **1280**) in both commands so pixel coordinates align.

---

## Prerequisites

| Requirement | Purpose |
|-------------|---------|
| `worker/models/BODD_yolov8n_0001.pt` | E-BARD YOLO — player detection (`download_model.py`) |
| `worker/models/court_pose_v1.pt` | Court pose model — paint keypoints (`train_court_model.py`) |
| `worker/models/yolov8n-pose.pt` | COCO pose — ankle feet + torso crops (auto-downloaded by Ultralytics) |
| `kit_book.json` (optional) | Named home/away jersey colors for team classification |

See the main [README](../README.md) for setup, model download, and court model training.

For models, training paths, and architectural rationale, see [MODELS_AND_DESIGN.md](MODELS_AND_DESIGN.md).

---

## Architecture

```
Video
  │
  ├─ detect_court.py ──────────────────► court_detect.json
  │     YOLO court pose + snap + flow fill
  │
  └─ track_players.py ◄── court_detect.json
        │
        ├─ Frame sampling (sampleFps)
        ├─ YOLO player detect + Deep OC-SORT track
        ├─ YOLO-pose (ankles + torso)
        ├─ Foot EMA → Homography → Court xy EMA
        ├─ Team classify + Jersey OCR (post-pass)
        ├─ Filter / merge / gap fill
        └─► player_movement.json
              │
              └─ play-renderer.html / play-diagram.html
```

### Module map

| Module | Role |
|--------|------|
| `scripts/detect_court.py` | Court pose inference, overlay, JSON export |
| `scripts/track_players.py` | Main pipeline orchestrator |
| `player_tracker.py` | YOLO detect + Deep OC-SORT streaming |
| `player_pose.py` | COCO pose — ankle foot point, torso crop |
| `court_projection.py` | Static & zoom-aware homography → court meters |
| `court_overlay.py` | Paint quads, overlay fit scoring, LK flow warp |
| `team_classifier.py` | Jersey color K-means + kit book |
| `jersey_ocr.py` | Selective PaddleOCR on best crops |
| `player_config.py` | Typed config (`PipelineConfig`) |

---

## Step 1: Court detection

**Script:** `worker/scripts/detect_court.py`

Detects the **paint (D) quadrilateral** — four floor-line corners that define a homography from image pixels to court meters.

### What it does per frame

1. Sample video at `sampleFps` (same default as tracking).
2. Run `court_pose_v1.pt` (YOLO pose, class `court`, 4 keypoints).
3. **Snap** keypoints to strong floor-line edges (gradient snap).
4. **Score** the resulting overlay; reject degenerate fits.
5. On low-confidence or missed frames, **LK optical flow** warps the last good quad (`source: "flow"`).
6. Write normalized keypoints `(x, y)` in 0–1 image space.

### Key CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | — | Output `court_detect.json` |
| `--output` | — | Optional annotated MP4 |
| `--preset` | `fibaHalf` | Court dimensions: `fibaHalf`, `nbaHalf`, `fiba3x3`, `fiba`, `nba` |
| `--keypoint-order` | `0,1,2,3` | Corner order; try `3,2,1,0` or `auto` if mirrored |
| `--conf` | `0.50` | Min court detection confidence |
| `--max-width` | `1280` | Downscale width (must match tracking) |

### Keypoint order

Default **0,1,2,3** = bottom-left, bottom-right, top-right, top-left (baseline pair, then free-throw pair). Wrong order mirrors or rotates the overlay — adjust with `--keypoint-order`.

---

## Step 2: Player tracking

**Script:** `worker/scripts/track_players.py`

Consumes video + optional `court_detect.json` (or static `court_calibration.json` from [`court-marker.html`](../court-marker.html)).

### Key CLI options

| Flag | Description |
|------|-------------|
| `--court-detect PATH` | Zoom-aware homography (recommended for pan/zoom clips) |
| `--calibration PATH` | Static homography from manual markers (ignored if `--court-detect` set) |
| `--kit-book PATH` | `kit_book.json` for home/away team IDs |
| `--config PATH` | Full pipeline JSON (see [Configuration](#configuration-reference)) |
| `--no-pose` | Bbox foot + rectangle torso instead of pose ankles/crops |
| `--no-team` / `--no-ocr` / `--no-merge` | Disable stages |
| `--sample-fps` | Override frame sampling rate (default 10) |

---

## Per-point algorithm

For **each sampled frame** and **each player track**, one movement point is produced through these stages:

### 1. Frame sampling

`extract_frames.iter_sampled_frames` yields frames at `sampleFps` (default 10 Hz), optionally downscaled to `maxWidth`.

### 2. Detection & tracking

YOLO E-BARD runs `model.track()` with **Deep OC-SORT**:

- Persistent `track_id` per person
- **ReID** appearance embeddings (`withReid: true`)
- **Camera motion compensation** (`gmcMethod: sparseOptFlow`) for sideline pans
- `trackBuffer: 90` keeps lost IDs alive ~9 s at 10 fps

Each detection becomes a `PlayerObservation` with normalized bbox `(x1, y1, x2, y2)`.

Optional early filter: `--court-polygon` drops detections whose bbox foot falls outside a polygon from `hoop_roi.json`.

### 3. Pose match

If team classification is enabled and `--no-pose` is not set, YOLO-pose runs on the full frame. Each player bbox is matched to the best overlapping pose (IoU ≥ 0.25).

### 4. Foot point (image space)

**Preferred:** midpoint of **both** confident COCO ankles (indices 15, 16), normalized to 0–1, clipped near the player bbox.

**Fallback:** bbox bottom-center `( (x1+x2)/2, y2 )`.

Requiring both ankles avoids left/right ankle flicker that caused erratic court Y movement.

```python
# player_pose.py — returns None unless BOTH ankles are confident
foot = foot_xy_from_pose(obs, pose, (frame_w, frame_h), min_conf=0.4)
raw_x, raw_y = foot if foot else (obs.foot_x, obs.foot_y)
```

### 5. Foot EMA (per track)

Exponential moving average on `(px, py)` per `track_id`:

```
foot = α × raw + (1 − α) × foot_prev     (α = footEmaAlpha, default 0.3)
```

Smooths pixel jitter before homography.

### 6. Homography projection

```
court_raw = H(t) × (px, py)
```

See [Court projection](#court-projection). Returns `None` if the point lands outside court ± `courtMargin` meters — detection is **dropped** (bench, crowd, cameraman).

### 7. Court xy EMA (per track)

Second smoothing layer on court meters:

```
court = β × court_raw + (1 − β) × court_prev     (β = courtXyEmaAlpha, default 0.4)
```

Hides residual homography step changes.

### 8. Store point

If calibrated and `court_xy` is valid, append to `track_points[track_id]`:

```json
{
  "t": 1.2,
  "frame": 12,
  "px": 0.4521,
  "py": 0.7834,
  "x": 7.35,
  "y": 4.12,
  "conf": 0.87
}
```

In parallel (does not affect position): team classifier and OCR collector accumulate crops per track.

---

## Court projection

Two modes, selected at startup in `track_players.py`:

### A. Static calibration (`CourtProjector`)

From `court_calibration.json` (manual clicks in `court-marker.html`):

- One fixed 3×3 homography from ≥4 pixel↔court point pairs
- **Ignores zoom** — good for fixed-camera clips, wrong when the broadcast zooms

### B. Court detect (`PoseCourtProjector`) — recommended

From `court_detect.json`:

1. Load per-frame paint keypoints for the chosen `paintIndex` (which D on court).
2. Build a **time series of smoothed paint quads** (see [Homography smoothing](#homography-smoothing)).
3. At query time `t`, use the latest smoothed quad with `t_sec ≤ t`.
4. `cv2.findHomography(paint_pixels → court_meters)` then `perspectiveTransform` on the foot point.

**Flow frames are ignored** for player meters: when court detect falls back to optical-flow warped keypoints (`source: "flow"`), the projector holds the last pose-based homography. Flow quads jitter and would slide all players together.

**Quality gate:** Each pose quad must invert to a plausible court overlay score (`_normalized_homography_ok`) before adoption.

---

## Homography smoothing

Raw per-frame paint keypoints jitter with model noise and cause all players to “breathe” on court. `_build_pose_paint_samples` in `court_projection.py` stabilizes the quad timeline:

```
Raw pose keypoints
       │
       ▼
  Paint EMA (paintEmaAlpha = 0.25)
       │
       ▼
  Drift vs last adopted quad
       │
       ├─ drift < paintAdoptThresh ──────────► hold H
       │
       ├─ drift ≥ thresh for K frames ───────► ramp adopt (paintAdoptHysteresis = 3)
       │
       ├─ drift ≥ thresh × paintFastAdoptMult ► short ramp (real zoom/pan)
       │
       └─ ramp over paintRampFrames (8) ─────► blend quad linearly, no step jump
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `paintEmaAlpha` | 0.25 | Low-pass on incoming keypoints |
| `paintAdoptThresh` | 0.012 | Min mean vertex drift (norm. image units) to consider update |
| `paintAdoptHysteresis` | 3 | Consecutive frames above threshold before ramp starts |
| `paintRampFrames` | 8 | Frames to blend old → new quad |
| `paintFastAdoptMult` | 3.0 | Large drift bypasses hysteresis (zoom) with 2-frame ramp |
| `courtXyEmaAlpha` | 0.4 | Per-track court meter smoothing after projection |

---

## Team classification & jersey OCR

Applied **after** the frame loop, per kept track.

### Team classification (`team_classifier.py`)

Default method: **`color`** (HSV/Lab K-means on torso crops).

1. Each frame: crop jersey torso via pose shoulders/hips, or proportional bbox rectangle (`--no-pose`).
2. Extract dominant chromatic color; accumulate samples per track.
3. At `finalize()`: K-means clusters across all tracks; map clusters to **home/away** via `kit_book.json` Lab distance.
4. Assign `team` id and median `jerseyColor` RGB per track.

Optional **`dino`** method uses DINOv2 embeddings instead of raw color.

### Kit book (`kit_book.json`)

```json
{
  "teams": [
    { "id": "away", "color": [40, 80, 210], "images": ["./kits/blue_1.png", "..."] },
    { "id": "home", "color": [210, 45, 45], "images": ["./kits/red_1.png", "..."] }
  ]
}
```

Named team IDs flow into the output and renderer.

### Jersey OCR (`jersey_ocr.py`)

1. During tracking: collect sharp, large-enough number crops every N frames.
2. At `finalize()`: OCR top-K crops with PaddleOCR; confidence-weighted vote.
3. Assign `jerseyNumber` if vote share exceeds threshold.

OCR is best-effort; merge logic can still connect tracks without numbers.

---

## Track merging & gap fill

Post-processing in `track_players.py`.

### Filter kept tracks

Drop tracks that are:

- Not `"player"` label
- Fewer than `minTrackPoints` (default 15)
- Less than `minOnCourtFraction` (70%) of points projecting onto court

When calibrated, only on-court points are retained per track.

### Merge broken tracks (`merge_tracks`)

When Deep OC-SORT loses a player and assigns a new ID, fragments merge if:

- Time gap ≤ `maxGapSec` (6 s)
- Court distance ≤ `maxCourtDistance` (10 m)
- Implied speed ≤ `maxSpeedMps` (8 m/s)
- **Strong signal:** same `jerseyNumber` + same `team`
- **Weaker signal:** same team + similar jersey color (when OCR missing)

Merged player gets concatenated `points` sorted by `t` and `mergedTrackIds` listing absorbed track IDs.

### Gap fill (`fill_track_gaps`)

For gaps between consecutive points (occlusion, missed detections):

1. Linearly interpolate `(px, py)` in image space at `sampleFps` steps.
2. **Re-project** each interpolated point with `projector.project(px, py, t_sec=t)` — critical when homography changes with zoom (lerping court meters would slide under a moving camera).
3. Mark `"interpolated": true` for renderer ghost styling.

Only fills gaps ≤ `maxFillGapSec` and larger than ~1.5 sample periods.

---

## Output format

**File:** `player_movement.json`

### Top level

| Field | Description |
|-------|-------------|
| `video` | Source clip filename |
| `sampleFps` | Sampling rate used |
| `frameCount` | Number of sampled frames processed |
| `court` | `{ layout, length, width, unit, preset }` or `null` |
| `teams` | `[{ id, color }]` — cluster colors for rendering |
| `config` | Full pipeline config snapshot |
| `players` | Array of player track objects |

### Player object

| Field | Description |
|-------|-------------|
| `trackId` | Primary Deep OC-SORT ID |
| `label` | `"player"` (or `"referee"` if enabled) |
| `team` | `"home"`, `"away"`, or `null` |
| `jerseyNumber` | OCR result or `null` |
| `jerseyConfidence` | OCR vote share or `null` |
| `jerseyColor` | `[R, G, B]` median jersey color |
| `mergedTrackIds` | Other track IDs merged into this player |
| `points` | Time series of positions |

### Point object

| Field | Description |
|-------|-------------|
| `t` | Timestamp (seconds) |
| `frame` | Sampled frame index |
| `px`, `py` | Foot in normalized image coords |
| `x`, `y` | Foot in court meters (null if uncalibrated) |
| `conf` | Detection confidence (null if interpolated) |
| `interpolated` | Optional; `true` for gap-filled points |

### Court detect JSON (`court_detect.json`)

| Field | Description |
|-------|-------------|
| `format` | `"court-detect-v1"` |
| `preset` | Court preset name |
| `keypointOrder` | Corner index order |
| `paintIndex` | Which paint quad on multi-paint courts |
| `frames[]` | Per-frame records |
| `frames[].t_sec` | Timestamp |
| `frames[].keypoints[]` | `{ x, y, conf }` normalized |
| `frames[].source` | `"flow"` when LK-warped (ignored for player H) |

---

## Configuration reference

Pass `--config pipeline.json`. CLI flags override file values. Sections mirror `PipelineConfig` in `player_config.py`.

### Example config

```json
{
  "detection": {
    "sampleFps": 10,
    "maxWidth": 1280,
    "conf": 0.35,
    "footEmaAlpha": 0.3,
    "minOnCourtFraction": 0.7
  },
  "tracking": {
    "trackerType": "deepocsort",
    "trackBuffer": 90,
    "minTrackPoints": 15,
    "withReid": true,
    "gmcMethod": "sparseOptFlow"
  },
  "homography": {
    "courtDetectFile": "worker/output/court_detect.json",
    "courtMargin": 0.3,
    "clampToCourt": false,
    "paintEmaAlpha": 0.25,
    "paintAdoptThresh": 0.012,
    "paintAdoptHysteresis": 3,
    "paintRampFrames": 8,
    "paintFastAdoptMult": 3.0,
    "courtXyEmaAlpha": 0.4
  },
  "team": {
    "enabled": true,
    "method": "color",
    "kitBookFile": "kit_book.json",
    "usePose": true,
    "kmeansClusters": 3
  },
  "jerseyOcr": {
    "enabled": true
  },
  "merge": {
    "enabled": true,
    "maxGapSec": 6,
    "fillGaps": true,
    "maxFillGapSec": 6
  }
}
```

### Detection (`detection`)

| Key | Default | Description |
|-----|---------|-------------|
| `modelPath` | E-BARD weights | YOLO player detector |
| `sampleFps` | 10 | Frames per second to process |
| `maxWidth` | 1280 | Downscale width |
| `conf` | 0.35 | Min detection confidence |
| `footEmaAlpha` | 0.3 | Foot pixel smoothing (1 = off) |
| `minOnCourtFraction` | 0.7 | Min fraction of on-court points to keep track |
| `courtPolygonFile` | null | Optional polygon pre-filter |

### Tracking (`tracking`)

| Key | Default | Description |
|-----|---------|-------------|
| `trackerType` | `deepocsort` | `deepocsort` or `bytetrack` |
| `trackBuffer` | 90 | Frames to keep lost ID alive |
| `minTrackPoints` | 15 | Min points to keep a track |
| `withReid` | true | Appearance ReID |
| `appearanceThresh` | 0.92 | ReID match strictness |
| `gmcMethod` | `sparseOptFlow` | Camera motion compensation |

### Homography (`homography`)

| Key | Default | Description |
|-----|---------|-------------|
| `courtDetectFile` | null | Path to court detect JSON |
| `calibrationFile` | null | Static calibration (lower priority) |
| `courtMargin` | 0.3 | Drop points farther than this outside court (m) |
| `clampToCourt` | false | If true, drop outside court (never snap to edge) |
| `paintEmaAlpha` | 0.25 | Paint keypoint EMA |
| `paintAdoptThresh` | 0.012 | H update drift threshold |
| `paintAdoptHysteresis` | 3 | Frames before H ramp |
| `paintRampFrames` | 8 | H blend duration |
| `paintFastAdoptMult` | 3.0 | Fast ramp on large drift |
| `courtXyEmaAlpha` | 0.4 | Court meter smoothing per track |

### Team (`team`)

| Key | Default | Description |
|-----|---------|-------------|
| `method` | `color` | `color` or `dino` |
| `kitBookFile` | null | Kit book JSON path |
| `usePose` | true | Pose-based torso crop |
| `kmeansClusters` | 3 | K for color clustering |

### Merge (`merge`)

| Key | Default | Description |
|-----|---------|-------------|
| `maxGapSec` | 6 | Max seconds between fragments to merge |
| `maxCourtDistance` | 10 | Max meters between fragment endpoints |
| `maxSpeedMps` | 8 | Reject implausible merges |
| `fillGaps` | true | Interpolate missing points |
| `maxFillGapSec` | 6 | Max gap to fill |

---

## Tuning & troubleshooting

### All players jump together (Y or X pulse)

**Cause:** Homography (H) updating on noisy paint keypoints or flow-warped quads.

**Fixes:**

- Ensure `--court-detect` is used (not static calibration on zoom clips)
- Increase `paintAdoptHysteresis` (4–5) and `paintRampFrames` (12–15)
- Increase `paintAdoptThresh` (0.018–0.025)
- Lower `paintEmaAlpha` (0.15) for heavier keypoint smoothing
- Increase `courtXyEmaAlpha` smoothing (lower α = more smooth; try 0.25)

Flow keypoints are already ignored for player projection; do not re-enable them.

### Single player erratic up/down court movement

**Cause:** Foot point flicker (ankle vs bbox) or bad pose match.

**Fixes:**

- Keep pose enabled (default); both ankles required for pose foot
- Adjust `footEmaAlpha` (lower = smoother feet)
- Check pose match quality on crowded frames

### Wrong team colors

**Cause:** Torso crop contamination (shorts, floor), or missing kit book.

**Fixes:**

- Provide `kit_book.json` with representative jersey images
- Ensure `usePose: true` for shoulder/hip crops
- Increase `minSamplesPerTrack`

### Tracks split into multiple IDs

**Fixes:**

- Increase `trackBuffer`
- Enable merge (`merge.enabled: true`)
- Enable OCR for stronger merge signal
- Lower `appearanceThresh` cautiously (may confuse same-kit players)

### Players missing from output

**Cause:** Off-court projection filter (`courtMargin`, bad H) or short tracks.

**Fixes:**

- Verify court overlay in `detect_court.py` MP4 looks aligned
- Try `--keypoint-order` variants
- Lower `minTrackPoints` / `minOnCourtFraction` temporarily to diagnose

### Static vs court-detect

| Mode | When to use |
|------|-------------|
| `--calibration` | Fixed camera, no zoom, manual markers accurate |
| `--court-detect` | Broadcast pan/zoom, no manual calibration |

---

## Source files & tests

### Primary scripts

```
yolo/worker/scripts/detect_court.py    # Court pose → court_detect.json
yolo/worker/scripts/track_players.py   # Main pipeline → player_movement.json
```

### Core libraries

```
yolo/worker/player_tracker.py          # YOLO + Deep OC-SORT
yolo/worker/player_pose.py             # Ankle foot, pose torso crop
yolo/worker/court_projection.py        # Homography + smoothing
yolo/worker/court_overlay.py           # Paint quads, flow warp, scoring
yolo/worker/team_classifier.py         # Team colors
yolo/worker/jersey_ocr.py              # Jersey numbers
yolo/worker/player_config.py           # PipelineConfig
```

### Unit tests

```bash
cd yolo/worker
pytest court_projection.spec.py court_overlay.spec.py \
       team_classifier.spec.py scripts/track_players.spec.py -q
```

### Visualization

| Tool | Input |
|------|-------|
| [`play-renderer.html`](../play-renderer.html) | `player_movement.json` — animated top-down replay |
| [`play-diagram.html`](../play-diagram.html) | `player_movement.json` + reconstruction |
| [`court-marker.html`](../court-marker.html) | Manual static calibration |
| `detect_court.py --output` | Annotated court overlay MP4 |

---

## Related documentation

- [Main YOLO README](../README.md) — setup, ball tracking, court model training
- [`court-marker.html`](../court-marker.html) — manual calibration UI
- [`kit_book.json`](../kit_book.json) — example team color book
