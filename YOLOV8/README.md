# YOLOV8 basketball play-diagram pipeline

Standalone Python pipeline: basketball clip → player/ball tracks in court feet → top-down play diagram.

Does **not** import `yolo/worker`. Reuses E-BARD weights, sample clips, and `court-calibration-v1` JSON from that folder.

## Setup

```bash
cd YOLOV8
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py
```

The downloader copies `yolo/worker/models/BODD_yolov8n_0001.pt` if it already exists, otherwise fetches E-BARD YOLOv8n from Hugging Face.

You can also reuse the existing `yolo/.venv` — it already has Ultralytics, OpenCV, and matplotlib.

## Run

Stage 1 only (confirm boxes on a sample clip before the rest of the pipeline):

```bash
python run_pipeline.py --input ../yolo/new.mp4 --stage detect --output-dir output
```

Inspect `output/detections.json` and `output/detections_annotated.mp4`.

Full MVP (detect → track → homography → smooth → stub events → diagram):

```bash
python run_pipeline.py \
  --input ../yolo/new.mp4 \
  --calibration ../yolo/court_calibration.json \
  --output output/play_diagram.png
```

`--stage detect|track|calibrate|trajectories|events|render|all` runs through that stage. Later stages read JSON from `--output-dir`.

## Stages

| Stage | Module | Output |
|---|---|---|
| 1 Detection | `pipeline/detect.py` | `detections.json`, `detections_annotated.mp4` |
| 2 Tracking | `pipeline/track.py` | `tracks.json` (ByteTrack players, Kalman ball) |
| 3 Homography | `pipeline/calibrate.py` | `court_tracks.json` (feet) |
| 4 Trajectories | `pipeline/trajectories.py` | `trajectories.json` |
| 5 Events | `pipeline/events.py` | `events.json` (**stub**) |
| 6 Render | `pipeline/render.py` | PNG + SVG |

## Calibration

Automatic court-line detection is not used — it is unreliable on broadcast/sideline video. Mark ≥4 landmarks in [`yolo/court-marker.html`](../yolo/court-marker.html) and pass `--calibration`. Meter calibrations are converted to feet.

## Accuracy notes

Off-the-shelf models will not be perfect:

- **Ball** — tiny and motion-blurred. Fine-tune the ball class (this repo already has `yolo/worker/scripts/train_ball_model.py`) rather than switching YOLO versions.
- **Players** — bench/crowd fire as `player`. Raise `--conf` or mark a court polygon; a court-aware fine-tune is the longer fix.
- **IDs** — ByteTrack drops through screens/occlusions; `gaps` in `tracks.json` flag this. Stage 4 interpolates short holes.
- **COCO fallback** (`yolov8n.pt`) is only used if E-BARD weights are missing. Expect missed balls and extra people.
