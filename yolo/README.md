# YOLO Ball Tracking

Standalone Python worker for E-BARD basketball detection and frame-by-frame tracking.

## Setup

```bash
cd yolo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then install the detection model (required — see [Model](#model)).

## Evaluate detection (metrics only)

```bash
python worker/scripts/eval_ebard.py /path/to/clip.mp4
python worker/scripts/sweep_conf.py /path/to/clip.mp4
```

## Run ball-track worker (local)

```bash
python worker/main.py --video /path/to/clip.mp4 --output worker/output/ball_track.json
```

Options: `--sample-fps 10`, `--conf 0.20`, `--game-id`, `--job-id`

## Annotated video (ball + hoop boxes)

```bash
python worker/scripts/draw_ball_hoop_boxes.py /path/to/clip.mp4 --output worker/output/ball_hoop_boxes.mp4
```

Draws only `basketball` (green) and `hoop` (orange) — no player or referee labels. Options: `--conf 0.20`, `--max-width 1280`.

## Detect shot attempts

Hoop location from **YOLO `hoop` detections**. Full-court clips auto-cluster into **two basket locations**; each frame uses the **active shooting hoop** (detected hoop, or nearest basket to the ball). Half-court clips use a single cluster. Attempts trigger on rim crossing with lookback, plus a rim approach path for underbasket layups.

```bash
python worker/scripts/detect_attempts.py /path/to/clip.mp4 \
  --output worker/output/attempts.json \
  --visualize worker/output/attempts.mp4
```

Options: `--ball-conf 0.05`, `--hoop-conf 0.60`, `--sample-fps 20`, `--cooldown 1.0`, `--device mps`, `--hoop-roi worker/hoop_roi.json`.

## Detect wheelchair full-court attempts (left-camera)

Standalone pipeline for wheelchair full-court games with camera positioned on the left. It keeps the same attempt rules and output schema as `detect_attempts.py`, but uses an independent detector module so it can evolve separately.

```bash
python worker/scripts/detect_attempts_wheelchair.py /path/to/clip.mp4 \
  --output worker/output/attempts_wheelchair.json \
  --visualize worker/output/attempts_wheelchair.mp4
```

Primary options are the same as `detect_attempts.py`: `--ball-conf`, `--hoop-conf`, `--sample-fps`, `--cooldown`, `--model`, `--ball-model`, `--device`, `--hoop-roi`, `--max-width`.

### Ball-only attempt video (flash banner)

Separate from `--visualize` on `detect_attempts.py` — green ball boxes only, **ATTEMPT DETECTED** flashes for 0.75s:

```bash
python worker/scripts/visualize_attempts_ball.py clip_60s.mp4 \
  --ball-model worker/models/ball_finetuned_v3.pt \
  --hoop-roi worker/hoop_roi.json \
  --sample-fps 25 \
  --output worker/output/attempts_ball_only.mp4 \
  --device mps
```

From existing JSON:

```bash
python worker/scripts/visualize_attempts_ball.py clip_60s.mp4 \
  --attempts worker/output/attempts.json \
  --ball-model worker/models/ball_finetuned_v3.pt \
  --output worker/output/attempts_ball_only.mp4
```

Options: `--flash-sec 0.75`, `--ball-conf`, `--max-width 1280`.

Mark hoops and/or a **court polygon** in [`hoop-marker.html`](hoop-marker.html). Export `hoop_roi.json`:

- **Hoops only** — manual basket locations (`leftHoop` / `rightHoop` / `courtHoops`).
- **Court polygon only** — YOLO detects hoops; `courtPolygon` filters ball detections.
- **Both** — manual hoops plus court boundary.

Ball tracking uses **Kalman** per-frame E-BARD detection with hoop-region crop fallback when full-frame ball detection misses.

## Train fine-tuned ball model (hybrid with E-BARD)

Roboflow export (or frames from `export_labeled_frames.py`) can be fine-tuned for ball detection while E-BARD continues to detect hoops.

```bash
cd yolo && source .venv/bin/activate

# 1. Prepare dataset (bbox labels, train/val split, data.local.yaml)
python worker/scripts/prepare_ball_dataset.py

# 2. Train (requires BODD weights from download_model.py)
python worker/scripts/train_ball_model.py \
  --data worker/models/ball_dataset_prepared/data.local.yaml

# 3. Run attempts with hybrid detectors
python worker/scripts/detect_attempts.py /path/to/clip.mp4 \
  --hoop-roi worker/hoop_roi.json \
  --ball-model worker/models/ball_finetuned_v1.pt \
  --device mps
```

Options:

- `prepare_ball_dataset.py --source` — Roboflow folder (default: `worker/models/Finetune E-BARD.v1i.yolov8`)
- `prepare_ball_dataset.py --output` — prepared copy (default: `worker/models/ball_dataset_prepared`)
- `train_ball_model.py --epochs`, `--imgsz`, `--device`, `--pretrained`
- `detect_attempts.py --ball-model` — fine-tuned ball weights; `--model` stays E-BARD for hoop
- `draw_ball_hoop_boxes.py --ball-model` — visualize hybrid detections
- `visualize_attempts_ball.py` — ball-only boxes + **ATTEMPT DETECTED** flash (does not use `--visualize` on `detect_attempts.py`)

Training artifacts land in `worker/models/runs/ball_finetune/`; the best checkpoint is copied to `worker/models/ball_finetuned_v1.pt`.

## Train court pose model

Roboflow **Court Detection V3** export is a YOLOv8 pose dataset (class `court` + 4 corner keypoints). Prepare a train/val split, then fine-tune `yolov8n-pose.pt`.

```bash
cd yolo && source .venv/bin/activate

python worker/scripts/prepare_court_dataset.py
python worker/scripts/train_court_model.py \
  --data worker/models/court_dataset_prepared/data.local.yaml
```

Options:

- `prepare_court_dataset.py --source` — Roboflow folder (default: `Court Detection V3.yolov8`)
- `prepare_court_dataset.py --output` — prepared copy (default: `worker/models/court_dataset_prepared`)
- `train_court_model.py --epochs`, `--imgsz`, `--batch`, `--device`, `--pretrained`

Ultralytics downloads `yolov8n-pose.pt` if it is not already cached. Training artifacts land in `worker/models/runs/court_pose/`; the best checkpoint is copied to `worker/models/court_pose_v1.pt`.

### Detect court (pose overlay + JSON)

Requires `worker/models/court_pose_v1.pt` from training above.

```bash
cd yolo && source .venv/bin/activate
python worker/scripts/detect_court.py new.mp4 \
  --output worker/output/court_detect.mp4 \
  --json worker/output/court_detect.json \
  --preset fibaHalf \
  --device mps
```

Detects the paint (D) keypoints, snaps them to floor-line edges, and overlays boundary, paint, 3pt/2pt arc, and rim. Low-confidence corners reuse the last good homography so the tilt does not drop out. Keypoints stay numbered 0–3.

Keypoint order defaults to **0,1,2,3** = bottom-left, bottom-right, top-right, top-left (baseline pair then free-throw pair). If the overlay is mirrored, try `--keypoint-order 3,2,1,0` or `--keypoint-order auto`.

Feed that JSON into player tracking so `player_movement.json` court meters follow zoom (static `court_calibration.json` does not):

```bash
python worker/scripts/track_players.py new.mp4 \
  --court-detect worker/output/court_detect.json \
  --kit-book kit_book.json \
  --device mps
```

Use the same `--max-width` as `detect_court.py` (default 1280). `--court-detect` takes precedence over `--calibration`.

**Full pipeline documentation:** [docs/PLAYER_MOVEMENT.md](docs/PLAYER_MOVEMENT.md) — architecture, per-point algorithm, homography smoothing, config reference, and troubleshooting.

**Models & design decisions:** [docs/MODELS_AND_DESIGN.md](docs/MODELS_AND_DESIGN.md) — every model, training path, and architectural rationale.

**Fine-tuning (ball & court):** [docs/FINETUNING.md](docs/FINETUNING.md) — dataset requirements, prepare/train commands, verification.

Options: `--model`, `--conf` (default 0.50), `--imgsz`, `--max-width`, `--preset` (`fibaHalf`, `nbaHalf`, `fiba3x3`, `fiba`, `nba`), `--keypoint-order`.

## Model

Weights are **not** committed (see `.gitignore`). Download them after setup.

### Base model (required)

[E-BARD YOLOv8n](https://huggingface.co/GabrieleGiudici/E-BARD-detection-models) (`BODD_yolov8n_0001.pt`), CC-BY-4.0. Used for hoop (and default ball) detection.

```bash
cd yolo && source .venv/bin/activate
python worker/scripts/download_model.py
```

This writes `worker/models/BODD_yolov8n_0001.pt` (~6 MB). Needs network access and `huggingface_hub` (already in `requirements.txt`). Optional:

```bash
python worker/scripts/download_model.py --dest /path/to/dir
```

If download fails (auth / rate limit), open the [Hugging Face repo](https://huggingface.co/GabrieleGiudici/E-BARD-detection-models), download `BODD_yolov8n_0001.pt` manually, and place it at `worker/models/BODD_yolov8n_0001.pt`.

### Fine-tuned ball model (optional)

Hybrid pipelines (`--ball-model`) expect a local checkpoint such as `worker/models/ball_finetuned_v3.pt`. That file is produced by training (see [Train fine-tuned ball model](#train-fine-tuned-ball-model-hybrid-with-e-bard)); it is not downloaded by `download_model.py`. Without it, omit `--ball-model` and the base E-BARD weights handle both ball and hoop.

## Layout

```
yolo/
  requirements.txt
  README.md
  docs/
    PLAYER_MOVEMENT.md      # player tracking pipeline (comprehensive)
    MODELS_AND_DESIGN.md    # models, training, design decisions
    FINETUNING.md           # ball & court fine-tuning guide
  .gitignore
  hoop-marker.html
  court-marker.html
  play-renderer.html
  kit_book.json
  worker/
    main.py           # worker entrypoint
    detector.py
    tracker.py
    attempt_detector.py
    process_job.py
    player_tracker.py
    player_pose.py
    court_projection.py
    court_overlay.py
    team_classifier.py
    jersey_ocr.py
    player_config.py
    hoop_roi.json
    models/           # weights downloaded / trained locally (.gitkeep only in git)
    scripts/
      download_model.py
      prepare_court_dataset.py
      train_court_model.py
      detect_court.py
      track_players.py
    output/           # run artifacts (gitignored; .gitkeep kept)
```
