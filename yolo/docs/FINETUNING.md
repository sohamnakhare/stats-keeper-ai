# Fine-Tuning Guide: Ball & Court Models

Step-by-step guide for training the two locally fine-tuned models in this repo. Both use **Ultralytics YOLOv8** with Roboflow-exported datasets.

| Model | Task | Base weights | Output | Required for |
|-------|------|--------------|--------|--------------|
| **Ball fine-tune** | Basketball bbox detection | E-BARD `BODD_yolov8n_0001.pt` | `ball_finetuned_v*.pt` | Shot attempts (optional — E-BARD works as fallback) |
| **Court pose v1** | Paint (D) 4-corner keypoints | `yolov8n-pose.pt` | `court_pose_v1.pt` | Zoom-aware court + player movement |

**Not covered here** (no fine-tuning required):

- **E-BARD** — download only (`download_model.py`)
- **Player detection** — uses E-BARD `player` class as-is
- **YOLOv8n-pose (COCO)** — Ultralytics pretrained, used for player ankles/torso

See also: [MODELS_AND_DESIGN.md](MODELS_AND_DESIGN.md) for why these models exist.

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Ball fine-tuning](#ball-fine-tuning)
3. [Court pose fine-tuning](#court-pose-fine-tuning)
4. [After training](#after-training)
5. [When to retrain](#when-to-retrain)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Environment

```bash
cd yolo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### GPU

Training defaults to **`mps`** on macOS and **`cuda:0`** elsewhere. Override with `--device`.

CPU training is not supported by the worker inference path and is impractically slow for YOLO fine-tuning.

### Base weights

**Ball** — E-BARD must exist before training:

```bash
python worker/scripts/download_model.py
# → worker/models/BODD_yolov8n_0001.pt
```

**Court** — Ultralytics auto-downloads `yolov8n-pose.pt` on first train if not cached locally.

### Roboflow exports (local, gitignored)

| Dataset | Default path | Export format |
|---------|--------------|---------------|
| Ball | `worker/models/Finetune E-BARD.v1i.yolov8/` | YOLOv8 **detection** |
| Court | `Court Detection V3.yolov8/` (under `yolo/`) | YOLOv8 **pose** |

Download from Roboflow Universe and unzip into the paths above, or pass `--source` to the prepare scripts.

---

## Ball fine-tuning

### Purpose

Improve **basketball bbox recall** on small/blurred broadcast balls. Hoop detection stays on E-BARD — runtime uses a **hybrid detector** (`HybridBallHoopDetector`):

- Fine-tuned weights → `basketball`
- E-BARD → `hoop`

### Dataset requirements

| Item | Specification |
|------|---------------|
| Format | YOLOv8 object detection |
| Classes | **1 class** — `basketball` (class id `0`) |
| Label line | `0 cx cy w h` (normalized 0–1) |
| Images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` |
| Pairing | One `.txt` label per image, same stem |
| Negatives | **Empty `.txt` files allowed** — kept as hard negatives |
| Layout | Roboflow export: `train/images`, `train/labels`, optional `valid/` |

**Invalid labels are skipped** (malformed lines, wrong class id, zero-size boxes).

### Label example

```
0 0.512000 0.384000 0.024000 0.024000
```

### Step 1 — Prepare dataset

```bash
cd yolo && source .venv/bin/activate

python worker/scripts/prepare_ball_dataset.py \
  --source "worker/models/Finetune E-BARD.v1i.yolov8" \
  --output worker/models/ball_dataset_prepared \
  --val-fraction 0.2 \
  --seed 42
```

**What it does:**

- Validates and normalizes label files
- If Roboflow included `valid/` or `val/`, copies splits as-is
- Otherwise splits `train/` 80/20 into train/valid
- Writes `worker/models/ball_dataset_prepared/data.local.yaml`:

```yaml
path: /absolute/path/to/ball_dataset_prepared
train: train/images
val: valid/images
nc: 1
names: ['basketball']
```

### Step 2 — Train

```bash
python worker/scripts/train_ball_model.py \
  --data worker/models/ball_dataset_prepared/data.local.yaml \
  --pretrained worker/models/BODD_yolov8n_0001.pt \
  --output worker/models/ball_finetuned_v1.pt \
  --device mps
```

### Default hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `epochs` | 80 | |
| `imgsz` | 512 | Smaller than court — ball is tiny |
| `batch` | 4 | Reduce if OOM |
| `patience` | 20 | Early stopping |
| `pretrained` | E-BARD | **Do not** init from COCO for hybrid compatibility |

### Artifacts

| Path | Contents |
|------|----------|
| `worker/models/runs/ball_finetune/` | Ultralytics run (logs, plots, checkpoints) |
| `worker/models/runs/ball_finetune/weights/best.pt` | Best epoch |
| `worker/models/ball_finetuned_v1.pt` | **Stable copy** used by scripts |

Bump version in `--output` (e.g. `ball_finetuned_v2.pt`) when iterating.

### Step 3 — Use in pipelines

```bash
# Shot attempts (hybrid)
python worker/scripts/detect_attempts.py clip.mp4 \
  --ball-model worker/models/ball_finetuned_v1.pt \
  --hoop-roi worker/hoop_roi.json \
  --device mps

# Ball-only visualization
python worker/scripts/draw_ball_hoop_boxes.py clip.mp4 \
  --ball-model worker/models/ball_finetuned_v1.pt

# Ball-only attempt flash video
python worker/scripts/visualize_attempts_ball.py clip.mp4 \
  --ball-model worker/models/ball_finetuned_v3.pt \
  --device mps
```

Omit `--ball-model` to use E-BARD for both ball and hoop.

### Building your own ball dataset

Alternative to Roboflow: pseudo-label from E-BARD on your clips:

```bash
python worker/scripts/export_labeled_frames.py clip.mp4 \
  --output worker/models/my_ball_export \
  --sample-fps 10 \
  --ball-conf 0.15 \
  --export-negatives
```

Review/correct labels, then point `prepare_ball_dataset.py --source` at your export folder.

---

## Court pose fine-tuning

### Purpose

Detect the **paint (D) quadrilateral** — four floor corners — so homography can map player feet to court meters. Required for:

- `detect_court.py` — overlay + `court_detect.json`
- `track_players.py --court-detect` — zoom-aware player movement

### Dataset requirements

| Item | Specification |
|------|---------------|
| Format | YOLOv8 **pose** (keypoint) |
| Classes | **1 class** — `court` (class id `0`) |
| Keypoints | **4 corners** of the paint quad |
| Label fields | 17 total: `class cx cy w h` + 4 × (`kx ky v`) |
| Visibility `v` | `0` = absent, `1` = occluded, `2` = visible |
| Images | Same suffix rules as ball |
| Negatives | Empty `.txt` allowed |
| Default source | `Court Detection V3.yolov8/` (Roboflow, CC BY 4.0) |

### Label example

```
0 0.500000 0.450000 0.300000 0.250000 0.350000 0.550000 2 0.650000 0.550000 2 0.650000 0.350000 2 0.350000 0.350000 2
```

Fields: bbox center/size, then four keypoints `(x, y, visibility)`.

### Keypoint semantics (after training)

At inference, keypoints map to paint corners (see `court_overlay.py`):

| Index | Corner | Role |
|-------|--------|------|
| 0 | Bottom-left | Baseline |
| 1 | Bottom-right | Baseline |
| 2 | Top-right | Free-throw line |
| 3 | Top-left | Free-throw line |

If overlay is mirrored, fix at inference with `--keypoint-order 3,2,1,0` or `auto` — no retrain needed.

### Step 1 — Prepare dataset

```bash
cd yolo && source .venv/bin/activate

python worker/scripts/prepare_court_dataset.py \
  --source "Court Detection V3.yolov8" \
  --output worker/models/court_dataset_prepared \
  --val-fraction 0.2 \
  --seed 42
```

**What it does:**

- Validates pose label format (exactly 17 fields, class 0, valid visibility flags)
- Same train/valid split logic as ball
- Writes `data.local.yaml` with pose metadata:

```yaml
path: /absolute/path/to/court_dataset_prepared
train: train/images
val: valid/images
kpt_shape: [4, 3]
flip_idx: [1, 0, 3, 2]
nc: 1
names: ['court']
```

`flip_idx` tells Ultralytics how keypoints swap under horizontal flip augmentation.

### Step 2 — Train

```bash
python worker/scripts/train_court_model.py \
  --data worker/models/court_dataset_prepared/data.local.yaml \
  --pretrained yolov8n-pose.pt \
  --output worker/models/court_pose_v1.pt \
  --device mps
```

### Default hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `epochs` | 100 | Pose often needs more than detection |
| `imgsz` | 640 | Match `detect_court.py` default |
| `batch` | 8 | Reduce if OOM |
| `patience` | 20 | Early stopping |
| `pretrained` | `yolov8n-pose.pt` | COCO pose init (not E-BARD) |

### Artifacts

| Path | Contents |
|------|----------|
| `worker/models/runs/court_pose/` | Ultralytics run |
| `worker/models/runs/court_pose/weights/best.pt` | Best epoch |
| `worker/models/court_pose_v1.pt` | **Stable copy** — default for `detect_court.py` |

### Step 3 — Use in pipelines

```bash
# Court detect + overlay JSON
python worker/scripts/detect_court.py new.mp4 \
  --json worker/output/court_detect.json \
  --output worker/output/court_detect.mp4 \
  --preset fibaHalf \
  --model worker/models/court_pose_v1.pt \
  --device mps

# Player movement (consumes court_detect.json)
python worker/scripts/track_players.py new.mp4 \
  --court-detect worker/output/court_detect.json \
  --kit-book kit_book.json \
  --device mps
```

**Use the same `--max-width`** (default 1280) in court detect and player tracking.

---

## After training

### Verify ball model

```bash
python worker/scripts/draw_ball_hoop_boxes.py test_clip.mp4 \
  --ball-model worker/models/ball_finetuned_v1.pt \
  --output worker/output/ball_check.mp4
```

Compare detection rate vs E-BARD-only (no `--ball-model`).

### Verify court model

```bash
python worker/scripts/detect_court.py test_clip.mp4 \
  --model worker/models/court_pose_v1.pt \
  --output worker/output/court_check.mp4 \
  --json worker/output/court_check.json \
  --preset fibaHalf
```

Check overlay alignment: paint, 3pt arc, and rim should track the floor lines. Try `--keypoint-order auto` if corners look swapped.

### Evaluate E-BARD baseline (no fine-tune)

```bash
python worker/scripts/eval_ebard.py test_clip.mp4
python worker/scripts/sweep_conf.py test_clip.mp4
```

---

## When to retrain

### Ball

| Signal | Action |
|--------|--------|
| Frequent ball misses in attempt pipeline | Fine-tune or add frames from your broadcast style |
| New camera angle / resolution | Export frames from that footage, merge into dataset |
| False positives on lights / logos | Add hard negatives (empty labels) |

### Court

| Signal | Action |
|--------|--------|
| `detect_court.py` overlay misaligned | Check `--preset` and `--keypoint-order` first |
| Persistent misalignment on your venue | Add labeled frames from your clips to dataset |
| New court type (3×3, NBA vs FIBA) | Preset handles geometry; retrain only if paint detection fails |
| Player movement pulses despite smoothing | Usually homography tuning, not model — see [PLAYER_MOVEMENT.md](PLAYER_MOVEMENT.md) |

---

## Troubleshooting

### `Source dataset not found`

Roboflow export not unzipped. Place it at the default path or pass `--source`.

### `Pretrained weights not found` (ball)

Run `python worker/scripts/download_model.py` first.

### `Training finished but best.pt missing`

Training crashed or was interrupted. Check `worker/models/runs/*/`. Reduce `--batch` or `--imgsz`.

### OOM (out of memory)

```bash
# Ball
python worker/scripts/train_ball_model.py --batch 2 --imgsz 416

# Court
python worker/scripts/train_court_model.py --batch 4 --imgsz 512
```

### Many `Skipped (invalid labels)`

Open a skipped label file — must match exact field counts (5 for ball, 17 for court). Class id must be `0`.

### Court overlay mirrored / rotated

Inference fix, not retrain:

```bash
python worker/scripts/detect_court.py clip.mp4 \
  --keypoint-order auto \
  --preset fibaHalf \
  ...
```

### Ball fine-tune hurts hoop pipeline

Ensure `--pretrained` is E-BARD, not COCO. Hoop always uses separate E-BARD weights in hybrid mode.

---

## Quick reference commands

```bash
# ── Ball ──────────────────────────────────────────
python worker/scripts/prepare_ball_dataset.py
python worker/scripts/train_ball_model.py --device mps

# ── Court ─────────────────────────────────────────
python worker/scripts/prepare_court_dataset.py
python worker/scripts/train_court_model.py --device mps

# ── Download base E-BARD (ball init + hoop) ───────
python worker/scripts/download_model.py
```

---

## Related documentation

| Document | Contents |
|----------|----------|
| [MODELS_AND_DESIGN.md](MODELS_AND_DESIGN.md) | Why hybrid ball, court pose vs bbox, full model inventory |
| [PLAYER_MOVEMENT.md](PLAYER_MOVEMENT.md) | Using `court_detect.json` in tracking |
| [README.md](../README.md) | Project setup and CLI overview |
