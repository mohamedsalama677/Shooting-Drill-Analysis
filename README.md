# ScoutAI – Shooting Drill Analysis

Automated computer-vision pipeline that processes a football shooting-drill video and produces an annotated output video plus a JSON report covering shot count, foot used, shot power, scoring zone, gate errors, and missed distances.

---

## Features & Techniques

### Feature 1 · Distance Calibration

**Goal:** use the 4 traffic cones as real-world reference points to map pixel distances → meters, verifying the gate is ≥12 m from the goal line.

| Step | Code | Technique |
|---|---|---|
| Cone detection | [`detection/cone_detector.py` — `ConeDetector`](detection/cone_detector.py) | **YOLO-World** (`yolov8s-worldv2.pt`) zero-shot with text prompts `["traffic cone", "orange cone", "sports cone"]`. No training required. |
| De-duplicate overlapping detections | [`cone_detector.py:_dedupe_close`](detection/cone_detector.py) | Min-distance filter (40 px default) on raw centroids. |
| Pick best 4 from many candidates | [`cone_detector.py:_select_best_quad`](detection/cone_detector.py) | Brute-force `itertools.combinations` over the top-20 detections, scored by a *rectangleness* metric (side-length variance + diagonal equality). |
| Full homography calibration | [`calibration/homography.py`](calibration/homography.py) | `cv2.getPerspectiveTransform` on the 4 ordered cone centroids → a 3×3 **H matrix** mapping image plane → ground plane (meters). Gives accurate distance anywhere in the field of view. |
| 2-point fallback calibration | [`calibration/scale.py`](calibration/scale.py) | When only 2 cones are visible or the quad is degenerate, the known gate width (2 m) is divided by the pixel distance between the front pair → a **px_per_meter** scalar. |
| Per-frame cone refresh | [`main.py`](main.py) | Every 30 frames the cone detector reruns so a panning camera doesn't break the calibration. |

**Key settings:** `GATE_WIDTH_M`, `GATE_DEPTH_M`, `DIST_CONES_TO_GOAL_M`, `CONE_YOLO_CONF_THRESHOLD`, `CONE_DEDUP_MIN_DIST_PX`, `CALIBRATION_MAX_FRAMES`.

---

### Feature 2 · Shot Detection

**Goal:** detect when the ball is kicked by identifying a sudden velocity spike in the tracked trajectory.

| Step | Code | Technique |
|---|---|---|
| Ball & player detection | [`detection/detector.py`](detection/detector.py) | **YOLOv8n** (`yolov8n.pt`) on COCO classes 0 (person) and 32 (sports ball). Low confidence threshold (0.15) to catch small far balls. |
| Multi-ball tracking | [`detection/tracker.py` — `MultiBallTracker`](detection/tracker.py) | **BoT-SORT** (`botsort.yaml`) via Ultralytics `model.track()`. Maintains persistent track IDs across occlusions. |
| Short-gap recovery | [`main.py`](main.py) | When YOLO misses the ball for ≤10 frames (motion blur during kick), the tracker searches within an expanding radius (220 px base + 80 px/frame) for the nearest re-detection. |
| Velocity hysteresis | [`analysis/event_detector.py`](analysis/event_detector.py) | Speed rises above `SHOT_VELOCITY_HIGH_MPS` (2.0 m/s) to arm a candidate, then must stay above `SHOT_VELOCITY_LOW_MPS` (1.0 m/s) for the lookback window. Rejects slow dribble touches. |
| Multi-criteria validation | [`event_detector.py:detect_shots_from_tracks`](analysis/event_detector.py) | Each candidate must also pass: ball–player proximity at contact (≤120 px), minimum goalward travel (≥130 px), direction vector alignment with goal axis, and minimum post-kick displacement. |
| Shot de-duplication | [`event_detector.py`](analysis/event_detector.py) | Any two candidates within 24 frames of each other collapse to the stronger one. |

**Key settings:** `SHOT_VELOCITY_HIGH_MPS`, `SHOT_VELOCITY_LOW_MPS`, `SHOT_LOOKBACK_FRAMES`, `SHOT_LOOKAHEAD_FRAMES`, `SHOT_CONTACT_MAX_DIST_PX`, `SHOT_MIN_GOALWARD_PROGRESS_PX`, `SHOT_DEDUPE_WINDOW_FRAMES`.

---

### Feature 3 · Foot Used

**Goal:** identify which foot (left / right) was used for each kick.

| Step | Code | Technique |
|---|---|---|
| Pose estimation | [`detection/pose_estimator.py`](detection/pose_estimator.py) | **MediaPipe Pose** (`min_detection_confidence=0.5`, `min_tracking_confidence=0.5`). Lighter than YOLOv8-Pose and easier to deploy. |
| Ankle extraction | [`pose_estimator.py:get_ankles`](detection/pose_estimator.py) | Returns `{left: (x,y), right: (x,y)}` for landmarks 27 (left ankle) and 28 (right ankle). |
| Foot decision | [`analysis/shot_analyzer.py:determine_foot`](analysis/shot_analyzer.py) | Euclidean distance from each ankle to the ball center at the last contact frame. Closer ankle = kicking foot. |
| Two-pass architecture | [`main.py`](main.py) | Pass 1 writes the annotated video without pose (fast). Pass 2 re-seeks to each shot frame and runs MediaPipe only on those frames — avoids the overhead of running pose on every frame. |

**Key settings:** `POSE_MIN_DETECTION_CONFIDENCE`, `POSE_MIN_TRACKING_CONFIDENCE`, `LEFT_ANKLE_LANDMARK` (27), `RIGHT_ANKLE_LANDMARK` (28).

---

### Feature 4 · Goal Detection & Scoring Zone

**Goal:** detect the goal mouth, divide it into a 3×2 grid, and record which zone the ball entered.

```
┌──────────────────────────┐
│  TL(10) │  TC(7) │ TR(10)│
├─────────┼────────┼───────┤
│  BL(5)  │  BC(3) │ BR(5) │
└──────────────────────────┘
```

| Step | Code | Technique |
|---|---|---|
| Goalpost detection | [`detection/cone_detector.py` — `GoalDetector`](detection/cone_detector.py) | **Roboflow hosted inference API** (`detect.roboflow.com/goalpost-u6e0h/1`). Frame encoded as base64 JPEG and POSTed; API key read from `ROBO_API` env var. Sampled at 4 frames per run to minimise API calls. |
| Detection classification | [`cone_detector.py:_classify_predictions`](detection/cone_detector.py) | Splits predictions by shape: tall-narrow (aspect ≥1.4, height ≥80 px) → individual posts; wide-short (aspect ≤0.4, width ≥80 px) or catch-all → whole-goal bbox. Bboxes covering >65% of the frame are dropped as false positives. |
| Goal polygon assembly | [`cone_detector.py:_build_goal_polygon`](detection/cone_detector.py) | When two posts are returned, builds a perspective-correct trapezoid from their inner edges. Falls back to the bbox rectangle for whole-goal detections. |
| Scoring grid | [`analysis/shot_analyzer.py:analyze_shots`](analysis/shot_analyzer.py) | Goal bbox divided into 3 columns × 2 rows. Each cell has a name and a point value defined in `GOAL_ZONE_POINTS`. |
| Ball-into-goal detection | [`analysis/shot_analyzer.py:_check_goal_crossing`](analysis/shot_analyzer.py) | Looks ahead 90 frames from each shot. Ball centroid tested with `cv2.pointPolygonTest` against the goal polygon. |
| Trajectory extrapolation | [`analysis/shot_analyzer.py:_extrapolate_into_goal`](analysis/shot_analyzer.py) | When YOLO loses the ball before it reaches the goal, the release velocity vector is extrapolated forward in 1-px steps until it intersects the goal polygon. |
| Manual bbox override | [`config/settings.py`](config/settings.py) | Set `GOAL_MANUAL_BBOX = (x1, y1, x2, y2)` to bypass the API entirely for a fixed-camera setup. |

**Key settings:** `ROBOFLOW_GOAL_MODEL_ID`, `ROBOFLOW_GOAL_CONFIDENCE`, `ROBOFLOW_GOAL_MAX_FRAME_FRAC`, `GOAL_ZONE_POINTS`, `GOAL_LOOKAHEAD_FRAMES`, `GOAL_MANUAL_BBOX`, `ENABLE_GOAL_FEATURES`.

---

### Feature 5 · Shot From Outside Gate (Error Flag)

**Goal:** flag shots where the ball was still on the player's side of the cone gate at the moment of the kick.

| Step | Code | Technique |
|---|---|---|
| Gate error check | [`analysis/shot_analyzer.py:check_outside_gate`](analysis/shot_analyzer.py) | Ball pixel position at shot frame transformed to world coordinates. |
| Homography path | [`calibration/homography.py:pixel_to_meters`](calibration/homography.py) | `cv2.perspectiveTransform` with the H matrix; world Y > `GATE_DEPTH_M` means the ball is behind the gate back-line. |
| Scale fallback path | [`calibration/scale.py`](calibration/scale.py) | When only a `px_per_meter` scalar is available, pixel distance from ball to the nearest cone front-line is converted to meters. |
| Error reported | [`output/report_generator.py`](output/report_generator.py) | `outside_gate: true/false` appears per shot in `report.json`. |

---

### Feature 6 · Missed Shot Distance

**Goal:** when the ball doesn't score, calculate how far it missed the goal by.

| Step | Code | Technique |
|---|---|---|
| Post-shot tracking | [`analysis/shot_analyzer.py`](analysis/shot_analyzer.py) | Ball trajectory recorded for up to 90 frames after the shot. Short gaps (≤6 missing frames) are tolerated. |
| Exit point | [`analysis/shot_analyzer.py`](analysis/shot_analyzer.py) | Last reliably tracked ball position before tracking dies is used as the final position. |
| Goal geometry fallback | [`analysis/shot_analyzer.py`](analysis/shot_analyzer.py) | When no goal bbox was detected, the goal center is estimated from the calibration using `DIST_CONES_TO_GOAL_M` (12 m). |
| Distance calculation | [`analysis/shot_analyzer.py`](analysis/shot_analyzer.py) | Pixel distance from final ball position to nearest goal edge converted to meters via calibration. Reported as `missed_distance_m`. |

**Key settings:** `GOAL_LOOKAHEAD_FRAMES` (90), `GOAL_MISS_TRACK_MAX_GAP` (6), `GOAL_MISS_GEOMETRY_FALLBACK`, `DIST_CONES_TO_GOAL_M`.

---

## Pipeline Flow

```
Raw Video
    │
    ▼
Pass 1 — Frame-by-Frame Processing
  ├── YOLO-World → cone detection → homography calibration (or 2-point scale)
  ├── YOLOv8n + BoT-SORT → ball & player tracking
  ├── Velocity hysteresis → shot candidate detection
  ├── Roboflow API → goal bbox (sampled at frames 0, 15, 30, 45)
  └── OpenCV VideoWriter → annotated video
    │
    ▼
Pass 2 — Per-Shot Re-Analysis
  ├── Re-seek to each shot frame
  ├── MediaPipe Pose → ankle keypoints → foot determination
  ├── Homography → gate error flag
  ├── Ball trajectory look-ahead → goal crossing + scoring zone
  └── Missed distance calculation
    │
    ▼
Output
  ├── data/output/annotated_<name>.mp4
  └── data/output/report.json
```

---

## Output

### Annotated Video (`data/output/annotated_<name>.mp4`)

| Overlay | When drawn |
|---|---|
| Green box (player) + orange marker (ball) + magenta trail | Every frame |
| Red boxes (cones) + yellow gate outline | Every frame |
| Calibration badge (top-left): method + px/m | Every frame |
| Ball velocity readout | Every frame |
| Shot banner: `SHOT #N  LEFT FOOT  7.2 m/s  GOAL! Zone TR (+10 pts)` | During shot banner duration (30 frames) |
| Green scoring zone grid with highlighted cell | During shot banner duration |
| Right-side status panel: `GOAL  Shot #2  Zone TR  +10` | During shot banner duration |

### Report (`data/output/report.json`)

```json
{
  "drill_valid": true,
  "total_shots": 3,
  "shots": [
    {
      "index": 1,
      "frame": 145,
      "foot": "left",
      "velocity_mps": 8.3,
      "outside_gate": false,
      "scored": true,
      "scoring_zone": "TR",
      "zone_points": 10,
      "missed_distance_m": null
    }
  ],
  "total_points": 18,
  "drill_errors": []
}
```

---

## Project Structure

```
Shooting-Drill-Analysis/
├── main.py                   # Entry point — two-pass pipeline
├── config/
│   └── settings.py           # All thresholds, paths, and scoring constants
├── detection/
│   ├── detector.py           # YOLOv8n wrapper (ball + player)
│   ├── cone_detector.py      # YOLO-World cone detector + Roboflow goal detector
│   ├── tracker.py            # BoT-SORT multi-ball tracker
│   └── pose_estimator.py     # MediaPipe Pose ankle extraction
├── calibration/
│   ├── homography.py         # 4-cone H matrix + pixel↔meter transform
│   └── scale.py              # 2-point px_per_meter fallback
├── analysis/
│   ├── event_detector.py     # Velocity-hysteresis shot detection
│   ├── shot_analyzer.py      # Foot, power, zone, gate error, missed distance
│   └── drill_validator.py    # 3-shot sequence validation
├── output/
│   ├── annotator.py          # All drawing functions (trails, banners, grids)
│   ├── video_writer.py       # OpenCV VideoWriter wrapper
│   └── report_generator.py  # JSON report writer
├── utils/
│   ├── geometry.py           # euclidean, bbox_center, zone helpers
│   └── logger.py             # Logging setup
├── data/
│   ├── input/                # Drop your .mp4 / .MOV files here
│   └── output/               # Annotated video + report.json written here
├── models/                   # YOLO weights auto-downloaded on first run
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                      # ROBO_API=<your_roboflow_api_key>
```

---

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A Roboflow API key (free tier works — the goalpost model is public)

### 1 · Clone and configure

```bash
git clone <repo-url>
cd Shooting-Drill-Analysis
```

Create a `.env` file in the project root:

```
ROBO_API=your_roboflow_api_key_here
```

> Get your key at app.roboflow.com → Settings → API Keys.

### 2 · Add your video

Copy the drill video into `data/input/`:

```
data/input/drill.mp4
```

Supported formats: `.mp4`, `.MOV`, `.avi`, `.mkv` — any format OpenCV can decode.

### 3 · Build the image

```bash
docker compose build
```

First build takes ~5–10 minutes (downloads PyTorch, Ultralytics, MediaPipe). Subsequent builds are fully cached.

### 4 · Run the analysis

```bash
docker compose run --rm scout-drill-analysis --video "data/input/drill.mp4"
```

For a `.MOV` file with spaces in the name:

```bash
docker compose run --rm scout-drill-analysis --video "data/input/correct drill.MOV"
```

YOLO weights (`yolov8n.pt`, `yolov8s-worldv2.pt`) are downloaded automatically into `models/` on the first run and reused afterwards via the mounted volume.

### 5 · Collect the outputs

```
data/output/annotated_drill.mp4     ← annotated video
data/output/report.json             ← structured drill report
data/output/debug_calibration.png  ← cone detection debug frame
data/output/debug_goal.png          ← goal detection debug frame
```

---

## Configuration

All tunable constants live in [`config/settings.py`](config/settings.py). Common adjustments:

| Constant | Default | Purpose |
|---|---|---|
| `GATE_WIDTH_M` | `2.0` | Real-world gate width between cones (m) |
| `GATE_DEPTH_M` | `2.0` | Real-world gate depth front/back (m) |
| `DIST_CONES_TO_GOAL_M` | `12.0` | Required gate-to-goal distance (m) |
| `SHOT_VELOCITY_HIGH_MPS` | `2.0` | Speed threshold to arm a shot candidate (m/s) |
| `CONE_YOLO_CONF_THRESHOLD` | `0.05` | YOLO-World confidence for cone detection |
| `YOLO_CONF_THRESHOLD` | `0.15` | YOLOv8 confidence for ball/player detection |
| `ROBOFLOW_GOAL_CONFIDENCE` | `5` | Roboflow API confidence % for goal detection |
| `GOAL_MANUAL_BBOX` | `None` | Set to `(x1, y1, x2, y2)` to bypass Roboflow |
| `ENABLE_GOAL_FEATURES` | `True` | Set `False` to skip all goal detection |
| `DRILL_EXPECTED_SHOTS` | `3` | Number of shots expected in a valid drill |

### Disabling goal features (no API key)

In `config/settings.py`:

```python
ENABLE_GOAL_FEATURES = False
```

The pipeline will still detect shots, measure power, identify the foot used, and flag gate errors.

### Manual goal bbox (fixed camera)

If the camera doesn't move and you know the goal location in pixel coordinates:

```python
GOAL_MANUAL_BBOX = (x1, y1, x2, y2)
```

This bypasses the Roboflow API entirely.

---

## Technology Stack

| Component | Library / Model | Notes |
|---|---|---|
| Object detection | YOLOv8n (`yolov8n.pt`) | Ball + player, COCO pretrained |
| Zero-shot cone detection | YOLO-World (`yolov8s-worldv2.pt`) | Text-prompted, no fine-tuning |
| Ball / player tracking | BoT-SORT (built into Ultralytics) | Handles short occlusions |
| Pose estimation | MediaPipe Pose | Ankle landmarks for foot detection |
| Goal detection | Roboflow hosted API (`goalpost-u6e0h/1`) | REST API, free tier |
| Image processing | OpenCV (`cv2`) | All drawing, video I/O, homography |
| Perspective calibration | `cv2.getPerspectiveTransform` | 4-cone H matrix |
| HTTP client | `requests` | Roboflow API calls |
| Containerisation | Docker + Docker Compose v2 | GPU not required |

---

## Troubleshooting

**No cones detected / calibration fails**

Adjust `CONE_YOLO_CONF_THRESHOLD` (try `0.03`–`0.10`). The debug frame `data/output/debug_calibration.png` shows all detected cone candidates.

**No goal detected / `scored: null` in report**

Check your `ROBO_API` key in `.env`. Inspect `data/output/debug_goal.png`. For a fixed camera, set `GOAL_MANUAL_BBOX`.

**Shot not detected / count is 0**

Lower `SHOT_VELOCITY_HIGH_MPS` to `1.5` or `1.0` for a slower drill. Lower `YOLO_CONF_THRESHOLD` to `0.10` if the ball is small.

**Wrong foot reported**

The foot is determined at the last contact frame before the velocity spike. If the player is far from camera, try increasing `POSE_MIN_DETECTION_CONFIDENCE` slightly.
