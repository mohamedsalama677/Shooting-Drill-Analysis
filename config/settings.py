"""Centralized configuration constants for the ScoutAI shooting drill pipeline.

Tuning happens here — no magic numbers should live in business logic.
"""

# ── Real-world geometry (meters) ──────────────────────────────────────────────
GATE_WIDTH_M = 2.0           # left↔right cone separation
GATE_DEPTH_M = 2.0           # front↔back cone separation
DIST_CONES_TO_GOAL_M = 12.0
DIST_BALLS_TO_CONES_M = 5.0

# ── Cone detection (YOLO-World, zero-shot) ────────────────────────────────────
# Open-vocabulary YOLO model — set_classes() with text prompts at runtime.
# Auto-downloaded on first run into the mounted models/ volume.
CONE_YOLO_MODEL = "yolov8s-worldv2.pt"
CONE_YOLO_PROMPTS = ["traffic cone", "orange cone", "sports cone"]
CONE_YOLO_CONF_THRESHOLD = 0.05  # zero-shot detections often have low confidence

# Spatial dedup — any two centroids closer than this are merged into one.
CONE_DEDUP_MIN_DIST_PX = 40

# Best-quad selection — score combinations of N candidates for "rectangleness".
CONE_TOP_N_CANDIDATES = 20

# Number of frames to scan looking for a frame with all 4 cones visible.
CALIBRATION_MAX_FRAMES = 60

# Debug artifact: annotated calibration frame written to OUTPUT_DIR.
DEBUG_CALIBRATION_FRAME_FILENAME = "debug_calibration.png"

# ── Shot detection (velocity hysteresis on calibrated ball speed) ─────────────
# Lowered to catch real shots when the small ball is only tracked for a short,
# choppy run. Direction/release checks below carry most of the precision.
SHOT_VELOCITY_HIGH_MPS = 2.0
SHOT_VELOCITY_LOW_MPS = 1.0
SHOT_MAX_REASONABLE_MPS = 35.0
SHOT_BANNER_DURATION_FRAMES = 30   # how long the "SHOT!" banner is held
SHOT_LOOKBACK_FRAMES = 8
SHOT_LOOKAHEAD_FRAMES = 18
SHOT_CONTACT_MAX_DIST_PX = 120.0
SHOT_MIN_POST_TRAVEL_PX = 115.0
SHOT_MIN_PLAYER_SEPARATION_PX = 90.0
SHOT_MIN_RELATIVE_PLAYER_SEPARATION_PX = 45.0
SHOT_GOAL_DIRECTION_X = -1.0
SHOT_GOAL_DIRECTION_Y = -0.15
SHOT_MIN_GOALWARD_PROGRESS_PX = 130.0
SHOT_MIN_GOALWARD_RATIO = 0.62
SHOT_MIN_RELATIVE_GOALWARD_PROGRESS_PX = 110.0
SHOT_MAX_GOALWARD_REGRESSION_PX = 35.0
SHOT_STRONG_RELEASE_MPS = 7.0
SHOT_MIN_INITIAL_GOALWARD_PROGRESS_PX = 20.0
SHOT_DEDUPE_WINDOW_FRAMES = 24
SHOT_GLOBAL_PROXIMITY_RADIUS_PX = 80.0
SHOT_DEBUG_MAX_CANDIDATES = 250

# Short-gap recovery tries to keep a kicked ball alive when YOLO misses a few
# blurred frames. It is conservative so it does not invent long-range tracks.
BALL_RECOVERY_MAX_GAP_FRAMES = 10
BALL_RECOVERY_SEARCH_RADIUS_PX = 220
BALL_RECOVERY_SEARCH_GROWTH_PX = 80
BALL_RECOVERY_MAX_VERTICAL_JUMP_PX = 95
BALL_RECOVERY_MIN_SCORE = 0.62

# Per-frame cone re-detection: rerun cone detection every N frames so a
# panning camera doesn't break the calibration.
CONE_RECAL_INTERVAL_FRAMES = 30

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"
COCO_CLASS_PERSON = 0
COCO_CLASS_BALL = 32
# Lowered for iter-4 — player gets missed at distance with the previous 0.25.
YOLO_CONF_THRESHOLD = 0.15
YOLO_BALL_FALLBACK_CONF_THRESHOLD = 0.10
YOLO_BALL_FALLBACK_MIN_DIST_PX = 36.0
YOLO_BALL_FALLBACK_MAX_TRACKED_BALLS = 1
YOLO_TRACKER = "botsort.yaml"

# ── MediaPipe Pose ────────────────────────────────────────────────────────────
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
LEFT_ANKLE_LANDMARK = 27
RIGHT_ANKLE_LANDMARK = 28

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
MODELS_DIR = "models"
REPORT_FILENAME = "report.json"
ANNOTATED_VIDEO_FILENAME = "annotated.mp4"

# ── Visualization ─────────────────────────────────────────────────────────────
BALL_TRAIL_LENGTH = 30   # frames of trail to render
COLOR_PERSON = (0, 255, 0)       # green
COLOR_BALL = (0, 165, 255)       # orange
COLOR_CONE = (0, 0, 255)         # red
COLOR_GATE = (255, 255, 0)       # cyan
COLOR_TRAIL = (255, 0, 255)      # magenta
COLOR_BANNER_BG = (0, 0, 0)
COLOR_BANNER_TEXT = (0, 255, 255)

# ── Goal detection (Roboflow hosted-inference goalpost model) ─────────────────
# Feature 4 (scoring zone) and Feature 6 (missed distance) rely on a real
# goal-mouth bbox. We hit Roboflow's hosted detect.roboflow.com endpoint with
# the trained "goalpost-u6e0h" model and assemble the bbox from the returned
# post / crossbar detections.
ENABLE_GOAL_FEATURES = True

# detect.roboflow.com/{model_id}/{version}?api_key=...
ROBOFLOW_GOAL_MODEL_ID = "goalpost-u6e0h"
ROBOFLOW_GOAL_MODEL_VERSION = 1   # only version 1 is publicly available
ROBOFLOW_GOAL_CONFIDENCE = 5      # percent (0-100); model is conservative on small far goals
ROBOFLOW_GOAL_OVERLAP = 40        # percent NMS overlap threshold
ROBOFLOW_GOAL_TIMEOUT_S = 15.0    # request timeout (seconds)
ROBOFLOW_API_ENV_VAR = "ROBO_API" # which env var holds the API key (see .env)
# Reject detections whose bbox covers more than this fraction of either frame
# dimension — the model occasionally returns a near-frame-sized false positive.
ROBOFLOW_GOAL_MAX_FRAME_FRAC = 0.65

# Shape filters applied to whatever class names the Roboflow model returns so
# we still get sane geometry even if the model labels everything "goal_post".
GOAL_POST_MIN_HEIGHT_PX = 80      # vertical post height
GOAL_POST_MIN_ASPECT = 1.4        # height / width
GOAL_CROSSBAR_MIN_WIDTH_PX = 80
GOAL_CROSSBAR_MAX_ASPECT = 0.4

# When the Roboflow model returns ≥2 post-like detections, the pair must
# satisfy these to be accepted as the actual goal mouth.
GOAL_MIN_POSTS_REQUIRED = 2
GOAL_POST_PAIR_MIN_WIDTH_PX = 80
GOAL_POST_PAIR_MAX_WIDTH_PX = 900
GOAL_POST_Y_OVERLAP_MIN = 0.4
GOAL_POST_MIN_Y_FRAC = 0.15       # post top below this fraction of frame height
GOAL_POST_MAX_Y_FRAC = 0.95       # post bottom above this fraction of frame height

# Manual override: set to (x1, y1, x2, y2) to short-circuit detection.
# Leave as None to rely on the Roboflow detector.
GOAL_MANUAL_BBOX = None

GOAL_DEBUG_IMAGE_FILENAME = "debug_goal.png"

# ── Scoring zone point values ─────────────────────────────────────────────────
# Grid: 3 x-columns (0=left, 1=center, 2=right) × 2 y-rows (0=top, 1=bottom)
GOAL_ZONE_POINTS = {
    (0, 0): 10, (1, 0): 7, (2, 0): 10,
    (0, 1): 5,  (1, 1): 3, (2, 1): 5,
}
GOAL_ZONE_NAMES = {
    (0, 0): "TL", (1, 0): "TC", (2, 0): "TR",
    (0, 1): "BL", (1, 1): "BC", (2, 1): "BR",
}

# ── Post-shot ball tracking for goal crossing / missed distance ───────────────
GOAL_LOOKAHEAD_FRAMES = 90          # 3 s at 30 fps — covers ball travel to goal
GOAL_BALL_PROXIMITY_RADIUS_PX = 120.0
GOAL_MISS_TRACK_MAX_GAP = 6         # max consecutive frames with no detection
GOAL_MISS_GEOMETRY_FALLBACK = True  # estimate goal center from calibration when no bbox

# ── Drill validation ──────────────────────────────────────────────────────────
DRILL_EXPECTED_SHOTS = 3
DRILL_MAX_INTERVAL_S = 15.0
