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
