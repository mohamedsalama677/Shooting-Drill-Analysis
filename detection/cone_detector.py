"""Cone detection (YOLO-World) and goal-post detection (Roboflow hosted API).

Cone detection is zero-shot via YOLO-World ("traffic cone" prompt).
Goal detection calls Roboflow's hosted `goalpost-u6e0h` model and assembles
the goal mouth from the returned post / crossbar predictions.
"""

import base64
import itertools
import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLOWorld

from config import settings
from utils.geometry import bbox_center
from utils.logger import get_logger

log = get_logger(__name__)

Point = Tuple[int, int]


def _dedupe_close(points: List[Point], min_dist: int) -> List[Point]:
    """Drop any point that's within min_dist of an already-kept point."""
    kept: List[Point] = []
    for pt in points:
        if all(math.hypot(pt[0] - k[0], pt[1] - k[1]) >= min_dist for k in kept):
            kept.append(pt)
    return kept


def _rectangleness_score(quad: List[Point]) -> float:
    """How close 4 points are to forming a rectangle (lower = more rectangular)."""
    pts = np.array(quad, dtype=np.float32)
    s = pts[:, 0] + pts[:, 1]
    yx = pts[:, 1] - pts[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(yx)]
    bl = pts[np.argmax(yx)]

    def d(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

    top, bot = d(tl, tr), d(bl, br)
    left, right = d(tl, bl), d(tr, br)
    diag1, diag2 = d(tl, br), d(tr, bl)
    mean_side = (top + bot + left + right) / 4 + 1e-6
    return (abs(top - bot) + abs(left - right) + abs(diag1 - diag2)) / mean_side


def _select_best_quad(
    candidates: List[Point],
    top_n: int = settings.CONE_TOP_N_CANDIDATES,
) -> Optional[List[Point]]:
    """Pick the 4 candidates that best form a rectangle."""
    if len(candidates) < 4:
        return None
    if len(candidates) == 4:
        return candidates
    pool = candidates[:top_n]
    best, best_score = None, float("inf")
    for combo in itertools.combinations(pool, 4):
        score = _rectangleness_score(list(combo))
        if score < best_score:
            best, best_score = list(combo), score
    log.info(f"Best-quad rectangleness score: {best_score:.3f}")
    return best


def _build_goal_polygon(
    posts: List[Tuple[float, float, float, float, float]],
    crossbars: List[Tuple[float, float, float, float, float]],
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """Return a 4-corner polygon [TL, TR, BR, BL] tracing the goal mouth.

    Posts seen from a perspective angle have different y1/y2 — using the inner
    edges of the leftmost & rightmost post bboxes yields a trapezoid that
    matches the actual mouth, instead of an axis-aligned rectangle that
    over-/under-covers it.
    """
    bx1, by1, bx2, by2 = bbox
    if len(posts) >= 2:
        ordered = sorted(posts, key=lambda p: (p[0] + p[2]) / 2.0)
        left = ordered[0]
        right = ordered[-1]
        # Inner edges of each post define the mouth's left/right sides.
        lx = float(left[2])      # right edge of left post
        rx = float(right[0])     # left edge of right post
        l_top, l_bot = float(left[1]), float(left[3])
        r_top, r_bot = float(right[1]), float(right[3])
        if crossbars:
            cb = max(crossbars, key=lambda c: (c[2] - c[0]) * c[4])
            cb_y_bottom = float(cb[3])
            l_top = min(l_top, cb_y_bottom)
            r_top = min(r_top, cb_y_bottom)
        return np.array(
            [[lx, l_top], [rx, r_top], [rx, r_bot], [lx, l_bot]],
            dtype=np.float32,
        )
    # Single-post / no-post fallback: degenerate to bbox-shaped polygon.
    return np.array(
        [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]],
        dtype=np.float32,
    )


class ConeDetector:
    """YOLO-World cone detector. Set classes once; reuse across frames."""

    def __init__(
        self,
        model_path: str = settings.CONE_YOLO_MODEL,
        prompts: List[str] = settings.CONE_YOLO_PROMPTS,
    ):
        log.info(f"Loading YOLO-World cone model: {model_path}")
        self.model = YOLOWorld(model_path)
        self.model.set_classes(prompts)
        log.info(f"Cone prompts: {prompts}")

    def detect(self, frame: np.ndarray) -> Tuple[List[Point], List[Point], list]:
        """Run YOLO-World on a frame.

        Returns (chosen_4, all_candidates, raw_boxes_xyxy) — raw boxes are
        useful for drawing the debug image with full bbox rectangles.
        """
        results = self.model(
            frame,
            conf=settings.CONE_YOLO_CONF_THRESHOLD,
            imgsz=640,
            verbose=False,
        )[0]

        raw_boxes: list = []
        candidates: List[Point] = []
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            cx, cy = bbox_center(xyxy)
            raw_boxes.append(tuple(xyxy))
            candidates.append((int(cx), int(cy)))

        deduped = _dedupe_close(candidates, settings.CONE_DEDUP_MIN_DIST_PX)
        log.info(f"YOLO-World cone candidates: {len(candidates)} raw "
                 f"({len(deduped)} after dedup)")
        chosen = _select_best_quad(deduped) or []
        return chosen, deduped, raw_boxes


class GoalDetector:
    """Goal-post detector backed by the Roboflow hosted inference API.

    Calls https://detect.roboflow.com/{model_id}/{version} with the frame
    encoded as base64 JPEG, then groups the returned predictions into posts
    vs. crossbars by shape (aspect ratio + size). The chosen post pair is
    fed through the same strict pair-validation used previously so we don't
    accept e.g. two unrelated vertical objects as a goal.
    """

    _API_BASE = "https://detect.roboflow.com"

    def __init__(self) -> None:
        self.api_key = os.environ.get(settings.ROBOFLOW_API_ENV_VAR)
        if not self.api_key:
            raise RuntimeError(
                f"Roboflow API key not set: expected env var "
                f"`{settings.ROBOFLOW_API_ENV_VAR}` (see .env). "
                "Either populate .env or set ENABLE_GOAL_FEATURES=False."
            )
        self.url = (
            f"{self._API_BASE}/{settings.ROBOFLOW_GOAL_MODEL_ID}/"
            f"{settings.ROBOFLOW_GOAL_MODEL_VERSION}"
        )
        self._session = requests.Session()
        log.info(f"Roboflow goal detector ready: {self.url}")

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        result = self.detect_with_score(frame)
        return result[0] if result is not None else None

    def detect_with_score(
        self, frame: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
        """Hit the Roboflow API once and assemble (bbox, score, polygon).

        Returns None when no posts are detected or the post-pair validation
        rejects everything that came back.
        """
        if settings.GOAL_MANUAL_BBOX is not None:
            x1, y1, x2, y2 = settings.GOAL_MANUAL_BBOX
            polygon = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            )
            return (settings.GOAL_MANUAL_BBOX, float("inf"), polygon)

        predictions = self._call_roboflow(frame)
        if not predictions:
            log.info("Roboflow returned 0 predictions for this frame")
            return None

        h_frame, w_frame = frame.shape[:2]
        max_w = w_frame * settings.ROBOFLOW_GOAL_MAX_FRAME_FRAC
        max_h = h_frame * settings.ROBOFLOW_GOAL_MAX_FRAME_FRAC
        posts, crossbars, whole_goal = self._classify_predictions(
            predictions, max_w=max_w, max_h=max_h
        )
        log.info(
            f"Roboflow: {len(predictions)} preds → posts={len(posts)} "
            f"crossbars={len(crossbars)} whole={len(whole_goal)}"
        )

        # If the model returns a single big "goal" bbox (no posts), use it directly.
        if not posts and whole_goal:
            best = max(whole_goal, key=lambda g: (g[2] - g[0]) * (g[3] - g[1]) * g[4])
            x1, y1, x2, y2 = int(best[0]), int(best[1]), int(best[2]), int(best[3])
            bbox = (x1, y1, x2, y2)
            polygon = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            )
            log.info(f"Goal accepted (whole-goal detection): bbox={bbox}")
            return (bbox, best[4] * (x2 - x1) * (y2 - y1), polygon)

        # Posts path: y-band filter then strict pair validation.
        h_frame = frame.shape[0]
        y_min = h_frame * settings.GOAL_POST_MIN_Y_FRAC
        y_max = h_frame * settings.GOAL_POST_MAX_Y_FRAC
        posts = [p for p in posts if p[1] >= y_min and p[3] <= y_max]

        if len(posts) < settings.GOAL_MIN_POSTS_REQUIRED:
            log.info(
                f"Goal rejected: {len(posts)} post(s) after y-filter "
                f"(need ≥{settings.GOAL_MIN_POSTS_REQUIRED})"
            )
            return None

        posts.sort(key=lambda p: p[0])
        best_pair: Optional[Tuple[tuple, tuple]] = None
        best_pair_score = 0.0
        for i, lp in enumerate(posts):
            for rp in posts[i + 1:]:
                width = rp[0] - lp[0]
                if not (
                    settings.GOAL_POST_PAIR_MIN_WIDTH_PX
                    <= width
                    <= settings.GOAL_POST_PAIR_MAX_WIDTH_PX
                ):
                    continue
                overlap = max(0.0, min(lp[3], rp[3]) - max(lp[1], rp[1]))
                union = max(lp[3], rp[3]) - min(lp[1], rp[1])
                if union > 0 and overlap / union < settings.GOAL_POST_Y_OVERLAP_MIN:
                    continue
                pair_score = (lp[4] + rp[4]) * width
                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_pair = (lp, rp)

        if best_pair is None:
            log.info("Goal rejected: no valid post pair after width/y-overlap test")
            return None

        lp, rp = best_pair
        x1 = int(lp[0])
        x2 = int(rp[2])
        y1 = int(min(lp[1], rp[1]))
        y2 = int(max(lp[3], rp[3]))
        if crossbars:
            matching = [
                c for c in crossbars
                if c[0] >= x1 - 30 and c[2] <= x2 + 30
                and c[1] <= y1 + (y2 - y1) * 0.3
            ]
            if matching:
                crossbar = max(matching, key=lambda c: (c[2] - c[0]) * c[4])
                y1 = min(y1, int(crossbar[1]))

        bbox = (x1, y1, x2, y2)
        polygon = _build_goal_polygon([lp, rp], crossbars, bbox)
        log.info(f"Goal accepted: bbox={bbox} pair_score={best_pair_score:.2f}")
        return (bbox, best_pair_score, polygon)

    def _call_roboflow(self, frame: np.ndarray) -> List[dict]:
        """POST the frame to detect.roboflow.com and return raw predictions."""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            log.warning("Roboflow goal detect: cv2.imencode failed")
            return []
        img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        params = {
            "api_key": self.api_key,
            "confidence": settings.ROBOFLOW_GOAL_CONFIDENCE,
            "overlap": settings.ROBOFLOW_GOAL_OVERLAP,
        }
        try:
            resp = self._session.post(
                self.url,
                params=params,
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=settings.ROBOFLOW_GOAL_TIMEOUT_S,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning(f"Roboflow goal detect request failed: {exc}")
            return []
        try:
            payload = resp.json()
        except ValueError:
            log.warning("Roboflow response was not JSON")
            return []
        return payload.get("predictions", []) or []

    @staticmethod
    def _classify_predictions(
        predictions: List[dict],
        max_w: float = float("inf"),
        max_h: float = float("inf"),
    ) -> Tuple[
        List[Tuple[float, float, float, float, float]],
        List[Tuple[float, float, float, float, float]],
        List[Tuple[float, float, float, float, float]],
    ]:
        """Split Roboflow predictions into (posts, crossbars, whole_goal).

        The goalpost-u6e0h model uses a single "goalpost" class for both
        individual posts and whole-goal bboxes — so we rely on shape (aspect
        ratio + dimensions) to decide which bucket each detection goes in.
        Detections wider/taller than max_w/max_h are dropped as bogus
        near-frame-sized false positives.
        """
        posts: List[Tuple[float, float, float, float, float]] = []
        crossbars: List[Tuple[float, float, float, float, float]] = []
        whole: List[Tuple[float, float, float, float, float]] = []
        for p in predictions:
            try:
                cx = float(p["x"])
                cy = float(p["y"])
                pw = float(p["width"])
                ph = float(p["height"])
                conf = float(p.get("confidence", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if pw <= 0 or ph <= 0:
                continue
            if pw > max_w or ph > max_h:
                continue
            x1 = cx - pw / 2.0
            y1 = cy - ph / 2.0
            x2 = cx + pw / 2.0
            y2 = cy + ph / 2.0
            cls = str(p.get("class", "")).lower()
            aspect = ph / pw

            is_tall_post = (
                ph >= settings.GOAL_POST_MIN_HEIGHT_PX
                and aspect >= settings.GOAL_POST_MIN_ASPECT
            )
            is_wide_short = (
                pw >= settings.GOAL_CROSSBAR_MIN_WIDTH_PX
                and aspect <= settings.GOAL_CROSSBAR_MAX_ASPECT
            )

            if "crossbar" in cls:
                crossbars.append((x1, y1, x2, y2, conf))
            elif is_tall_post:
                posts.append((x1, y1, x2, y2, conf))
            elif is_wide_short:
                # A wide short bbox from a "goalpost"-class model is the
                # whole goal mouth (left post + crossbar + right post), not a
                # bare crossbar. Treat it as a whole-goal candidate.
                whole.append((x1, y1, x2, y2, conf))
            else:
                # Catch-all (square-ish): goal mouth captured by a single bbox.
                whole.append((x1, y1, x2, y2, conf))
        return posts, crossbars, whole
