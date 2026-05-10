"""Cone detection via YOLO-World (zero-shot, text-prompted).

No training data, no manual input, no HSV color matching. Drives a YOLO-World
model with a text prompt like "traffic cone" and returns up to 4 cone centroids
that best form a rectangle.
"""

import itertools
import math
from typing import List, Optional, Tuple

import numpy as np
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
