"""Player + ball detection via YOLOv8 (COCO classes 0 + 32).

Cone detection lives in `detection/cone_detector.py` (YOLO-World, zero-shot).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from config import settings
from utils.geometry import bbox_center, euclidean
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    confidence: float
    track_id: Optional[int] = None


class Detector:
    """Wraps a YOLOv8 model. Single forward pass returns persons and balls."""

    def __init__(self, model_path: str = settings.YOLO_MODEL):
        log.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> dict:
        """Run YOLO+BoT-SORT tracker; bucket results by class.

        Using `model.track(persist=True)` keeps stable IDs across frames for
        more consistent player bboxes (used by pose at shot frames).
        """
        results = self.model.track(
            frame,
            classes=[settings.COCO_CLASS_PERSON, settings.COCO_CLASS_BALL],
            conf=settings.YOLO_CONF_THRESHOLD,
            tracker=settings.YOLO_TRACKER,
            persist=True,
            imgsz=640,
            verbose=False,
        )[0]

        persons: List[Detection] = []
        balls: List[Detection] = []
        if results.boxes is None:
            return {"persons": persons, "balls": balls}
        for box in results.boxes:
            cls_id = int(box.cls[0])
            xyxy = tuple(box.xyxy[0].tolist())
            conf = float(box.conf[0])
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])
            det = Detection(xyxy=xyxy, confidence=conf, track_id=track_id)
            if cls_id == settings.COCO_CLASS_PERSON:
                persons.append(det)
            elif cls_id == settings.COCO_CLASS_BALL:
                balls.append(det)
        if len(balls) <= settings.YOLO_BALL_FALLBACK_MAX_TRACKED_BALLS:
            balls.extend(self._fallback_ball_detections(frame, balls))
        return {"persons": persons, "balls": balls}

    def _fallback_ball_detections(
        self,
        frame: np.ndarray,
        tracked_balls: List[Detection],
    ) -> List[Detection]:
        """Recover small kicked balls that BoT-SORT suppresses.

        The tracker can keep the large/static foreground ball and drop a small
        low-confidence ball flying toward goal. A ball-only prediction pass
        keeps those observations available; `MultiBallTracker` will attach a
        fallback ID when no YOLO track ID exists.
        """
        results = self.model.predict(
            frame,
            classes=[settings.COCO_CLASS_BALL],
            conf=settings.YOLO_BALL_FALLBACK_CONF_THRESHOLD,
            imgsz=640,
            verbose=False,
        )[0]
        if results.boxes is None:
            return []

        tracked_centers = [bbox_center(det.xyxy) for det in tracked_balls]
        recovered: List[Detection] = []
        for box in results.boxes:
            xyxy = tuple(box.xyxy[0].tolist())
            center = bbox_center(xyxy)
            if any(
                euclidean(center, tracked_center)
                <= settings.YOLO_BALL_FALLBACK_MIN_DIST_PX
                for tracked_center in tracked_centers
            ):
                continue
            tracked_centers.append(center)
            recovered.append(Detection(
                xyxy=xyxy,
                confidence=float(box.conf[0]),
                track_id=None,
            ))
        return recovered
