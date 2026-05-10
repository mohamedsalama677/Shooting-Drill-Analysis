"""MediaPipe Pose wrapper. Returns left/right ankle positions in full-frame px."""

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
# Explicit submodule import — works across MediaPipe versions where the
# top-level `mp.solutions` lazy attribute may not be exposed.
from mediapipe.python.solutions import pose as mp_pose

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


class PoseEstimator:
    """Single-person pose estimator. Crops to a person ROI for accuracy."""

    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=settings.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.POSE_MIN_TRACKING_CONFIDENCE,
        )

    def get_ankles(
        self, frame: np.ndarray, person_bbox: Sequence[float]
    ) -> Optional[dict]:
        """Run pose on the cropped person ROI; return ankle positions in full-frame px.

        Returns None if pose detection fails.
        """
        h_full, w_full = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in person_bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_full, x2); y2 = min(h_full, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.pose.process(roi_rgb)
        if not results.pose_landmarks:
            return None

        roi_h, roi_w = roi.shape[:2]
        lm = results.pose_landmarks.landmark

        def to_full_frame(idx: int) -> Tuple[float, float]:
            p = lm[idx]
            return (x1 + p.x * roi_w, y1 + p.y * roi_h)

        return {
            "left": to_full_frame(settings.LEFT_ANKLE_LANDMARK),
            "right": to_full_frame(settings.RIGHT_ANKLE_LANDMARK),
        }

    def close(self):
        self.pose.close()
