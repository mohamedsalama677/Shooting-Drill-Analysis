"""Per-shot analysis. For now: determine which foot kicked the ball.

Future home of: scoring zone, error flags, shot power readout.
"""

from typing import Optional, Sequence

import numpy as np

from detection.pose_estimator import PoseEstimator
from utils.geometry import euclidean
from utils.logger import get_logger

log = get_logger(__name__)


def determine_foot(
    frame: np.ndarray,
    person_bbox: Sequence[float],
    ball_pos_px: Sequence[float],
    pose: PoseEstimator,
) -> Optional[str]:
    """Return 'left' or 'right' — whichever ankle is closer to the ball center.

    Returns None if pose detection fails (logged at warning level).
    """
    ankles = pose.get_ankles(frame, person_bbox)
    if ankles is None:
        log.warning("Pose detection failed — cannot determine kicking foot")
        return None

    ball = (float(ball_pos_px[0]), float(ball_pos_px[1]))
    d_left = euclidean(ankles["left"], ball)
    d_right = euclidean(ankles["right"], ball)
    return "left" if d_left < d_right else "right"
