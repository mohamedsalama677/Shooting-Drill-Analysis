"""Pixel ↔ meters via a 4-cone homography.

The 4 cones form a known rectangle on the ground plane (gate width × depth).
Map their pixel centroids to that rectangle and we get a perspective transform
H that converts any pixel to its real-world ground coordinates.
"""

from typing import Sequence, Tuple

import cv2
import numpy as np

from config import settings
from utils.geometry import euclidean
from utils.logger import get_logger

log = get_logger(__name__)


class DegenerateCalibrationError(RuntimeError):
    """Raised when the 4 cones don't form a valid quadrilateral.

    Typical cause: cones are nearly collinear in image space because of a low
    camera angle. Caller should fall back to a 2-point scale calibration.
    """


def _order_cones_tl_tr_bl_br(
    cones: Sequence[Tuple[int, int]]
) -> np.ndarray:
    """Sort 4 points into [TL, TR, BL, BR] in image coords (y grows downward).

    TL = min(x+y), BR = max(x+y), TR = min(y-x), BL = max(y-x).
    """
    pts = np.array(cones, dtype=np.float32)
    s = pts[:, 0] + pts[:, 1]
    yx = pts[:, 1] - pts[:, 0]
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(yx)],  # TR
        pts[np.argmax(yx)],  # BL
        pts[np.argmax(s)],   # BR
    ], dtype=np.float32)


def compute_homography(
    cones: Sequence[Tuple[int, int]],
    gate_width_m: float = settings.GATE_WIDTH_M,
    gate_depth_m: float = settings.GATE_DEPTH_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (H, ordered_cones_pixel) where H maps pixel → meters on the ground plane.

    Raises ValueError if fewer than 4 cones are provided.
    """
    if len(cones) < 4:
        raise ValueError(f"Need 4 cones for homography, got {len(cones)}")

    src = _order_cones_tl_tr_bl_br(cones)
    # Degeneracy check: if the TL/TR/BL/BR ordering collapses to fewer than 4
    # distinct points (cones nearly collinear in image), homography would be
    # rank-deficient — caller should switch to 2-point scale calibration.
    unique_pts = {(float(p[0]), float(p[1])) for p in src.tolist()}
    if len(unique_pts) < 4:
        raise DegenerateCalibrationError(
            f"Cones are nearly collinear in image space; ordered to "
            f"{len(unique_pts)} unique points — can't compute homography"
        )
    dst = np.array([
        [0.0,            0.0],
        [gate_width_m,   0.0],
        [0.0,            gate_depth_m],
        [gate_width_m,   gate_depth_m],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise RuntimeError("Homography solve failed")
    log.info(f"Homography established from cones: {src.tolist()}")
    return H, src


def pixel_to_meters(point_px: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    pt = np.array([[[point_px[0], point_px[1]]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(warped[0]), float(warped[1])


def pixel_distance_to_meters(
    p1_px: Tuple[float, float], p2_px: Tuple[float, float], H: np.ndarray
) -> float:
    p1_m = pixel_to_meters(p1_px, H)
    p2_m = pixel_to_meters(p2_px, H)
    return euclidean(p1_m, p2_m)
