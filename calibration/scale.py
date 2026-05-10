"""2-point linear scale calibration (with smart 4-cone side detection).

Used when the 4 cones don't form a clear projective rectangle (camera angle is
too oblique for a 4-point homography to converge).

Strategy:
    - Pairwise distances among all detected cones.
    - The 2 farthest are typically DIAGONALS of the cone gate (= 2 × gate_width × √2).
    - The shortest 4 distances (when there are 4 cones) are typically the 4 SIDES
      of the gate quadrilateral; their mean ≈ gate_width.
    - We use this mean-side approach to get a px/m ratio that actually matches
      the gate width, not the diagonal.
"""

import itertools
import math
from typing import List, Tuple

from config import settings
from utils.geometry import euclidean
from utils.logger import get_logger

log = get_logger(__name__)

Point = Tuple[float, float]


def compute_scale(
    cones_px: List[Point],
    gate_width_m: float = settings.GATE_WIDTH_M,
) -> Tuple[float, Tuple[Point, Point]]:
    """Return (px_per_meter, (ref_pt_a, ref_pt_b)).

    With 4 cones: average the 4 SHORTEST pairwise distances (the gate's sides)
    and treat that mean as `gate_width_m`. The (ref_a, ref_b) returned is the
    PAIR closest to that mean side length, used purely for visual overlay.

    With 2-3 cones: fall back to the 2 farthest cones = `gate_width_m`.
    """
    if len(cones_px) < 2:
        raise ValueError(f"Need ≥2 cones for scale calibration, got {len(cones_px)}")

    pairs = list(itertools.combinations(cones_px, 2))
    distances = [(euclidean(a, b), (a, b)) for a, b in pairs]
    distances.sort(key=lambda t: t[0])  # ascending

    if len(cones_px) >= 4:
        # 4 cones → 6 pairs total. The 4 shortest are the sides of the
        # quadrilateral, the 2 longest are the diagonals.
        sides = distances[:4]
        mean_side_px = sum(d for d, _ in sides) / 4.0
        px_per_meter = mean_side_px / gate_width_m
        # Pick a side closest to the mean for the visual overlay.
        ref = min(sides, key=lambda t: abs(t[0] - mean_side_px))[1]
        log.info(
            f"Scale (4-cone mean side): {px_per_meter:.2f} px/m "
            f"(mean side {mean_side_px:.1f} px ↔ {gate_width_m} m; "
            f"sides={[round(d) for d, _ in sides]}, "
            f"diagonals={[round(d) for d, _ in distances[4:]]})"
        )
        return px_per_meter, ref

    # Fewer than 4 cones — fall back to "2 farthest = gate width".
    px_dist, ref = distances[-1]
    px_per_meter = px_dist / gate_width_m
    log.info(
        f"Scale (2 farthest cones): {px_per_meter:.2f} px/m "
        f"(from {px_dist:.1f} px ↔ {gate_width_m} m)"
    )
    return px_per_meter, ref


def pixel_distance_to_meters_scale(
    p1: Point, p2: Point, px_per_meter: float
) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1]) / px_per_meter
