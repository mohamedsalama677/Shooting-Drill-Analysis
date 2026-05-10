"""Pure geometric helpers used across detection, calibration, and analysis."""

import math
from typing import Iterable, Sequence, Tuple

Point = Tuple[float, float]


def bbox_center(xyxy: Sequence[float]) -> Point:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_foot_point(xyxy: Sequence[float]) -> Point:
    """The bottom-center of a bbox — best ground-plane proxy for a standing person."""
    x1, _, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, float(y2))


def euclidean(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def nearest_point(target: Point, candidates: Iterable[Point]) -> Tuple[int, float]:
    """Return (index, distance) of the candidate closest to target."""
    best_idx, best_dist = -1, float("inf")
    for i, c in enumerate(candidates):
        d = euclidean(target, c)
        if d < best_dist:
            best_idx, best_dist = i, d
    return best_idx, best_dist
