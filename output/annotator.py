"""Frame annotation: bounding boxes, ball trail, calibration overlay, shot banner."""

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import settings


def draw_detections(
    frame: np.ndarray,
    persons: Sequence,
    balls: Sequence,
    cones: Sequence[Tuple[int, int]],
) -> None:
    for det in persons:
        x1, y1, x2, y2 = [int(v) for v in det.xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), settings.COLOR_PERSON, 2)
        cv2.putText(frame, f"player {det.confidence:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, settings.COLOR_PERSON, 1)
    for det in balls:
        x1, y1, x2, y2 = [int(v) for v in det.xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), settings.COLOR_BALL, 2)
    for (cx, cy) in cones:
        cv2.circle(frame, (cx, cy), 8, settings.COLOR_CONE, -1)


def draw_ball_trail(frame: np.ndarray, recent: Iterable[Tuple[int, int]]) -> None:
    pts = list(recent)
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], settings.COLOR_TRAIL, 2)


# Distinct colors so multiple ball tracks don't visually merge.
_TRACK_PALETTE = [
    (255, 0, 255),   # magenta
    (0, 255, 255),   # yellow
    (255, 255, 0),   # cyan
    (0, 255, 0),     # green
    (255, 128, 0),   # blue-orange
    (128, 0, 255),   # purple
]


def draw_multi_ball_trails(frame: np.ndarray, trails_per_track: dict) -> None:
    """Draw one colored trail per ball track."""
    for tid, recent in trails_per_track.items():
        color = _TRACK_PALETTE[tid % len(_TRACK_PALETTE)]
        pts = list(recent)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 2)


def draw_scale_segment(
    frame: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    px_per_meter: float,
) -> None:
    """Yellow segment between the 2 ref cones used for 2-point scale calibration."""
    a = (int(p1[0]), int(p1[1]))
    b = (int(p2[0]), int(p2[1]))
    cv2.line(frame, a, b, (0, 255, 255), 3)
    cv2.circle(frame, a, 6, (0, 255, 255), -1)
    cv2.circle(frame, b, 6, (0, 255, 255), -1)
    midpoint = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2 - 12)
    cv2.putText(frame, f"scale: {px_per_meter:.0f} px/m", midpoint,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)


def draw_gate(frame: np.ndarray, ordered_cones_px: np.ndarray) -> None:
    """ordered_cones_px is the [TL, TR, BL, BR] np.float32 array from homography."""
    if ordered_cones_px is None:
        return
    pts = ordered_cones_px.astype(np.int32)
    # draw gate quadrilateral: TL→TR→BR→BL→TL
    cv2.polylines(frame, [np.array([pts[0], pts[1], pts[3], pts[2]])],
                  isClosed=True, color=settings.COLOR_GATE, thickness=2)


def draw_shot_banner(
    frame: np.ndarray,
    shot_index: int,
    foot: Optional[str],
    velocity_mps: float,
) -> None:
    h, w = frame.shape[:2]
    foot_str = (foot or "?").upper()
    text = f"SHOT #{shot_index}  {foot_str} FOOT  {velocity_mps:.1f} m/s"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    pad = 14
    x = (w - tw) // 2
    y = 60
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad),
                  settings.COLOR_BANNER_BG, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                settings.COLOR_BANNER_TEXT, 2, cv2.LINE_AA)


def draw_velocity_readout(frame: np.ndarray, v_mps: Optional[float]) -> None:
    if v_mps is None:
        return
    cv2.putText(frame, f"ball: {v_mps:5.2f} m/s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_calibration_badge(frame: np.ndarray, method: str) -> None:
    """Small badge in the top-right showing how calibration was sourced."""
    text = f"CAL: {method}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    h, w = frame.shape[:2]
    x = w - tw - 18
    y = 32
    cv2.rectangle(frame, (x - 8, y - th - 6), (x + tw + 8, y + 6),
                  settings.COLOR_BANNER_BG, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                settings.COLOR_BANNER_TEXT, 2, cv2.LINE_AA)
