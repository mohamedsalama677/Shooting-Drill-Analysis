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
    scored: Optional[bool] = None,
    scoring_zone: Optional[str] = None,
    zone_points: Optional[int] = None,
    outside_gate: Optional[bool] = None,
    missed_distance_m: Optional[float] = None,
) -> None:
    h, w = frame.shape[:2]
    foot_str = (foot or "?").upper()
    line1 = f"SHOT #{shot_index}  {foot_str} FOOT  {velocity_mps:.1f} m/s"
    if outside_gate:
        line1 += "  [OUTSIDE GATE]"

    if scored is True and scoring_zone is not None and zone_points is not None:
        line2: Optional[str] = f"GOAL!  Zone {scoring_zone}  (+{zone_points} pts)"
        line2_color = (0, 255, 0)
    elif scored is False and missed_distance_m is not None:
        line2 = f"MISS  {missed_distance_m:.1f} m from goal"
        line2_color = (0, 100, 255)
    elif scored is False:
        line2 = "MISS"
        line2_color = (0, 100, 255)
    else:
        line2 = None
        line2_color = settings.COLOR_BANNER_TEXT

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale1, scale2 = 1.0, 0.75
    thick = 2
    pad = 14

    (tw1, th1), _ = cv2.getTextSize(line1, font, scale1, thick)
    total_h = th1 + pad * 2
    if line2 is not None:
        (tw2, th2), _ = cv2.getTextSize(line2, font, scale2, thick)
        total_h += th2 + pad
        box_w = max(tw1, tw2) + pad * 2
    else:
        tw2 = th2 = 0
        box_w = tw1 + pad * 2

    x = (w - box_w) // 2
    y_top = 10
    cv2.rectangle(frame, (x, y_top), (x + box_w, y_top + total_h),
                  settings.COLOR_BANNER_BG, -1)

    y1_text = y_top + pad + th1
    color1 = (0, 165, 255) if outside_gate else settings.COLOR_BANNER_TEXT
    cv2.putText(frame, line1, (x + pad, y1_text), font, scale1,
                color1, thick, cv2.LINE_AA)

    if line2 is not None:
        y2_text = y1_text + pad + th2
        cv2.putText(frame, line2, (x + pad, y2_text), font, scale2,
                    line2_color, thick, cv2.LINE_AA)


def draw_goal_bbox(
    frame: np.ndarray,
    goal_bbox: Tuple[int, int, int, int],
) -> None:
    """Draw the detected goal bounding box in white (axis-aligned fallback)."""
    x1, y1, x2, y2 = goal_bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(frame, "GOAL", (x1 + 4, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def draw_goal_polygon(
    frame: np.ndarray,
    goal_polygon: np.ndarray,
) -> None:
    """Draw the perspective-correct goal mouth polygon (TL, TR, BR, BL).

    Traces the actual goal shape: a trapezoid that follows the camera angle,
    rather than an axis-aligned rectangle that misaligns with the real mouth.
    """
    pts = goal_polygon.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
    # Anchor the label at the polygon's top-left corner (smallest y, then x).
    flat = goal_polygon.reshape(-1, 2)
    anchor_idx = int(np.lexsort((flat[:, 0], flat[:, 1]))[0])
    ax, ay = int(flat[anchor_idx, 0]), int(flat[anchor_idx, 1])
    cv2.putText(frame, "GOAL", (ax + 4, ay + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def draw_shot_status_panel(
    frame: np.ndarray,
    shot_index: int,
    scored: Optional[bool],
    scoring_zone: Optional[str] = None,
    zone_points: Optional[int] = None,
    missed_distance_m: Optional[float] = None,
) -> None:
    """Right-side vertical panel showing GOAL / NO GOAL for the current shot.

    Drawn for SHOT_BANNER_DURATION_FRAMES after each shot (the caller controls
    persistence). Colors: green for GOAL, red for NO GOAL, gray when unknown.
    """
    h, w = frame.shape[:2]
    panel_w = max(180, int(w * 0.13))
    panel_h = max(140, int(h * 0.20))
    x1 = w - panel_w - 12
    y1 = 12
    x2 = w - 12
    y2 = y1 + panel_h

    if scored is True:
        accent = (0, 200, 0)
        title = "GOAL"
    elif scored is False:
        accent = (40, 40, 220)
        title = "NO GOAL"
    else:
        accent = (140, 140, 140)
        title = "UNKNOWN"

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), accent, 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(title, font, 1.1, 3)
    tx = x1 + (panel_w - tw) // 2
    ty = y1 + th + 18
    cv2.putText(frame, title, (tx, ty), font, 1.1, accent, 3, cv2.LINE_AA)

    sub = f"Shot #{shot_index}"
    (sw, sh), _ = cv2.getTextSize(sub, font, 0.6, 2)
    cv2.putText(frame, sub, (x1 + (panel_w - sw) // 2, ty + sh + 14),
                font, 0.6, (230, 230, 230), 2, cv2.LINE_AA)

    if scored is True and scoring_zone is not None and zone_points is not None:
        line = f"Zone {scoring_zone}  +{zone_points}"
    elif scored is False and missed_distance_m is not None:
        line = f"Miss {missed_distance_m:.1f} m"
    else:
        line = ""
    if line:
        (lw, lh), _ = cv2.getTextSize(line, font, 0.6, 2)
        cv2.putText(frame, line, (x1 + (panel_w - lw) // 2, ty + sh + 14 + lh + 14),
                    font, 0.6, accent, 2, cv2.LINE_AA)


def draw_scoring_zone(
    frame: np.ndarray,
    goal_bbox: Tuple[int, int, int, int],
    highlight_zone: Optional[str] = None,
    points: Optional[int] = None,
) -> None:
    """Draw the 3×2 scoring zone grid on the goal bbox.

    When highlight_zone is given, fills that cell green (semi-transparent)
    and labels its point value.
    """
    from config import settings as _s
    x1, y1, x2, y2 = goal_bbox
    w = x2 - x1
    h = y2 - y1
    col_w = w // 3
    row_h = h // 2

    # Grid lines
    cv2.line(frame, (x1 + col_w, y1), (x1 + col_w, y2), (180, 180, 180), 1)
    cv2.line(frame, (x1 + 2 * col_w, y1), (x1 + 2 * col_w, y2), (180, 180, 180), 1)
    cv2.line(frame, (x1, y1 + row_h), (x2, y1 + row_h), (180, 180, 180), 1)

    if highlight_zone is not None:
        zone_to_cr = {v: k for k, v in _s.GOAL_ZONE_NAMES.items()}
        cr = zone_to_cr.get(highlight_zone)
        if cr:
            col, row = cr
            zx1 = x1 + col * col_w
            zx2 = zx1 + col_w
            zy1 = y1 + row * row_h
            zy2 = zy1 + row_h
            overlay = frame.copy()
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            if points is not None:
                cv2.putText(frame, f"+{points}", (zx1 + 4, zy2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                            cv2.LINE_AA)


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
