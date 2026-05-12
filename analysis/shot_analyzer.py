"""Per-shot analysis: foot detection, scoring zone, gate error flag, missed distance."""

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from calibration.homography import pixel_to_meters
from config import settings
from detection.pose_estimator import PoseEstimator
from utils.geometry import euclidean
from utils.logger import get_logger

log = get_logger(__name__)

Point = Tuple[float, float]

# File-based debug log for goal crossing diagnostics.
_goal_dbg_file = None


def _goal_dbg(msg: str) -> None:
    """Write a debug message to both stdout and the goal_debug.txt file."""
    global _goal_dbg_file
    print(msg, flush=True)
    if _goal_dbg_file is not None:
        _goal_dbg_file.write(msg + "\n")
        _goal_dbg_file.flush()


def _open_goal_dbg(output_dir: str) -> None:
    """Open the goal debug log file in the given output directory."""
    global _goal_dbg_file
    if _goal_dbg_file is not None:
        _goal_dbg_file.close()
    path = os.path.join(output_dir, "goal_debug.txt")
    _goal_dbg_file = open(path, "w")



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


def check_outside_gate(
    ball_pos_px: Point,
    cal_H: Optional[np.ndarray],
    ordered_cones_px: Optional[np.ndarray],
    px_per_meter: Optional[float],
) -> Optional[bool]:
    """Return True if ball was on the player's side of the gate at shot moment.

    Homography path: transforms ball to world coords; world y > GATE_DEPTH_M means
    the ball is still on the near (player) side of the gate back-line.
    Scale fallback: projects ball and near-gate midpoint onto the goal direction;
    if ball hasn't reached the near gate line yet, it's outside.
    Returns None when no calibration data is available.
    """
    if cal_H is not None:
        try:
            _wx, world_y = pixel_to_meters(ball_pos_px, cal_H)
            return float(world_y) > float(settings.GATE_DEPTH_M)
        except Exception:
            pass

    if ordered_cones_px is not None and len(ordered_cones_px) >= 4:
        # ordered_cones_px order: [TL, TR, BL, BR] from homography.py
        bl = ordered_cones_px[2]
        br = ordered_cones_px[3]
        near_gate_mid = ((bl[0] + br[0]) / 2.0, (bl[1] + br[1]) / 2.0)
        gx = float(settings.SHOT_GOAL_DIRECTION_X)
        gy = float(settings.SHOT_GOAL_DIRECTION_Y)
        norm = math.hypot(gx, gy) or 1.0
        gux, guy = gx / norm, gy / norm
        proj_ball = ball_pos_px[0] * gux + ball_pos_px[1] * guy
        proj_gate = near_gate_mid[0] * gux + near_gate_mid[1] * guy
        return proj_ball < proj_gate

    return None


def _follow_ball(
    start_frame: int,
    start_pos: Point,
    per_frame_all_balls: List[List[Tuple[int, Point]]],
) -> Tuple[Point, bool]:
    """Greedily follow ball for GOAL_LOOKAHEAD_FRAMES frames.

    Returns (final_pos, exited_frame) where exited_frame is True when the ball
    left the tracked region before the lookahead window ended.
    """
    n_frames = len(per_frame_all_balls)
    end_frame = min(n_frames, start_frame + settings.GOAL_LOOKAHEAD_FRAMES + 1)
    current_pos = start_pos
    gap_count = 0
    exited = False

    for frame_idx in range(start_frame, end_frame):
        candidates = per_frame_all_balls[frame_idx] if frame_idx < n_frames else []
        if not candidates:
            gap_count += 1
            if gap_count > settings.GOAL_MISS_TRACK_MAX_GAP:
                exited = True
                break
            continue

        best_pos: Optional[Point] = None
        best_dist = float("inf")
        for _, pos in candidates:
            d = math.hypot(pos[0] - current_pos[0], pos[1] - current_pos[1])
            if d < best_dist:
                best_pos, best_dist = pos, d

        if best_pos is None or best_dist > settings.GOAL_BALL_PROXIMITY_RADIUS_PX:
            gap_count += 1
            if gap_count > settings.GOAL_MISS_TRACK_MAX_GAP:
                exited = True
                break
            continue

        gap_count = 0
        current_pos = best_pos

    return current_pos, exited


def _point_in_goal(
    pos: Point,
    goal_polygon: Optional[np.ndarray],
    goal_bbox: Tuple[int, int, int, int],
) -> bool:
    """Polygon containment when available, else fall back to bbox bounds."""
    if goal_polygon is not None:
        contour = goal_polygon.astype(np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(contour, (float(pos[0]), float(pos[1])), False) >= 0
    x1, y1, x2, y2 = goal_bbox
    return x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2


def _check_goal_crossing(
    shot_frame_idx: int,
    ball_pos_px: Point,
    per_frame_all_balls: List[List[Tuple[int, Point]]],
    goal_bbox: Tuple[int, int, int, int],
    goal_polygon: Optional[np.ndarray] = None,
    velocity_vec_px_per_frame: Optional[Point] = None,
) -> Tuple[bool, Point]:
    """Follow ball for GOAL_LOOKAHEAD_FRAMES and check if it enters the goal mouth.

    Two-stage check:
      1. Greedy follow on per-frame ball detections — returns scored=True the
         moment a tracked ball lands inside the polygon.  Candidates are
         filtered for **directional consistency** with the shot's release
         vector so stationary (resting) balls near the goal cannot be
         confused with the in-flight shot ball.
      2. Trajectory extrapolation fallback — when the ball is lost before
         reaching the goal, project the shot's release vector forward from the
         last known position and test segment-vs-polygon intersection. Catches
         shots where YOLO drops the fast-moving ball mid-flight.

    Uses cv2.pointPolygonTest on the perspective-correct goal polygon when
    provided, falling back to axis-aligned bbox bounds otherwise.
    Returns (scored, entry_or_final_position).
    """
    n_frames = len(per_frame_all_balls)
    end_frame = min(n_frames, shot_frame_idx + settings.GOAL_LOOKAHEAD_FRAMES + 1)
    current_pos = ball_pos_px
    last_observed_frame = shot_frame_idx
    gap_count = 0
    _goal_dbg(f"\n[GOAL-DBG] _check_goal_crossing: shot_frame={shot_frame_idx} "
              f"ball_pos=({ball_pos_px[0]:.0f},{ball_pos_px[1]:.0f}) "
              f"vel={velocity_vec_px_per_frame} "
              f"goal_bbox={goal_bbox} lookahead={end_frame - shot_frame_idx}")

    # We keep a running velocity updated from actual observations so the
    # Stage-2 extrapolation reflects where the ball was *actually* heading.
    prev_obs_pos = ball_pos_px
    prev_obs_frame = shot_frame_idx
    running_vel = velocity_vec_px_per_frame  # fallback = release vector

    # Track the last position/velocity where the ball was still moving fast.
    # If the tracker latches onto the player (who walks slowly toward the goal),
    # we detect the velocity drop and revert to extrapolation from the last
    # fast-moving position.
    fast_pos = ball_pos_px
    fast_vel = velocity_vec_px_per_frame
    fast_frame = shot_frame_idx

    # Precompute unit direction of the shot for angular filtering.
    shot_dir_valid = False
    shot_ux, shot_uy = 0.0, 0.0
    shot_speed = 0.0
    if velocity_vec_px_per_frame is not None:
        shot_speed = math.hypot(velocity_vec_px_per_frame[0],
                                velocity_vec_px_per_frame[1])
        if shot_speed > 1e-3:
            shot_ux = velocity_vec_px_per_frame[0] / shot_speed
            shot_uy = velocity_vec_px_per_frame[1] / shot_speed
            shot_dir_valid = True

    # Maximum angular deviation (cosine) — ~60° half-cone.
    _MIN_DIR_COS = 0.40
    # If tracked velocity drops below this fraction of original shot speed,
    # consider the tracker to have latched onto the wrong object (player, etc.).
    _VEL_DROP_FRAC = 0.20

    for frame_idx in range(shot_frame_idx, end_frame):
        candidates = per_frame_all_balls[frame_idx] if frame_idx < n_frames else []
        if not candidates:
            gap_count += 1
            if gap_count > settings.GOAL_MISS_TRACK_MAX_GAP:
                break
            continue

        best_pos: Optional[Point] = None
        best_dist = float("inf")
        for _, pos in candidates:
            d = math.hypot(pos[0] - current_pos[0], pos[1] - current_pos[1])

            # --- directional consistency filter ---
            # Skip candidates whose direction from the *shot origin* is
            # substantially different from the shot's release vector. This
            # prevents latching onto resting balls that happen to be close.
            if shot_dir_valid and frame_idx > shot_frame_idx:
                dx_from_origin = pos[0] - ball_pos_px[0]
                dy_from_origin = pos[1] - ball_pos_px[1]
                dist_from_origin = math.hypot(dx_from_origin, dy_from_origin)
                # Only filter when the candidate is far enough from the kick
                # point that the direction is meaningful (> 30 px).
                if dist_from_origin > 30.0:
                    cos_angle = (
                        (dx_from_origin * shot_ux + dy_from_origin * shot_uy)
                        / dist_from_origin
                    )
                    if cos_angle < _MIN_DIR_COS:
                        continue  # off-trajectory — skip

            if d < best_dist:
                best_pos, best_dist = pos, d

        if best_pos is None or best_dist > settings.GOAL_BALL_PROXIMITY_RADIUS_PX:
            gap_count += 1
            if gap_count > settings.GOAL_MISS_TRACK_MAX_GAP:
                break
            continue

        gap_count = 0
        # Update running velocity from observed trajectory.
        dt = frame_idx - prev_obs_frame
        if dt > 0:
            running_vel = (
                (best_pos[0] - prev_obs_pos[0]) / dt,
                (best_pos[1] - prev_obs_pos[1]) / dt,
            )
            # --- velocity-drop detection ---
            # If the tracked object's speed drops below _VEL_DROP_FRAC of the
            # original shot speed, the tracker has almost certainly latched
            # onto the player or a resting ball. Stop tracking and fall back
            # to extrapolation from the last fast-moving position.
            cur_speed = math.hypot(running_vel[0], running_vel[1])
            if cur_speed >= _VEL_DROP_FRAC * shot_speed:
                # Still fast — update the "last known fast" bookmark.
                fast_pos = best_pos
                fast_vel = running_vel
                fast_frame = frame_idx
            elif (frame_idx - shot_frame_idx) > 3 and shot_speed > 5.0:
                # Velocity has dropped significantly — stop tracking.
                _goal_dbg(f"[GOAL-DBG] Velocity drop at frame {frame_idx}: "
                          f"cur_speed={cur_speed:.1f} vs shot_speed={shot_speed:.1f} "
                          f"— reverting to fast_pos=({fast_pos[0]:.0f},{fast_pos[1]:.0f})")
                break

        prev_obs_pos = best_pos
        prev_obs_frame = frame_idx
        current_pos = best_pos
        last_observed_frame = frame_idx

        if _point_in_goal(current_pos, goal_polygon, goal_bbox):
            _goal_dbg(f"[GOAL-DBG] Stage1 HIT: ball tracked INTO goal at frame {frame_idx} "
                      f"pos=({current_pos[0]:.0f},{current_pos[1]:.0f})")
            return True, current_pos

    # Stage 2: trajectory extrapolation when tracking dies before goal.
    # Always use the ORIGINAL release velocity for extrapolation direction.
    # The running velocity from 1-5 frames of noisy tracking is unreliable —
    # it often captures bounces to wrong objects (player, cones, etc.).
    # The release velocity is the best available directional signal.
    observed_count = last_observed_frame - shot_frame_idx
    fast_observed = fast_frame - shot_frame_idx

    # Choose the best starting position for extrapolation:
    # - If we have a fast bookmark, start from there (closest to real ball path)
    # - Otherwise start from the shot origin
    extrap_pos = fast_pos if fast_observed > 0 else ball_pos_px
    extrap_from_frame = fast_frame if fast_observed > 0 else shot_frame_idx

    _goal_dbg(f"[GOAL-DBG] Stage1 done: tracking ended at frame {last_observed_frame} "
              f"(observed {observed_count} frames, fast_observed={fast_observed}) "
              f"final_pos=({current_pos[0]:.0f},{current_pos[1]:.0f}) "
              f"extrap_from=({extrap_pos[0]:.0f},{extrap_pos[1]:.0f})")

    # Require at least 1 frame of actual tracking for extrapolation.
    # With 0 tracked frames, the release velocity alone is too unreliable
    # (it would cause false GOALs for wide shots whose velocity vector
    # happens to point near the goal).
    if fast_observed < 1:
        _goal_dbg(f"[GOAL-DBG] Skipping extrapolation: no tracking evidence "
                  f"(fast_observed={fast_observed})")
    elif velocity_vec_px_per_frame is not None:
        hit, hit_pos = _extrapolate_into_goal(
            extrap_pos,
            velocity_vec_px_per_frame,  # ALWAYS use original release velocity
            extrap_from_frame,
            end_frame,
            goal_polygon,
            goal_bbox,
            shot_frame_idx=shot_frame_idx,
        )
        if hit and hit_pos is not None:
            return True, hit_pos

    return False, current_pos


def _extrapolate_into_goal(
    last_pos: Point,
    velocity_vec_px_per_frame: Point,
    last_observed_frame: int,
    end_frame: int,
    goal_polygon: Optional[np.ndarray],
    goal_bbox: Tuple[int, int, int, int],
    shot_frame_idx: int = 0,
) -> Tuple[bool, Optional[Point]]:
    """Project the shot's release vector from last_pos and test for goal entry.

    Samples integer frame steps within the remaining lookahead window. The
    release vector is in pixels-per-frame, so a step of `i` corresponds to
    where the ball would be `i` frames after last_observed_frame, assuming
    constant velocity. First sample inside the polygon wins.

    The extrapolation is capped at GOAL_EXTRAPOLATION_MAX_FRAMES to prevent
    runaway projections when the ball is lost early and the velocity vector
    is unreliable.
    """
    vx, vy = velocity_vec_px_per_frame
    speed_pxpf = math.hypot(vx, vy)
    if speed_pxpf < 1e-3:
        return False, None
    frames_remaining = max(0, end_frame - last_observed_frame - 1)
    if frames_remaining == 0:
        return False, None

    # Cap the extrapolation window to prevent runaway projections.
    # 45 frames (~1.5 s at 30 fps) covers even slow shots.
    # The caller already guards against zero-tracking cases.
    _MAX_EXTRAPOLATION_FRAMES = 45
    frames_remaining = min(frames_remaining, _MAX_EXTRAPOLATION_FRAMES)

    # Sub-frame sampling so we don't step over a thin polygon between frames.
    samples_per_frame = 4
    total_samples = frames_remaining * samples_per_frame
    _goal_dbg(f"[GOAL-DBG] Extrapolating: from=({last_pos[0]:.0f},{last_pos[1]:.0f}) "
              f"vel=({vx:.1f},{vy:.1f}) frames_remaining={frames_remaining} "
              f"observed_frames={last_observed_frame - shot_frame_idx}")
    for i in range(1, total_samples + 1):
        t = i / samples_per_frame
        x = last_pos[0] + vx * t
        y = last_pos[1] + vy * t
        if _point_in_goal((x, y), goal_polygon, goal_bbox):
            _goal_dbg(f"[GOAL-DBG] Extrapolation HIT at t={t:.1f} frames "
                      f"pos=({x:.0f},{y:.0f})")
            return True, (float(x), float(y))
    _goal_dbg(f"[GOAL-DBG] Extrapolation MISS — no intersection")
    return False, None


def classify_scoring_zone(
    ball_pos_px: Point,
    goal_bbox: Tuple[int, int, int, int],
) -> Tuple[str, int]:
    """Return (zone_name, points) for a ball position inside the goal bbox.

    Grid: 3 x-columns × 2 y-rows
        TL(10) | TC(7) | TR(10)
        BL(5)  | BC(3) | BR(5)
    """
    x1, y1, x2, y2 = goal_bbox
    bx, by = float(ball_pos_px[0]), float(ball_pos_px[1])
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    col = min(int((bx - x1) / (w / 3.0)), 2)
    row = min(int((by - y1) / (h / 2.0)), 1)
    col = max(col, 0)
    row = max(row, 0)
    zone = settings.GOAL_ZONE_NAMES[(col, row)]
    points = settings.GOAL_ZONE_POINTS[(col, row)]
    return zone, points


def compute_missed_distance(
    shot_frame_idx: int,
    ball_pos_px: Point,
    per_frame_all_balls: List[List[Tuple[int, Point]]],
    goal_bbox: Optional[Tuple[int, int, int, int]],
    cal_H: Optional[np.ndarray],
    px_per_meter: Optional[float],
    frame_width: int,
    frame_height: int,
    velocity_vec_px_per_frame: Optional[Point] = None,
) -> Optional[float]:
    """Track ball after shot, return distance in meters to nearest goalpost.

    Returns None if calibration is unavailable.
    """
    n_frames = len(per_frame_all_balls)
    end_frame = min(n_frames, shot_frame_idx + settings.GOAL_LOOKAHEAD_FRAMES + 1)
    current_pos = ball_pos_px
    gap_count = 0

    # Directional filter — same logic as _check_goal_crossing.
    shot_dir_valid = False
    shot_ux, shot_uy = 0.0, 0.0
    if velocity_vec_px_per_frame is not None:
        shot_speed = math.hypot(velocity_vec_px_per_frame[0],
                                velocity_vec_px_per_frame[1])
        if shot_speed > 1e-3:
            shot_ux = velocity_vec_px_per_frame[0] / shot_speed
            shot_uy = velocity_vec_px_per_frame[1] / shot_speed
            shot_dir_valid = True
    _MIN_DIR_COS = 0.40

    for frame_idx in range(shot_frame_idx, end_frame):
        candidates = per_frame_all_balls[frame_idx] if frame_idx < n_frames else []
        if not candidates:
            gap_count += 1
            if gap_count > settings.GOAL_MISS_TRACK_MAX_GAP:
                break
            continue

        best_pos: Optional[Point] = None
        best_dist = float("inf")
        for _, pos in candidates:
            bx, by = pos
            if bx < 0 or bx > frame_width or by < 0 or by > frame_height:
                continue

            # Directional consistency filter
            if shot_dir_valid and frame_idx > shot_frame_idx:
                dx_from_origin = bx - ball_pos_px[0]
                dy_from_origin = by - ball_pos_px[1]
                dist_from_origin = math.hypot(dx_from_origin, dy_from_origin)
                if dist_from_origin > 30.0:
                    cos_angle = (
                        (dx_from_origin * shot_ux + dy_from_origin * shot_uy)
                        / dist_from_origin
                    )
                    if cos_angle < _MIN_DIR_COS:
                        continue

            d = math.hypot(bx - current_pos[0], by - current_pos[1])
            if d < best_dist:
                best_pos, best_dist = pos, d

        if best_pos is None or best_dist > settings.GOAL_BALL_PROXIMITY_RADIUS_PX:
            gap_count += 1
            if gap_count > settings.GOAL_MISS_TRACK_MAX_GAP:
                break
            continue

        gap_count = 0
        current_pos = best_pos

    final_pos = current_pos

    if goal_bbox is not None:
        x1, y1, x2, y2 = goal_bbox
        cy = (y1 + y2) / 2.0
        post_left: Point = (float(x1), cy)
        post_right: Point = (float(x2), cy)
        dist_left_px = euclidean(final_pos, post_left)
        dist_right_px = euclidean(final_pos, post_right)
        nearest_post = post_left if dist_left_px <= dist_right_px else post_right
        nearest_px = min(dist_left_px, dist_right_px)

        if cal_H is not None:
            try:
                world_ball = pixel_to_meters(final_pos, cal_H)
                world_post = pixel_to_meters(nearest_post, cal_H)
                return math.hypot(
                    world_ball[0] - world_post[0],
                    world_ball[1] - world_post[1],
                )
            except Exception:
                pass
        if px_per_meter is not None and px_per_meter > 0:
            return nearest_px / px_per_meter
        return None

    # Geometry fallback: estimate goal center from calibration when no goal bbox
    if settings.GOAL_MISS_GEOMETRY_FALLBACK and cal_H is not None:
        try:
            world_ball = pixel_to_meters(final_pos, cal_H)
            goal_center_world: Point = (
                settings.GATE_WIDTH_M / 2.0,
                -float(settings.DIST_CONES_TO_GOAL_M),
            )
            return math.hypot(
                world_ball[0] - goal_center_world[0],
                world_ball[1] - goal_center_world[1],
            )
        except Exception:
            pass

    return None


def analyze_shots(
    shots: list,
    per_frame_all_balls: List[List[Tuple[int, Point]]],
    goal_bbox: Optional[Tuple[int, int, int, int]],
    cal_H: Optional[np.ndarray],
    ordered_cones_px: Optional[np.ndarray],
    px_per_meter: Optional[float],
    frame_width: int,
    frame_height: int,
    goal_polygon: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Populate Feature 4/5/6 fields on each ShotEvent in-place."""
    if output_dir is not None:
        _open_goal_dbg(output_dir)
    for shot in shots:
        # Feature 5: outside gate check
        shot.outside_gate = check_outside_gate(
            shot.ball_pos_px, cal_H, ordered_cones_px, px_per_meter
        )

        if goal_bbox is None:
            shot.scored = None
        else:
            # Feature 4: goal crossing + scoring zone
            scored, entry_pos = _check_goal_crossing(
                shot.frame_idx,
                shot.ball_pos_px,
                per_frame_all_balls,
                goal_bbox,
                goal_polygon=goal_polygon,
                velocity_vec_px_per_frame=getattr(
                    shot, "velocity_vec_px_per_frame", None
                ),
            )
            shot.scored = scored
            if scored:
                shot.scoring_zone, shot.zone_points = classify_scoring_zone(
                    entry_pos, goal_bbox
                )
                log.info(
                    f"Shot #{shot.index}: GOAL zone={shot.scoring_zone} "
                    f"pts={shot.zone_points}"
                )
            else:
                # Feature 6: missed shot distance
                shot.missed_distance_m = compute_missed_distance(
                    shot.frame_idx,
                    shot.ball_pos_px,
                    per_frame_all_balls,
                    goal_bbox,
                    cal_H,
                    px_per_meter,
                    frame_width,
                    frame_height,
                    velocity_vec_px_per_frame=getattr(
                        shot, "velocity_vec_px_per_frame", None
                    ),
                )
                log.info(
                    f"Shot #{shot.index}: MISS dist={shot.missed_distance_m}"
                )
