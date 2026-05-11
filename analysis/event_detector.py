"""Shot event detection from tracked ball trajectories.

A valid shot is more than a velocity spike: the ball should be near the player
at contact, then travel away after the kick. This rejects dribbles and setup
touches while keeping the shot count variable for any input video.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from calibration.homography import pixel_distance_to_meters
from calibration.scale import pixel_distance_to_meters_scale
from config import settings
from utils.geometry import euclidean
from utils.logger import get_logger

log = get_logger(__name__)

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Calibration = Union[np.ndarray, float]   # H matrix OR px_per_meter


@dataclass
class ShotEvent:
    index: int
    frame_idx: int
    ball_pos_px: Point
    velocity_mps: float
    track_id: Optional[int] = None
    confidence: float = 0.0
    source: str = "scored"
    debug: dict = field(default_factory=dict)
    # Per-frame pixel displacement vector at release — used to extrapolate the
    # ball trajectory when post-shot tracking dies before the ball reaches the
    # goal. None when the candidate had no usable next-frame observation.
    velocity_vec_px_per_frame: Optional[Point] = None
    # Feature 5 — gate error flag
    outside_gate: Optional[bool] = None
    # Feature 4 — goal detection & scoring zone
    scored: Optional[bool] = None
    scoring_zone: Optional[str] = None      # "TL","TC","TR","BL","BC","BR"
    zone_points: Optional[int] = None
    # Feature 6 — missed shot distance
    missed_distance_m: Optional[float] = None


@dataclass
class ShotCandidate:
    frame_idx: int
    previous_frame_idx: int
    ball_pos_px: Point
    previous_ball_pos_px: Point
    velocity_mps: float
    track_id: Optional[int]
    source: str
    accepted: bool = False
    rejection_reason: str = ""
    confidence: float = 0.0
    contact_distance_px: Optional[float] = None
    post_travel_px: float = 0.0
    player_separation_px: Optional[float] = None
    initial_goalward_progress_px: float = 0.0
    initial_goalward_ratio: float = 0.0
    goalward_progress_px: float = 0.0
    goalward_ratio: float = 0.0
    relative_goalward_progress_px: Optional[float] = None
    metrics: dict = field(default_factory=dict)

    def to_debug_dict(self) -> dict:
        return {
            "frame": int(self.frame_idx),
            "contact_frame": int(self.previous_frame_idx),
            "track_id": self.track_id,
            "source": self.source,
            "accepted": bool(self.accepted),
            "rejection_reason": self.rejection_reason,
            "confidence": round(float(self.confidence), 3),
            "velocity_mps": round(float(self.velocity_mps), 3),
            "ball_pos_px": [int(self.ball_pos_px[0]), int(self.ball_pos_px[1])],
            "contact_ball_pos_px": [
                int(self.previous_ball_pos_px[0]),
                int(self.previous_ball_pos_px[1]),
            ],
            "contact_distance_px": _round_optional(self.contact_distance_px),
            "post_travel_px": round(float(self.post_travel_px), 1),
            "player_separation_px": _round_optional(self.player_separation_px),
            "initial_goalward_progress_px": round(
                float(self.initial_goalward_progress_px), 1
            ),
            "initial_goalward_ratio": round(float(self.initial_goalward_ratio), 3),
            "goalward_progress_px": round(float(self.goalward_progress_px), 1),
            "goalward_ratio": round(float(self.goalward_ratio), 3),
            "relative_goalward_progress_px": _round_optional(
                self.relative_goalward_progress_px
            ),
            "metrics": self.metrics,
        }


def _round_optional(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 1)


def _pixel_dist_meters(p1: Point, p2: Point, cal: Calibration) -> float:
    """Dispatch to the right calibration helper based on type."""
    if isinstance(cal, np.ndarray):
        return pixel_distance_to_meters(p1, p2, cal)
    return pixel_distance_to_meters_scale(p1, p2, cal)


def _cal_at_frame(
    calibrations: Union[Calibration, Sequence[Calibration]],
    frame_idx: int,
) -> Calibration:
    if isinstance(calibrations, (list, tuple)):
        if not calibrations:
            raise ValueError("calibrations list is empty")
        if frame_idx < len(calibrations):
            return calibrations[frame_idx]
        return calibrations[-1]
    return calibrations


def _bbox_at(
    person_bboxes: Optional[Sequence[Optional[BBox]]],
    frame_idx: int,
) -> Optional[BBox]:
    if person_bboxes is None or frame_idx < 0 or frame_idx >= len(person_bboxes):
        return None
    return person_bboxes[frame_idx]


def _distance_point_to_bbox(point: Point, bbox: Optional[BBox]) -> Optional[float]:
    if bbox is None:
        return None
    x, y = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def _bbox_foot_point(bbox: Optional[BBox]) -> Optional[Point]:
    if bbox is None:
        return None
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def _contact_threshold_px(bbox: Optional[BBox]) -> float:
    if bbox is None:
        return float("inf")
    height = max(1.0, bbox[3] - bbox[1])
    return max(settings.SHOT_CONTACT_MAX_DIST_PX, height * 0.25)


def _build_per_frame(
    trajectories: Dict[int, List[Optional[Point]]],
) -> List[List[Tuple[int, Point]]]:
    if not trajectories:
        return []
    n_frames = max(len(t) for t in trajectories.values())
    per_frame: List[List[Tuple[int, Point]]] = [[] for _ in range(n_frames)]
    for tid, traj in trajectories.items():
        for frame_idx, pos in enumerate(traj):
            if pos is not None:
                per_frame[frame_idx].append((tid, pos))
    return per_frame


def _velocity_mps(
    p1: Point,
    p2: Point,
    f1: int,
    f2: int,
    fps: float,
    calibrations: Union[Calibration, Sequence[Calibration]],
) -> float:
    dt_frames = f2 - f1
    if dt_frames <= 0:
        return 0.0
    dt_s = dt_frames / fps
    cal = _cal_at_frame(calibrations, f1)
    return _pixel_dist_meters(p1, p2, cal) / dt_s if dt_s > 0 else 0.0


def _goal_direction_unit() -> Point:
    dx = float(settings.SHOT_GOAL_DIRECTION_X)
    dy = float(settings.SHOT_GOAL_DIRECTION_Y)
    norm = math.hypot(dx, dy)
    if norm <= 0.0:
        return (-1.0, 0.0)
    return (dx / norm, dy / norm)


def _project_delta(start: Point, end: Point, unit: Point) -> float:
    return (end[0] - start[0]) * unit[0] + (end[1] - start[1]) * unit[1]


@dataclass
class MotionEvidence:
    post_travel_px: float
    player_separation_px: Optional[float]
    followed_frames: int
    goalward_progress_px: float
    goalward_ratio: float
    relative_goalward_progress_px: Optional[float]


def _track_candidates(
    trajectories: Dict[int, List[Optional[Point]]],
    fps: float,
    calibrations: Union[Calibration, Sequence[Calibration]],
    high_mps: float,
) -> Iterable[ShotCandidate]:
    for tid, traj in trajectories.items():
        last_pos: Optional[Point] = None
        last_frame: Optional[int] = None
        for frame_idx, pos in enumerate(traj):
            if pos is None:
                continue
            if last_pos is not None and last_frame is not None:
                velocity = _velocity_mps(
                    last_pos, pos, last_frame, frame_idx, fps, calibrations
                )
                if velocity >= high_mps:
                    yield ShotCandidate(
                        frame_idx=frame_idx,
                        previous_frame_idx=last_frame,
                        ball_pos_px=pos,
                        previous_ball_pos_px=last_pos,
                        velocity_mps=velocity,
                        track_id=tid,
                        source="track",
                    )
            last_pos = pos
            last_frame = frame_idx


def _global_candidates(
    per_frame: List[List[Tuple[int, Point]]],
    fps: float,
    calibrations: Union[Calibration, Sequence[Calibration]],
    high_mps: float,
) -> Iterable[ShotCandidate]:
    for frame_idx in range(1, len(per_frame)):
        for tid, pos in per_frame[frame_idx]:
            best_dist = float("inf")
            best_prev: Optional[Tuple[int, Point, int]] = None
            for back in range(1, settings.SHOT_LOOKBACK_FRAMES + 1):
                prev_frame = frame_idx - back
                if prev_frame < 0:
                    break
                for prev_tid, prev_pos in per_frame[prev_frame]:
                    dist = euclidean(prev_pos, pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_prev = (prev_tid, prev_pos, prev_frame)
                if best_dist <= settings.SHOT_GLOBAL_PROXIMITY_RADIUS_PX:
                    break
            if best_prev is None or best_dist <= settings.SHOT_GLOBAL_PROXIMITY_RADIUS_PX:
                continue
            _prev_tid, prev_pos, prev_frame = best_prev
            velocity = _velocity_mps(prev_pos, pos, prev_frame, frame_idx, fps, calibrations)
            if velocity >= high_mps:
                yield ShotCandidate(
                    frame_idx=frame_idx,
                    previous_frame_idx=prev_frame,
                    ball_pos_px=pos,
                    previous_ball_pos_px=prev_pos,
                    velocity_mps=velocity,
                    track_id=tid,
                    source="global",
                )


def _follow_future_motion(
    candidate: ShotCandidate,
    per_frame: List[List[Tuple[int, Point]]],
    person_bboxes: Optional[Sequence[Optional[BBox]]],
) -> MotionEvidence:
    """Follow the closest plausible future ball and measure continued travel."""
    max_travel = euclidean(candidate.previous_ball_pos_px, candidate.ball_pos_px)
    goal_unit = _goal_direction_unit()
    max_separation: Optional[float] = None
    max_goalward_progress = max(
        0.0,
        _project_delta(
            candidate.previous_ball_pos_px,
            candidate.ball_pos_px,
            goal_unit,
        ),
    )
    max_goalward_ratio = (
        max_goalward_progress / max_travel if max_travel > 1e-6 else 0.0
    )
    max_relative_goalward: Optional[float] = None
    followed_frames = 0
    current_pos = candidate.ball_pos_px
    initial_px_per_frame = max_travel / max(1, candidate.frame_idx - candidate.previous_frame_idx)

    contact_player_ref = _bbox_foot_point(
        _bbox_at(person_bboxes, candidate.previous_frame_idx)
        or _bbox_at(person_bboxes, candidate.frame_idx)
    )

    for frame_idx in range(candidate.frame_idx, min(
        len(per_frame),
        candidate.frame_idx + settings.SHOT_LOOKAHEAD_FRAMES + 1,
    )):
        if not per_frame[frame_idx]:
            continue

        max_step = max(
            settings.SHOT_GLOBAL_PROXIMITY_RADIUS_PX,
            initial_px_per_frame * max(1, frame_idx - candidate.frame_idx + 1) * 1.8 + 40.0,
        )
        best_tid: Optional[int] = None
        best_pos: Optional[Point] = None
        best_dist = float("inf")
        for tid, pos in per_frame[frame_idx]:
            goalward_progress = _project_delta(
                candidate.previous_ball_pos_px,
                pos,
                goal_unit,
            )
            if (
                goalward_progress
                < max_goalward_progress - settings.SHOT_MAX_GOALWARD_REGRESSION_PX
            ):
                continue
            dist = euclidean(current_pos, pos)
            if dist < best_dist:
                best_tid = tid
                best_pos = pos
                best_dist = dist

        if best_pos is None or best_dist > max_step:
            continue
        if candidate.track_id is not None and best_tid == candidate.track_id:
            current_pos = best_pos
        else:
            current_pos = best_pos
        followed_frames += 1
        travel = euclidean(candidate.previous_ball_pos_px, current_pos)
        max_travel = max(max_travel, travel)

        goalward_progress = max(
            0.0,
            _project_delta(candidate.previous_ball_pos_px, current_pos, goal_unit),
        )
        max_goalward_progress = max(max_goalward_progress, goalward_progress)
        if travel > 1e-6:
            max_goalward_ratio = max(max_goalward_ratio, goalward_progress / travel)

        bbox = _bbox_at(person_bboxes, frame_idx)
        separation = _distance_point_to_bbox(current_pos, bbox)
        if separation is not None:
            max_separation = separation if max_separation is None else max(max_separation, separation)

        future_player_ref = _bbox_foot_point(bbox)
        if contact_player_ref is not None and future_player_ref is not None:
            player_goalward = max(
                0.0,
                _project_delta(contact_player_ref, future_player_ref, goal_unit),
            )
            relative_goalward = goalward_progress - player_goalward
            max_relative_goalward = (
                relative_goalward
                if max_relative_goalward is None
                else max(max_relative_goalward, relative_goalward)
            )

    return MotionEvidence(
        post_travel_px=max_travel,
        player_separation_px=max_separation,
        followed_frames=followed_frames,
        goalward_progress_px=max_goalward_progress,
        goalward_ratio=max_goalward_ratio,
        relative_goalward_progress_px=max_relative_goalward,
    )


def _score_candidate(
    candidate: ShotCandidate,
    per_frame: List[List[Tuple[int, Point]]],
    person_bboxes: Optional[Sequence[Optional[BBox]]],
    high_mps: float,
) -> ShotCandidate:
    contact_bbox = _bbox_at(person_bboxes, candidate.previous_frame_idx)
    if contact_bbox is None:
        contact_bbox = _bbox_at(person_bboxes, candidate.frame_idx)

    contact_dist = _distance_point_to_bbox(candidate.previous_ball_pos_px, contact_bbox)
    threshold = _contact_threshold_px(contact_bbox)
    motion = _follow_future_motion(
        candidate, per_frame, person_bboxes
    )

    contact_ok = contact_dist is None or contact_dist <= threshold
    initial_travel = euclidean(candidate.previous_ball_pos_px, candidate.ball_pos_px)
    initial_goalward = max(
        0.0,
        _project_delta(
            candidate.previous_ball_pos_px,
            candidate.ball_pos_px,
            _goal_direction_unit(),
        ),
    )
    initial_goalward_ratio = (
        initial_goalward / initial_travel if initial_travel > 1e-6 else 0.0
    )
    strong_release_ok = (
        candidate.velocity_mps >= settings.SHOT_STRONG_RELEASE_MPS
        and initial_goalward >= settings.SHOT_MIN_INITIAL_GOALWARD_PROGRESS_PX
        and initial_goalward_ratio >= settings.SHOT_MIN_GOALWARD_RATIO
        and (
            motion.followed_frames <= 2
            or motion.player_separation_px is None
            or motion.player_separation_px >= settings.SHOT_MIN_PLAYER_SEPARATION_PX
        )
    )
    travel_ok = (
        motion.post_travel_px >= settings.SHOT_MIN_POST_TRAVEL_PX
        or strong_release_ok
    )
    goalward_ok = (
        (
            motion.goalward_progress_px >= settings.SHOT_MIN_GOALWARD_PROGRESS_PX
            and motion.goalward_ratio >= settings.SHOT_MIN_GOALWARD_RATIO
        )
        or strong_release_ok
    )
    relative_release_ok = (
        motion.player_separation_px is not None
        and motion.relative_goalward_progress_px is not None
        and motion.player_separation_px >= settings.SHOT_MIN_RELATIVE_PLAYER_SEPARATION_PX
        and motion.relative_goalward_progress_px
        >= settings.SHOT_MIN_RELATIVE_GOALWARD_PROGRESS_PX
    )
    separation_ok = (
        motion.player_separation_px is None
        or motion.player_separation_px >= settings.SHOT_MIN_PLAYER_SEPARATION_PX
        or strong_release_ok
    )
    speed_ok = candidate.velocity_mps <= settings.SHOT_MAX_REASONABLE_MPS

    speed_score = min(candidate.velocity_mps / max(high_mps, 0.01), 2.0) / 2.0
    travel_score = min(motion.post_travel_px / settings.SHOT_MIN_POST_TRAVEL_PX, 1.0)
    goal_score = min(motion.goalward_progress_px / settings.SHOT_MIN_GOALWARD_PROGRESS_PX, 1.0)
    contact_score = 0.65 if contact_dist is None else max(0.0, 1.0 - contact_dist / threshold)
    sep_score = 0.65 if motion.player_separation_px is None else min(
        motion.player_separation_px / settings.SHOT_MIN_PLAYER_SEPARATION_PX, 1.0
    )
    rel_score = 0.65 if motion.relative_goalward_progress_px is None else min(
        max(0.0, motion.relative_goalward_progress_px)
        / settings.SHOT_MIN_RELATIVE_GOALWARD_PROGRESS_PX,
        1.0,
    )
    confidence = (
        speed_score * 0.25
        + travel_score * 0.20
        + contact_score * 0.15
        + sep_score * 0.15
        + goal_score * 0.15
        + rel_score * 0.10
    )

    candidate.contact_distance_px = contact_dist
    candidate.post_travel_px = motion.post_travel_px
    candidate.player_separation_px = motion.player_separation_px
    candidate.initial_goalward_progress_px = initial_goalward
    candidate.initial_goalward_ratio = initial_goalward_ratio
    candidate.goalward_progress_px = motion.goalward_progress_px
    candidate.goalward_ratio = motion.goalward_ratio
    candidate.relative_goalward_progress_px = motion.relative_goalward_progress_px
    candidate.confidence = confidence
    candidate.metrics = {
        "contact_threshold_px": round(threshold, 1) if threshold != float("inf") else None,
        "followed_future_frames": motion.followed_frames,
        "relative_release_ok": bool(relative_release_ok),
        "strong_release_ok": bool(strong_release_ok),
    }

    if not speed_ok:
        candidate.rejection_reason = "unrealistic_velocity"
    elif not contact_ok:
        candidate.rejection_reason = "ball_not_near_player_at_contact"
    elif not goalward_ok:
        candidate.rejection_reason = "not_goalward"
    elif not travel_ok:
        candidate.rejection_reason = "insufficient_post_contact_travel"
    elif not separation_ok:
        candidate.rejection_reason = "ball_stayed_with_player"
    else:
        candidate.accepted = True
        candidate.rejection_reason = ""
    return candidate


def _dedupe_accepted(candidates: List[ShotCandidate]) -> List[ShotCandidate]:
    accepted = sorted((c for c in candidates if c.accepted), key=lambda c: c.previous_frame_idx)
    deduped: List[ShotCandidate] = []
    for candidate in accepted:
        if (
            deduped
            and candidate.previous_frame_idx - deduped[-1].previous_frame_idx
            <= settings.SHOT_DEDUPE_WINDOW_FRAMES
        ):
            current_score = (candidate.confidence, candidate.velocity_mps, candidate.post_travel_px)
            kept_score = (
                deduped[-1].confidence,
                deduped[-1].velocity_mps,
                deduped[-1].post_travel_px,
            )
            if current_score > kept_score:
                deduped[-1].accepted = False
                deduped[-1].rejection_reason = "deduped_by_nearby_stronger_candidate"
                candidate.accepted = True
                deduped[-1] = candidate
            else:
                candidate.accepted = False
                candidate.rejection_reason = "deduped_by_nearby_stronger_candidate"
            continue
        deduped.append(candidate)
    return deduped


def detect_shots_from_tracks(
    trajectories: Dict[int, List[Optional[Point]]],
    fps: float,
    calibrations: Union[Calibration, Sequence[Calibration]],
    person_bboxes: Optional[Sequence[Optional[BBox]]] = None,
    high_mps: float = settings.SHOT_VELOCITY_HIGH_MPS,
) -> Tuple[List[ShotEvent], List[ShotCandidate]]:
    """Detect a variable number of true shots and return debug candidates."""
    per_frame = _build_per_frame(trajectories)
    if not per_frame:
        return [], []

    raw_candidates = list(_track_candidates(trajectories, fps, calibrations, high_mps))
    raw_candidates.extend(_global_candidates(per_frame, fps, calibrations, high_mps))
    raw_candidates.sort(key=lambda c: (c.previous_frame_idx, c.frame_idx, c.source))

    scored = [
        _score_candidate(candidate, per_frame, person_bboxes, high_mps)
        for candidate in raw_candidates
    ]
    accepted = _dedupe_accepted(scored)

    shots: List[ShotEvent] = []
    for index, candidate in enumerate(accepted, start=1):
        dt_frames = max(1, candidate.frame_idx - candidate.previous_frame_idx)
        dx = candidate.ball_pos_px[0] - candidate.previous_ball_pos_px[0]
        dy = candidate.ball_pos_px[1] - candidate.previous_ball_pos_px[1]
        velocity_vec = (dx / dt_frames, dy / dt_frames)
        shots.append(ShotEvent(
            index=index,
            frame_idx=candidate.previous_frame_idx,
            ball_pos_px=candidate.previous_ball_pos_px,
            velocity_mps=candidate.velocity_mps,
            track_id=candidate.track_id,
            confidence=candidate.confidence,
            source=candidate.source,
            debug=candidate.to_debug_dict(),
            velocity_vec_px_per_frame=velocity_vec,
        ))

    log.info(
        f"[scored] Accepted {len(shots)} shots from {len(scored)} candidates "
        f"across {len(trajectories)} tracks"
    )
    return shots, scored


def _detect_shots_single_track(
    trajectory: List[Optional[Point]],
    fps: float,
    cal: Calibration,
    high_mps: float,
    low_mps: float,
    track_id: Optional[int] = None,
) -> List[ShotEvent]:
    """Backward-compatible wrapper for older callers."""
    del low_mps
    shots, _ = detect_shots_from_tracks({track_id or 0: trajectory}, fps, cal, None, high_mps)
    return shots


def detect_shots_multitrack(
    trajectories: Dict[int, List[Optional[Point]]],
    fps: float,
    cal: Calibration,
    high_mps: float = settings.SHOT_VELOCITY_HIGH_MPS,
    low_mps: float = settings.SHOT_VELOCITY_LOW_MPS,
) -> List[ShotEvent]:
    del low_mps
    shots, _ = detect_shots_from_tracks(trajectories, fps, cal, None, high_mps)
    return shots


def detect_shots_global(
    trajectories: Dict[int, List[Optional[Point]]],
    fps: float,
    cal: Calibration,
    high_mps: float = settings.SHOT_VELOCITY_HIGH_MPS,
    low_mps: float = settings.SHOT_VELOCITY_LOW_MPS,
    lookback_frames: int = 10,
    proximity_radius_px: float = 80.0,
) -> List[ShotEvent]:
    del low_mps, lookback_frames, proximity_radius_px
    shots, _ = detect_shots_from_tracks(trajectories, fps, cal, None, high_mps)
    return [s for s in shots if s.source == "global"]


def merge_shot_events(
    per_track: List[ShotEvent],
    global_shots: List[ShotEvent],
    merge_window_frames: int = settings.SHOT_DEDUPE_WINDOW_FRAMES,
) -> List[ShotEvent]:
    """Backward-compatible dedupe for callers that still merge two lists."""
    combined = sorted(per_track + global_shots, key=lambda s: s.frame_idx)
    merged: List[ShotEvent] = []
    for shot in combined:
        if merged and shot.frame_idx - merged[-1].frame_idx <= merge_window_frames:
            if (shot.confidence, shot.velocity_mps) > (
                merged[-1].confidence,
                merged[-1].velocity_mps,
            ):
                merged[-1] = shot
            continue
        merged.append(shot)
    for index, shot in enumerate(merged, start=1):
        shot.index = index
    return merged


def detect_shots(
    trajectory: List[Optional[Point]],
    fps: float,
    cal: Calibration,
    high_mps: float = settings.SHOT_VELOCITY_HIGH_MPS,
    low_mps: float = settings.SHOT_VELOCITY_LOW_MPS,
) -> List[ShotEvent]:
    return _detect_shots_single_track(trajectory, fps, cal, high_mps, low_mps)
