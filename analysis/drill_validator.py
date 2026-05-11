"""Validates the full 3-shot drill sequence: scoring, gate compliance, timing."""

from dataclasses import dataclass, field
from typing import List, Optional

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DrillSummary:
    shots_total: int
    shots_scored: int
    shots_missed: int
    outside_gate_count: int
    total_points: int
    shot_times_s: List[float]
    inter_shot_intervals_s: List[float]
    timing_errors: List[str]
    avg_velocity_mps: Optional[float]
    max_velocity_mps: Optional[float]
    drill_valid: bool
    validation_notes: List[str]
    per_shot: List[dict] = field(default_factory=list)


def validate_drill(
    shots: list,
    foot_per_shot: List[Optional[str]],
    fps: float,
    expected_shots: int = settings.DRILL_EXPECTED_SHOTS,
    max_interval_s: float = settings.DRILL_MAX_INTERVAL_S,
) -> DrillSummary:
    """Validate the drill and return a structured summary.

    drill_valid is True when:
    - exactly expected_shots were detected
    - no inter-shot interval exceeded max_interval_s
    - no shot was flagged as outside the gate
    """
    notes: List[str] = []
    timing_errors: List[str] = []

    shots_total = len(shots)
    if shots_total != expected_shots:
        notes.append(
            f"Expected {expected_shots} shots, detected {shots_total}."
        )

    shots_scored = sum(
        1 for s in shots if getattr(s, "scored", None) is True
    )
    shots_missed = sum(
        1 for s in shots if getattr(s, "scored", None) is False
    )
    outside_gate_count = sum(
        1 for s in shots if getattr(s, "outside_gate", None) is True
    )
    total_points = sum(
        getattr(s, "zone_points", None) or 0 for s in shots
    )

    shot_times_s = [round(s.frame_idx / fps, 3) for s in shots]
    inter_shot_intervals_s: List[float] = []
    for i in range(1, len(shot_times_s)):
        interval = round(shot_times_s[i] - shot_times_s[i - 1], 3)
        inter_shot_intervals_s.append(interval)
        if interval > max_interval_s:
            timing_errors.append(
                f"Interval between shots {i} and {i + 1} is "
                f"{interval:.1f}s (max {max_interval_s:.1f}s)."
            )

    velocities = [s.velocity_mps for s in shots if s.velocity_mps > 0]
    avg_v = round(sum(velocities) / len(velocities), 3) if velocities else None
    max_v = round(max(velocities), 3) if velocities else None

    if outside_gate_count > 0:
        notes.append(
            f"{outside_gate_count} shot(s) taken from outside the gate."
        )
    if timing_errors:
        notes.extend(timing_errors)

    drill_valid = (
        shots_total == expected_shots
        and not timing_errors
        and outside_gate_count == 0
    )

    foot_list = list(foot_per_shot) + [None] * max(0, shots_total - len(foot_per_shot))
    per_shot_dicts = []
    for shot, foot in zip(shots, foot_list):
        missed_m = getattr(shot, "missed_distance_m", None)
        per_shot_dicts.append({
            "index": shot.index,
            "time_s": round(shot.frame_idx / fps, 3),
            "velocity_mps": round(shot.velocity_mps, 3),
            "foot": foot,
            "scored": getattr(shot, "scored", None),
            "scoring_zone": getattr(shot, "scoring_zone", None),
            "zone_points": getattr(shot, "zone_points", None),
            "outside_gate": getattr(shot, "outside_gate", None),
            "missed_distance_m": round(missed_m, 2) if missed_m is not None else None,
        })

    log.info(
        f"[drill] valid={drill_valid} scored={shots_scored}/{shots_total} "
        f"points={total_points} outside_gate={outside_gate_count}"
    )

    return DrillSummary(
        shots_total=shots_total,
        shots_scored=shots_scored,
        shots_missed=shots_missed,
        outside_gate_count=outside_gate_count,
        total_points=total_points,
        shot_times_s=shot_times_s,
        inter_shot_intervals_s=inter_shot_intervals_s,
        timing_errors=timing_errors,
        avg_velocity_mps=avg_v,
        max_velocity_mps=max_v,
        drill_valid=drill_valid,
        validation_notes=notes,
        per_shot=per_shot_dicts,
    )
