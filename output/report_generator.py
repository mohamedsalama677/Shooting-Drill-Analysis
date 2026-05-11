"""JSON drill report — one entry per feature from arch.md."""

import json
from typing import List, Optional

from analysis.event_detector import ShotEvent
from config import settings


def write_report(
    shots: List[ShotEvent],
    foot_per_shot: List[Optional[str]],
    fps: float,
    width: int,
    height: int,
    frame_count: int,
    calibration: dict,
    output_path: str,
    drill_summary: Optional[dict] = None,
    debug: Optional[dict] = None,
) -> dict:
    """Write the drill report containing only the six arch.md features.

    Internal implementation details (track_id, confidence, ball_pos_px, etc.)
    are kept in code but excluded from the report output.
    The `debug` parameter is accepted for backward compatibility but not written.
    """
    duration_s = round(frame_count / fps, 3) if fps > 0 else None

    payload = {
        "drill": "shooting",
        "video": {
            "fps": round(float(fps), 3),
            "width": int(width),
            "height": int(height),
            "duration_s": duration_s,
        },
        # Feature 1 — Distance Calibration
        "calibration": {
            "method": calibration.get("method"),
            "gate_width_m": calibration.get("gate_width_m", settings.GATE_WIDTH_M),
            "gate_depth_m": calibration.get("gate_depth_m", settings.GATE_DEPTH_M),
            "px_per_meter": (
                round(calibration["px_per_meter"], 3)
                if calibration.get("px_per_meter") is not None else None
            ),
            "recalibrations": int(calibration.get("recalibrations", 0)),
        },
        "shots_detected": len(shots),
        "shots": [
            {
                "index": shot.index,
                "time_s": round(shot.frame_idx / fps, 3),
                # Feature 2 — Shot detection / shot power
                "velocity_mps": round(shot.velocity_mps, 3),
                # Feature 3 — Foot used
                "foot": foot,
                # Feature 5 — Shot from outside gate
                "outside_gate": getattr(shot, "outside_gate", None),
                # Feature 4 — Goal detection & scoring zone
                "scored": getattr(shot, "scored", None),
                "scoring_zone": getattr(shot, "scoring_zone", None),
                "zone_points": getattr(shot, "zone_points", None),
                # Feature 6 — Missed shot distance
                "missed_distance_m": (
                    round(shot.missed_distance_m, 2)
                    if getattr(shot, "missed_distance_m", None) is not None else None
                ),
            }
            for shot, foot in zip(shots, foot_per_shot)
        ],
    }

    if drill_summary is not None:
        payload["drill_summary"] = drill_summary

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload
