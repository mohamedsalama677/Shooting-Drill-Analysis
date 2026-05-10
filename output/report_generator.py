"""Combined JSON drill report — calibration metadata + multi-track shots + foot used."""

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
    debug: Optional[dict] = None,
) -> dict:
    """Serialize one combined report. Returns the dict for logging convenience.

    `calibration` should be a dict with keys: method (e.g. "yolo-world-homography"
    or "2-point-scale"), cones_px, gate_width_m, gate_depth_m, px_per_meter,
    recalibrations.
    """
    payload = {
        "drill": "shooting",
        "video": {
            "fps": round(float(fps), 3),
            "width": int(width),
            "height": int(height),
            "frame_count": int(frame_count),
        },
        "calibration": {
            "method": calibration.get("method"),
            "cones_px": [list(map(int, pt)) for pt in calibration.get("cones_px", [])],
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
                "frame": shot.frame_idx,
                "time_s": round(shot.frame_idx / fps, 3),
                "velocity_mps": round(shot.velocity_mps, 3),
                "foot": foot,
                "track_id": shot.track_id,
                "ball_pos_px": [int(shot.ball_pos_px[0]), int(shot.ball_pos_px[1])],
                "confidence": round(float(getattr(shot, "confidence", 0.0)), 3),
                "source": getattr(shot, "source", None),
            }
            for shot, foot in zip(shots, foot_per_shot)
        ],
    }
    if debug is not None:
        payload["debug"] = debug
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload
