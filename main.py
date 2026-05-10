"""ScoutAI Shooting Drill Analysis — entry point.

Two-pass pipeline:
    Pass 1: read whole video → calibrate (homography if cones form a quad,
            else 2-point linear scale) → per-track ball trajectories →
            write annotated video.
    Pass 2: for each detected shot, re-seek to that frame and run pose to
            determine which foot kicked.

Outputs: data/output/annotated.mp4 + data/output/report.json
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from analysis.event_detector import (
    detect_shots_from_tracks,
)
from analysis.shot_analyzer import determine_foot
from calibration.homography import (
    DegenerateCalibrationError,
    compute_homography,
)
from calibration.scale import compute_scale
from config import settings
from detection.cone_detector import ConeDetector
from detection.detector import Detector
from detection.pose_estimator import PoseEstimator
from detection.tracker import MultiBallTracker
from output.annotator import (
    draw_calibration_badge,
    draw_detections,
    draw_gate,
    draw_multi_ball_trails,
    draw_scale_segment,
    draw_shot_banner,
    draw_velocity_readout,
)
from output.report_generator import write_report
from output.video_writer import AnnotatedVideoWriter
from utils.logger import get_logger

log = get_logger("main")


# ── Calibration container ─────────────────────────────────────────────────────
@dataclass
class Calibration:
    method: str                                   # "yolo-world-homography" | "2-point-scale"
    H: Optional[np.ndarray] = None
    ordered_cones_px: Optional[np.ndarray] = None
    px_per_meter: Optional[float] = None
    scale_ref: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    raw_cones_px: List[Tuple[int, int]] = field(default_factory=list)

    def cal_value(self):
        """The thing event_detector / pixel_to_meters helpers consume."""
        return self.H if self.H is not None else self.px_per_meter


def _calibration_from_cones(cones: List[Tuple[int, int]]) -> Optional[Calibration]:
    """Try homography first; fall back to 2-point scale if degenerate."""
    if len(cones) < 2:
        return None
    if len(cones) >= 4:
        try:
            H, ordered = compute_homography(cones)
            return Calibration(
                method="yolo-world-homography",
                H=H,
                ordered_cones_px=ordered,
                raw_cones_px=cones,
            )
        except DegenerateCalibrationError as e:
            log.info(f"{e} — falling back to 2-point scale")
    px_per_meter, ref = compute_scale(cones)
    return Calibration(
        method="2-point-scale",
        px_per_meter=px_per_meter,
        scale_ref=ref,
        raw_cones_px=cones,
    )


def _save_debug_calibration(
    output_dir: str,
    frame: np.ndarray,
    raw_boxes: list,
    chosen: list,
) -> None:
    """Annotated calibration frame: every YOLO-World cone bbox + the chosen 4."""
    debug = frame.copy()
    for (x1, y1, x2, y2) in raw_boxes:
        cv2.rectangle(debug, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 255), 2)
    for (cx, cy) in chosen:
        cv2.circle(debug, (cx, cy), 14, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(output_dir, settings.DEBUG_CALIBRATION_FRAME_FILENAME), debug)
    log.info(f"Debug calibration image written to {output_dir}/"
             f"{settings.DEBUG_CALIBRATION_FRAME_FILENAME}")


def _initial_calibrate(
    cap: cv2.VideoCapture,
    cone_detector: "ConeDetector",
    output_dir: str,
) -> Optional[Calibration]:
    """Scan early frames until cone detection yields a usable calibration."""
    last_frame = None
    last_raw: list = []
    last_chosen: list = []
    cal: Optional[Calibration] = None

    for attempt in range(settings.CALIBRATION_MAX_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, attempt)
        ok, frame = cap.read()
        if not ok:
            break

        chosen, _candidates, raw_boxes = cone_detector.detect(frame)
        last_frame, last_raw, last_chosen = frame, raw_boxes, chosen

        cal = _calibration_from_cones(chosen)
        if cal is not None:
            log.info(f"Initial calibration on frame {attempt}: method={cal.method}")
            break

    if last_frame is not None:
        _save_debug_calibration(output_dir, last_frame, last_raw, last_chosen)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cal


def _draw_calibration_overlay(frame: np.ndarray, cal: Calibration) -> None:
    """Overlay the gate quad (homography) or the scale segment (2-point)."""
    if cal.method == "yolo-world-homography" and cal.ordered_cones_px is not None:
        draw_gate(frame, cal.ordered_cones_px)
    elif cal.method == "2-point-scale" and cal.scale_ref is not None:
        draw_scale_segment(frame, cal.scale_ref[0], cal.scale_ref[1],
                           cal.px_per_meter or 0.0)


def _ball_speed_mps(p1, p2, dt_s, cal_value) -> float:
    """Compute ball speed in m/s using whichever calibration is in effect."""
    from analysis.event_detector import _pixel_dist_meters
    return _pixel_dist_meters(p1, p2, cal_value) / dt_s if dt_s > 0 else 0.0


def run(video_path: str, output_dir: str) -> dict:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"Video: {width}x{height} @ {fps:.2f} fps")

    # ── Detectors ───────────────────────────────────────────────────────────
    detector = Detector()              # YOLOv8n (track): player + ball
    cone_detector = ConeDetector()     # YOLO-World: cones (zero-shot prompt)

    # ── Initial calibration ─────────────────────────────────────────────────
    cal = _initial_calibrate(cap, cone_detector, output_dir)
    if cal is None:
        log.warning("Calibration failed (need ≥2 cones). "
                    "Inspect data/output/debug_calibration.png to see what "
                    "YOLO-World matched. Try lowering CONE_YOLO_CONF_THRESHOLD "
                    "or adding prompts in config/settings.py.")
        cap.release()
        return {"error": "calibration_failed"}

    # ── Pass 1: multi-ball tracking, per-frame cone recal, annotated video ──
    tracker = MultiBallTracker()
    annotated_path = os.path.join(output_dir, settings.ANNOTATED_VIDEO_FILENAME)
    person_bboxes_per_frame: List[Optional[Tuple[float, float, float, float]]] = []
    calibrations_per_frame: List = []
    recal_count = 0
    prev_per_track: Dict[int, Tuple[float, float, int]] = {}  # tid → (x, y, frame_idx)

    with AnnotatedVideoWriter(annotated_path, fps, (width, height)) as writer:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Periodic recalibration to handle camera movement.
            if frame_idx > 0 and frame_idx % settings.CONE_RECAL_INTERVAL_FRAMES == 0:
                chosen, _, _ = cone_detector.detect(frame)
                new_cal = _calibration_from_cones(chosen)
                if new_cal is not None:
                    cal = new_cal
                    recal_count += 1

            calibrations_per_frame.append(cal.cal_value())

            dets = detector.detect(frame)
            per_track_balls = tracker.update(dets["balls"], frame=frame)

            # remember the most-confident person bbox for pass-2 pose
            if dets["persons"]:
                best_person = max(dets["persons"], key=lambda d: d.confidence)
                person_bboxes_per_frame.append(best_person.xyxy)
            else:
                person_bboxes_per_frame.append(None)

            # max ball speed across active tracks for the readout
            max_v: Optional[float] = None
            for tid, pos in per_track_balls.items():
                prev = prev_per_track.get(tid)
                if prev is not None:
                    dt_s = (frame_idx - prev[2]) / fps
                    v = _ball_speed_mps(
                        (prev[0], prev[1]), pos, dt_s, cal.cal_value()
                    )
                    if max_v is None or v > max_v:
                        max_v = v
                prev_per_track[tid] = (pos[0], pos[1], frame_idx)

            # ── annotate ────────────────────────────────────────────────────
            draw_detections(frame, dets["persons"], dets["balls"],
                            [tuple(map(int, c)) for c in cal.raw_cones_px])
            _draw_calibration_overlay(frame, cal)
            draw_multi_ball_trails(frame, tracker.all_recent_trails())
            draw_velocity_readout(frame, max_v)
            draw_calibration_badge(frame, cal.method)

            writer.write(frame)
            frame_idx += 1

    log.info(f"Pass 1 complete: {frame_idx} frames, {recal_count} recalibrations")
    log.info(f"Tracked {len(tracker.trajectories)} ball tracks")
    # Diagnostic: how many frames each track was alive
    for tid, traj in tracker.trajectories.items():
        seen = sum(1 for p in traj if p is not None)
        log.info(f"  track {tid}: detected in {seen}/{len(traj)} frames")

    # ── Detect variable number of true shots from scored kick evidence ─────
    shots, shot_candidates = detect_shots_from_tracks(
        tracker.trajectories,
        fps,
        calibrations_per_frame or cal.cal_value(),
        person_bboxes_per_frame,
    )

    # ── Pass 2: determine foot for each shot ────────────────────────────────
    pose = PoseEstimator()
    foot_per_shot: List[Optional[str]] = []
    try:
        for shot in shots:
            if shot.frame_idx >= len(person_bboxes_per_frame):
                log.warning(f"No person frame available at shot frame {shot.frame_idx}")
                foot_per_shot.append(None)
                continue
            person_bbox = person_bboxes_per_frame[shot.frame_idx]
            if person_bbox is None:
                log.warning(f"No person detected at shot frame {shot.frame_idx}")
                foot_per_shot.append(None)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, shot.frame_idx)
            ok, frame = cap.read()
            if not ok:
                foot_per_shot.append(None)
                continue
            foot = determine_foot(frame, person_bbox, shot.ball_pos_px, pose)
            log.info(f"Shot #{shot.index} (track {shot.track_id}): foot = {foot}")
            foot_per_shot.append(foot)
    finally:
        pose.close()
        cap.release()

    # ── Re-render annotated video with shot banners ─────────────────────────
    _overlay_shot_banners(annotated_path, shots, foot_per_shot, fps,
                          (width, height))

    # ── Write JSON report ───────────────────────────────────────────────────
    calibration_meta = {
        "method": cal.method,
        "cones_px": [list(map(int, c)) for c in cal.raw_cones_px],
        "gate_width_m": settings.GATE_WIDTH_M,
        "gate_depth_m": settings.GATE_DEPTH_M,
        "px_per_meter": cal.px_per_meter,
        "recalibrations": recal_count,
    }
    debug_payload = _write_debug_artifacts(output_dir, tracker, shot_candidates)
    report_path = os.path.join(output_dir, settings.REPORT_FILENAME)
    payload = write_report(
        shots=shots,
        foot_per_shot=foot_per_shot,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_idx,
        calibration=calibration_meta,
        output_path=report_path,
        debug=debug_payload,
    )
    log.info(f"Report written: {report_path}")
    log.info(f"Annotated video: {annotated_path}")
    return payload


def _write_debug_artifacts(output_dir: str, tracker: MultiBallTracker, shot_candidates: list) -> dict:
    """Write candidate-level diagnostics without changing the accepted shot API."""
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    candidate_dicts = [
        c.to_debug_dict()
        for c in sorted(shot_candidates, key=lambda item: (item.previous_frame_idx, item.frame_idx))
    ]
    track_summary = []
    for tid, traj in sorted(tracker.trajectories.items(), key=lambda item: item[0]):
        seen_frames = [idx for idx, pos in enumerate(traj) if pos is not None]
        track_summary.append({
            "track_id": tid,
            "detected_frames": len(seen_frames),
            "first_frame": seen_frames[0] if seen_frames else None,
            "last_frame": seen_frames[-1] if seen_frames else None,
        })

    debug_path = os.path.join(debug_dir, "shot_candidates.json")
    debug_file_payload = {
        "shot_candidates": candidate_dicts,
        "track_summary": track_summary,
    }
    with open(debug_path, "w") as f:
        json.dump(debug_file_payload, f, indent=2)

    max_inline = settings.SHOT_DEBUG_MAX_CANDIDATES
    return {
        "shot_candidates_path": debug_path,
        "shot_candidates_total": len(candidate_dicts),
        "shot_candidates_in_report": min(len(candidate_dicts), max_inline),
        "shot_candidates": candidate_dicts[:max_inline],
        "track_summary": track_summary,
    }


def _overlay_shot_banners(
    video_path: str,
    shots: list,
    foot_per_shot: list,
    fps: float,
    frame_size: Tuple[int, int],
) -> None:
    """Re-encode the annotated video with shot banners held for N frames."""
    if not shots:
        return
    cap = cv2.VideoCapture(video_path)
    tmp_path = video_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, frame_size)

    banner_frames = {}
    for shot, foot in zip(shots, foot_per_shot):
        for offset in range(settings.SHOT_BANNER_DURATION_FRAMES):
            banner_frames[shot.frame_idx + offset] = (shot, foot)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in banner_frames:
            shot, foot = banner_frames[idx]
            draw_shot_banner(frame, shot.index, foot, shot.velocity_mps)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    os.replace(tmp_path, video_path)


def parse_args():
    p = argparse.ArgumentParser(description="ScoutAI shooting drill analysis")
    p.add_argument("--video", required=True, help="Path to input drill video")
    p.add_argument("--output-dir", default=settings.OUTPUT_DIR,
                   help=f"Output directory (default: {settings.OUTPUT_DIR})")
    return p.parse_args()


def main():
    args = parse_args()
    run(args.video, args.output_dir)


if __name__ == "__main__":
    main()
