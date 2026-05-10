"""Multi-ball tracker - stores ball trajectories from one YOLO tracking pass."""

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import settings
from utils.geometry import bbox_center, euclidean
from utils.logger import get_logger

log = get_logger(__name__)

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


class MultiBallTracker:
    """Stores one trajectory per ball track.

    The detector owns the single YOLO/BoT-SORT call for each frame. This class
    only records the tracked ball detections and performs conservative short-gap
    template recovery when YOLO briefly loses a blurred ball.
    """

    def __init__(self, model=None, max_trail: int = settings.BALL_TRAIL_LENGTH):
        self.model = model
        self.trajectories: Dict[int, List[Optional[Point]]] = {}
        self.bboxes: Dict[int, List[Optional[BBox]]] = {}
        self.recent_per_track: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=max_trail)
        )
        self.last_seen: Dict[int, Tuple[Point, BBox, int]] = {}
        self.templates: Dict[int, np.ndarray] = {}
        self.frame_count = 0
        self._next_fallback_id = -1

    def update(self, balls: Sequence, frame: Optional[np.ndarray] = None) -> Dict[int, Point]:
        """Record ball detections from the shared detector pass.

        Calling YOLO tracking again here would advance the tracker twice for the
        same frame and can create unstable IDs. Detections without a YOLO ID are
        attached to the nearest recent track, or assigned a negative fallback ID.
        """
        per_track: Dict[int, Point] = {}
        per_bbox: Dict[int, BBox] = {}

        for det in balls:
            tid = det.track_id
            pos = bbox_center(det.xyxy)
            if tid is None:
                tid = self._fallback_track_id(pos, set(per_track))
            per_track[tid] = pos
            per_bbox[tid] = det.xyxy

        if frame is not None:
            recovered = self._recover_missing_tracks(frame, set(per_track))
            for tid, pos in recovered.items():
                per_track[tid] = pos
                last = self.last_seen.get(tid)
                if last is not None:
                    per_bbox[tid] = last[1]

        all_known_ids = set(self.trajectories.keys()) | set(per_track.keys())
        for tid in all_known_ids:
            if tid not in self.trajectories:
                self.trajectories[tid] = [None] * self.frame_count
                self.bboxes[tid] = [None] * self.frame_count

            self.trajectories[tid].append(per_track.get(tid))
            self.bboxes[tid].append(per_bbox.get(tid))

            if tid in per_track:
                pos = per_track[tid]
                self.recent_per_track[tid].append((int(pos[0]), int(pos[1])))
                if tid in per_bbox:
                    bbox = per_bbox[tid]
                    self.last_seen[tid] = (pos, bbox, self.frame_count)
                    if frame is not None:
                        template = self._crop_template(frame, bbox)
                        if template is not None:
                            self.templates[tid] = template

        self.frame_count += 1
        return per_track

    def all_recent_trails(self) -> Dict[int, Deque[Tuple[int, int]]]:
        """For drawing per-ball trails."""
        return self.recent_per_track

    def _fallback_track_id(self, pos: Point, already_used: set[int]) -> int:
        best_tid: Optional[int] = None
        best_dist = float("inf")
        for tid, (last_pos, _bbox, last_frame) in self.last_seen.items():
            if tid in already_used:
                continue
            if self.frame_count - last_frame > settings.BALL_RECOVERY_MAX_GAP_FRAMES:
                continue
            dist = euclidean(pos, last_pos)
            if dist < best_dist:
                best_tid = tid
                best_dist = dist

        if best_tid is not None and best_dist <= settings.BALL_RECOVERY_SEARCH_RADIUS_PX:
            return best_tid

        tid = self._next_fallback_id
        self._next_fallback_id -= 1
        return tid

    def _crop_template(self, frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = max(4, int(max(x2 - x1, y2 - y1) * 0.35))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    def _recover_missing_tracks(
        self,
        frame: np.ndarray,
        active_ids: set[int],
    ) -> Dict[int, Point]:
        recovered: Dict[int, Point] = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        for tid, (last_pos, _bbox, last_frame) in self.last_seen.items():
            if tid in active_ids:
                continue
            gap = self.frame_count - last_frame
            if gap <= 0 or gap > settings.BALL_RECOVERY_MAX_GAP_FRAMES:
                continue
            template = self.templates.get(tid)
            if template is None:
                continue

            th, tw = template.shape[:2]
            radius = (
                settings.BALL_RECOVERY_SEARCH_RADIUS_PX
                + settings.BALL_RECOVERY_SEARCH_GROWTH_PX * max(0, gap - 1)
            )
            x1 = max(0, int(last_pos[0] - radius))
            y1 = max(0, int(last_pos[1] - radius))
            x2 = min(w, int(last_pos[0] + radius))
            y2 = min(h, int(last_pos[1] + radius))
            if x2 - x1 < tw or y2 - y1 < th:
                continue

            search = gray[y1:y2, x1:x2]
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if score < settings.BALL_RECOVERY_MIN_SCORE:
                continue

            cx = x1 + loc[0] + tw / 2.0
            cy = y1 + loc[1] + th / 2.0
            if abs(cy - last_pos[1]) > settings.BALL_RECOVERY_MAX_VERTICAL_JUMP_PX * gap:
                continue
            if any(euclidean((cx, cy), p) < 24.0 for p in recovered.values()):
                continue

            recovered[tid] = (cx, cy)
            log.debug(
                f"Recovered ball track {tid} at frame {self.frame_count} "
                f"(template score {score:.2f})"
            )

        return recovered
