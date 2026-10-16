"""
tracker.py — Robust Multi-Object License Plate Tracker with Motion Prediction
=============================================================================
Combines IoU matching with Centroid Distance fallback and linear velocity
prediction to continuously follow moving plates across frames without freezing.

Track lifecycle
---------------
NEW  →  TRACKING  →  LOST  →  (re-detected → TRACKING | max_lost → expired)
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from . import config
from .detector import Detection


@dataclass
class Candidate:
    frame_no:   int
    yolo_conf:  float
    crop_bgr:   np.ndarray      # BGR crop — NOT preprocessed yet
    timestamp:  float = field(default_factory=time.time)


class Track:
    """Represents one tracked license plate with continuous motion estimation."""

    _id_counter = 0

    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    def __init__(self, det: Detection, crop: np.ndarray, frame_no: int):
        self.track_id:        int = Track._next_id()
        self.bbox:            tuple = (det.x1, det.y1, det.x2, det.y2)
        self.confidence:      float = det.confidence
        self.lost_count:      int   = 0
        self.state:           str   = "TRACKING"   # TRACKING | LOST
        self.frame_created:   int = frame_no
        self.last_seen_frame: int = frame_no
        self._window_start:   float = time.time()

        # Motion estimation (center & velocity)
        self.width:  float = float(max(2, det.x2 - det.x1))
        self.height: float = float(max(2, det.y2 - det.y1))
        self.cx:     float = float(det.x1 + det.x2) / 2.0
        self.cy:     float = float(det.y1 + det.y2) / 2.0
        self.vx:     float = 0.0
        self.vy:     float = 0.0

        # Candidate pool (bounded)
        self._candidates: list[Candidate] = []

        # Best-frame OCR state
        self.ocr_queued:   bool  = False    # submitted to OCR worker?
        self.ocr_done:     bool  = False    # result available?
        self.ocr_raw:      str   = ""
        self.ocr_final:    str   = ""
        self.ocr_conf:     float = 0.0
        self.ocr_history:  deque = deque(maxlen=10)  # temporal voter

        self._add_candidate(det, crop, frame_no)

    # ── Motion Prediction ─────────────────────────────────────────────────────

    def predict(self):
        """Advance the bounding box by the current velocity vector."""
        if self.lost_count > 0:
            # Dampen velocity when track is temporarily unobserved
            self.vx *= 0.80
            self.vy *= 0.80

        self.cx += self.vx
        self.cy += self.vy

        hw = self.width / 2.0
        hh = self.height / 2.0
        self.bbox = (
            int(round(self.cx - hw)),
            int(round(self.cy - hh)),
            int(round(self.cx + hw)),
            int(round(self.cy + hh))
        )

    # ── Candidate management ──────────────────────────────────────────────────

    def _add_candidate(self, det: Detection, crop: np.ndarray, frame_no: int):
        if crop is None or crop.size == 0:
            return

        cand = Candidate(frame_no=frame_no, yolo_conf=det.confidence,
                         crop_bgr=crop.copy())

        if len(self._candidates) < config.MAX_CANDIDATES_PER_TRACK:
            self._candidates.append(cand)
        else:
            # Replace worst candidate if this one is better
            worst_idx = min(range(len(self._candidates)),
                            key=lambda i: self._candidates[i].yolo_conf)
            if cand.yolo_conf > self._candidates[worst_idx].yolo_conf:
                self._candidates[worst_idx] = cand

    def get_best_candidate(self) -> Optional[Candidate]:
        if not self._candidates:
            return None
        return max(self._candidates, key=lambda c: c.yolo_conf)

    def should_trigger_ocr(self) -> bool:
        """
        OCR is triggered when:
        - candidates are available AND
        - NOT already queued/done AND
        - (capture window expired OR pool full)
        """
        if self.ocr_queued or self.ocr_done:
            return False
        if not self._candidates:
            return False
        window_elapsed = (time.time() - self._window_start) >= config.CAPTURE_WINDOW_SECONDS
        pool_full      = len(self._candidates) >= config.MAX_CANDIDATES_PER_TRACK
        return window_elapsed or pool_full

    # ── OCR result / temporal voting ──────────────────────────────────────────

    def update_ocr_result(self, raw_text: str, final_text: str, conf: float):
        self.ocr_raw   = raw_text
        self.ocr_conf  = conf
        self.ocr_done  = True

        if final_text and final_text not in ("LOW_QUALITY_PLATE", "UNREADABLE_PLATE"):
            self.ocr_history.append((final_text, conf))

        self.ocr_final = self._voted_result()

    def _voted_result(self) -> str:
        """Choose the most frequent (and highest-confidence on tie) OCR reading."""
        if not self.ocr_history:
            return self.ocr_raw
        counts: dict[str, list] = {}
        for text, c in self.ocr_history:
            counts.setdefault(text, []).append(c)
        best = max(counts, key=lambda t: (len(counts[t]),
                                          sum(counts[t]) / len(counts[t])))
        return best

    # ── Update from new detection ─────────────────────────────────────────────

    def update(self, det: Detection, crop: np.ndarray, frame_no: int):
        new_cx = float(det.x1 + det.x2) / 2.0
        new_cy = float(det.y1 + det.y2) / 2.0
        new_w  = float(det.x2 - det.x1)
        new_h  = float(det.y2 - det.y1)

        # Estimate instantaneous velocity with exponential smoothing (alpha = 0.6)
        inst_vx = new_cx - self.cx
        inst_vy = new_cy - self.cy
        self.vx = 0.6 * inst_vx + 0.4 * self.vx
        self.vy = 0.6 * inst_vy + 0.4 * self.vy

        self.cx = new_cx
        self.cy = new_cy
        self.width  = 0.7 * new_w + 0.3 * self.width
        self.height = 0.7 * new_h + 0.3 * self.height

        self.bbox            = (det.x1, det.y1, det.x2, det.y2)
        self.confidence      = det.confidence
        self.lost_count      = 0
        self.state           = "TRACKING"
        self.last_seen_frame = frame_no
        self._add_candidate(det, crop, frame_no)

    def mark_lost(self):
        self.lost_count += 1
        self.state = "LOST"
        # Advance position along predicted trajectory
        self.predict()

    @property
    def is_expired(self) -> bool:
        return self.lost_count > config.MAX_LOST_FRAMES

    @property
    def should_draw(self) -> bool:
        """Only draw active tracks or brief 3-frame occlusions (never draw long stale boxes)."""
        return self.state == "TRACKING" or self.lost_count <= 3

    @property
    def display_text(self) -> str:
        if self.ocr_final:
            return self.ocr_final
        if self.ocr_raw and self.ocr_raw not in ("LOW_QUALITY_PLATE", "UNREADABLE_PLATE"):
            return self.ocr_raw
        if self.ocr_queued:
            return "OCR PENDING..."
        return "TRACKING..."


class Tracker:
    """
    Continuous Multi-Object Tracker for License Plates.

    Combines:
      1. Forward velocity prediction.
      2. IoU greedy association.
      3. Centroid distance matching fallback for fast-moving plates.
      4. Dynamic lifecycle management (creates, updates, expires).
    """

    def __init__(self):
        self._tracks: list[Track] = []

    @property
    def tracks(self) -> list[Track]:
        return self._tracks

    def step_prediction(self) -> list[Track]:
        """Advance all active tracks using velocity prediction when YOLO is skipped."""
        for track in self._tracks:
            track.predict()
        return [t for t in self._tracks if t.should_draw]

    def update(self, detections: list[Detection],
               crops: list[np.ndarray],
               frame_no: int) -> tuple[list[Track], float]:
        """
        Matches detections to tracks and updates motion state.
        Returns (active_tracks, tracking_time_ms).
        """
        t0 = time.perf_counter()

        # Step 0: Advance all tracks by velocity prediction prior to matching
        for track in self._tracks:
            track.predict()

        matched_track_idx  = set()
        matched_detect_idx = set()

        if self._tracks and detections:
            # ── Step 1: Primary IoU Matching ──────────────────────────────
            iou_matrix = np.zeros((len(self._tracks), len(detections)))
            for ti, track in enumerate(self._tracks):
                t_det = Detection(
                    x1=track.bbox[0], y1=track.bbox[1],
                    x2=track.bbox[2], y2=track.bbox[3],
                    confidence=track.confidence,
                    cx=int(round(track.cx)),
                    cy=int(round(track.cy)),
                )
                for di, det in enumerate(detections):
                    iou_matrix[ti, di] = t_det.iou(det)

            # Greedy matching by highest IoU
            while True:
                max_val = np.max(iou_matrix)
                if max_val < config.IOU_MATCH_THRESHOLD:
                    break
                ti, di = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                self._tracks[ti].update(detections[di], crops[di], frame_no)
                matched_track_idx.add(ti)
                matched_detect_idx.add(di)
                iou_matrix[ti, :] = -1
                iou_matrix[:, di] = -1

            # ── Step 2: Centroid Distance Matching Fallback ───────────────
            # Handles fast-moving vehicles where IoU drops below threshold
            unmatched_tracks = [ti for ti in range(len(self._tracks)) if ti not in matched_track_idx]
            unmatched_dets   = [di for di in range(len(detections)) if di not in matched_detect_idx]

            if unmatched_tracks and unmatched_dets:
                dist_matrix = np.full((len(unmatched_tracks), len(unmatched_dets)), 1e6)
                for i, ti in enumerate(unmatched_tracks):
                    track = self._tracks[ti]
                    for j, di in enumerate(unmatched_dets):
                        det = detections[di]
                        dx = track.cx - det.cx
                        dy = track.cy - det.cy
                        dist = np.hypot(dx, dy)
                        max_allowed = max(track.width * 1.5, config.CENTROID_DISTANCE_THRESHOLD)
                        if dist <= max_allowed:
                            dist_matrix[i, j] = dist

                while True:
                    min_dist = np.min(dist_matrix)
                    if min_dist >= 1e5:
                        break
                    i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                    ti = unmatched_tracks[i]
                    di = unmatched_dets[j]
                    self._tracks[ti].update(detections[di], crops[di], frame_no)
                    matched_track_idx.add(ti)
                    matched_detect_idx.add(di)
                    dist_matrix[i, :] = 1e6
                    dist_matrix[:, j] = 1e6

        # ── Step 3: Mark unmatched tracks as LOST ─────────────────────────
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_idx:
                track.mark_lost()

        # ── Step 4: Create new tracks for unmatched detections ────────────
        for di, (det, crop) in enumerate(zip(detections, crops)):
            if di not in matched_detect_idx:
                self._tracks.append(Track(det, crop, frame_no))

        # ── Step 5: Prune expired tracks ─────────────────────────────────
        self._tracks = [t for t in self._tracks if not t.is_expired]

        track_ms = (time.perf_counter() - t0) * 1000.0
        return self._tracks, track_ms
