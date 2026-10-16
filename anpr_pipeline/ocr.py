"""
ocr.py — Async background PaddleOCR worker
===========================================
• Runs PaddleOCR in a dedicated daemon thread.
• Main video loop NEVER blocks waiting for OCR.
• Accepts plate crops via a bounded queue (drops excess to stay real-time).
• Calls the IMMUTABLE preprocessing.preprocess_plate() before OCR.
• Writes results back into the Track object directly.

OCR pipeline per job:
    plate_crop (BGR)
         ↓
    preprocessing.preprocess_plate()      ← UNCHANGED BLACK BOX
         ↓
    PaddleOCR
         ↓
    validate_and_clean_indian_plate()
         ↓
    track.update_ocr_result()
"""

import os
import re
import time
import queue
import threading
import numpy as np
from typing import Optional

from . import config

# ── PaddleOCR environment flags (must be set before import) ───────────────────
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"]     = "0"


# ── Plate text helpers (copied verbatim from app.py) ─────────────────────────
_INDIAN_STATES = {
    "AN","AP","AR","AS","BR","CH","CG","DD","DL","DN","GA","GJ","HP",
    "HR","JH","JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL",
    "OD","OR","PB","PY","RJ","SK","TN","TR","TS","UK","UA","UP","WB",
}

_CHAR_TO_NUM = {'O':'0','Q':'0','D':'0','I':'1','L':'1','T':'7',
                'Z':'2','S':'5','B':'8','G':'6'}
_NUM_TO_CHAR = {'0':'O','1':'I','2':'Z','5':'S','8':'B','6':'G','4':'A'}


def _clean(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', text).upper()


def _format_indian_plate(raw: str) -> str:
    """Positional character correction for Indian plate format. KEEP AS-IS."""
    clean = _clean(raw)
    if len(clean) < 5:
        return clean

    chars = list(clean)
    n = len(chars)

    for k in range(min(2, n)):          # positions 0-1: must be letters
        if chars[k].isdigit() and chars[k] in _NUM_TO_CHAR:
            chars[k] = _NUM_TO_CHAR[chars[k]]

    for k in range(2, min(4, n)):       # positions 2-3: must be digits
        if chars[k].isalpha() and chars[k] in _CHAR_TO_NUM:
            chars[k] = _CHAR_TO_NUM[chars[k]]

    if n >= 8:                          # last 4: must be digits
        for k in range(n - 4, n):
            if chars[k].isalpha() and chars[k] in _CHAR_TO_NUM:
                chars[k] = _CHAR_TO_NUM[chars[k]]

    return "".join(chars)


def _parse_paddle_result(p_res) -> list:
    """Handles PaddleOCR 3.x dict format and classic 2.x list format."""
    boxes = []
    if not p_res:
        return boxes
    for entry in p_res:
        if isinstance(entry, dict):
            texts  = entry.get('rec_texts', [])
            scores = entry.get('rec_scores', [])
            polys  = entry.get('dt_polys', entry.get('rec_polys', []))
            for i, txt in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 0.9
                bbox  = polys[i]  if i < len(polys)  else [[0,0],[10,0],[10,10],[0,10]]
                boxes.append((bbox, str(txt), score))
        elif isinstance(entry, (list, tuple)):
            for line in entry:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    bbox = line[0]
                    if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                        boxes.append((bbox, str(line[1][0]), float(line[1][1])))
                    else:
                        boxes.append((bbox, str(line[1]), 0.9))
    return boxes


def _validate_plate(ocr_boxes, crop_shape=None):
    """Filter noise boxes, sort L→R, validate Indian plate. Returns dict."""
    if not ocr_boxes:
        return {"valid": False, "plate_text": "", "raw_ocr_text": "",
                "confidence": 0.0, "post_processed": False}

    crop_h = crop_shape[0] if crop_shape else 100
    valid  = []

    for bbox, text, prob in ocr_boxes:
        t = text.strip().upper()
        if t in ("IND", "INDIA", "IN", "IIND"):
            continue
        pts   = np.array(bbox, dtype=np.int32)
        min_x = int(np.min(pts[:, 0]))
        box_h = int(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        if box_h < 0.15 * crop_h:
            continue
        valid.append((min_x, t, prob, bbox))

    if not valid:
        return {"valid": False, "plate_text": "", "raw_ocr_text": "",
                "confidence": 0.0, "post_processed": False}

    valid.sort(key=lambda b: b[0])
    raw       = "".join(b[1] for b in valid)
    avg_conf  = sum(b[2] for b in valid) / len(valid)
    formatted = _format_indian_plate(raw)

    prefix    = formatted[:2] if len(formatted) >= 2 else ""
    is_valid  = prefix in _INDIAN_STATES and (6 <= len(formatted) <= 11)

    return {
        "valid":          is_valid,
        "plate_text":     formatted,
        "raw_ocr_text":   raw,
        "confidence":     round(avg_conf, 3),
        "post_processed": formatted != _clean(raw),
    }


# ─────────────────────────────────────────────────────────────────────────────
# OCR JOB dataclass
# ─────────────────────────────────────────────────────────────────────────────

class OCRJob:
    __slots__ = ('track_id', 'best_crop', 'yolo_conf', 'track_ref')

    def __init__(self, track_id, best_crop, yolo_conf, track_ref):
        self.track_id  = track_id
        self.best_crop = best_crop    # BGR crop (BEFORE preprocessing)
        self.yolo_conf = yolo_conf    # for logging only
        self.track_ref = track_ref    # weak reference to Track object


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC OCR WORKER
# ─────────────────────────────────────────────────────────────────────────────

class AsyncOCRWorker:
    """
    Daemon thread that processes OCR jobs without blocking the capture loop.

    Preprocessing pipeline → PaddleOCR calls happen here, NOT in main thread.
    """

    def __init__(self):
        self._queue:   queue.Queue  = queue.Queue(maxsize=config.MAX_OCR_QUEUE_SIZE)
        self._running: bool         = False
        self._engine               = None        # PaddleOCR instance
        self._thread: Optional[threading.Thread] = None

        # Performance counters
        self.total_jobs   = 0
        self.total_ms     = 0.0
        self.last_ocr_ms  = 0.0

    def start(self):
        """Initialize PaddleOCR and start the background thread."""
        try:
            from paddleocr import PaddleOCR
            print("[OCR] Initializing PaddleOCR engine...")
            self._engine = PaddleOCR(lang='en', enable_mkldnn=False)
            print("[OCR] PaddleOCR ready.")
        except Exception as e:
            print(f"[OCR] ⚠  PaddleOCR init failed: {e}  — OCR disabled.")
            self._engine = None

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="OCRWorker")
        self._thread.start()
        return self

    def submit(self, job: OCRJob) -> bool:
        """
        Non-blocking enqueue.  Drops job if queue is full (keeps pipeline fast).
        Returns True if accepted.
        """
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            return False

    def _loop(self):
        from .preprocessing import preprocess_plate   # IMMUTABLE black box
        MIN_VALID = 0.20
        MIN_UNFMT = 0.40

        while self._running:
            try:
                job: OCRJob = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                # ── Step 1: Preprocessing (UNCHANGED) ────────────────────
                preprocessed, quality = preprocess_plate(job.best_crop)

                if preprocessed is None:
                    job.track_ref.update_ocr_result(
                        quality["status"], quality["status"], 0.0)
                    self._queue.task_done()
                    continue

                raw_text  = ""
                final_text = ""
                ocr_conf   = 0.0

                # ── Step 2: PaddleOCR ─────────────────────────────────────
                if self._engine is not None:
                    try:
                        p_res    = self._engine.ocr(preprocessed)
                        ocr_boxes = _parse_paddle_result(p_res)
                    except Exception as e:
                        print(f"[OCR] Inference error: {e}")
                        ocr_boxes = []
                else:
                    ocr_boxes = []

                # ── Step 3: Validate & clean ──────────────────────────────
                result = _validate_plate(ocr_boxes, preprocessed.shape)
                raw_text  = result["raw_ocr_text"]
                ocr_conf  = result["confidence"]

                if result["valid"] and ocr_conf >= MIN_VALID:
                    final_text = result["plate_text"]
                elif result["plate_text"] and len(result["plate_text"]) >= 6 \
                        and ocr_conf >= MIN_UNFMT:
                    final_text = result["plate_text"]
                else:
                    final_text = "UNREADABLE_PLATE"

                job.track_ref.update_ocr_result(raw_text, final_text, ocr_conf)
                print(f"[OCR] Track {job.track_id} | YOLO conf {job.yolo_conf:.2f} "
                      f"| raw: '{raw_text}' | final: '{final_text}' | ocr_conf: {ocr_conf:.2f}")

            except Exception as e:
                print(f"[OCR] Unhandled error for track {job.track_id}: {e}")
            finally:
                elapsed = (time.perf_counter() - t0) * 1000.0
                self.last_ocr_ms  = elapsed
                self.total_ms    += elapsed
                self.total_jobs  += 1
                try:
                    self._queue.task_done()
                except Exception:
                    pass

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def avg_ocr_ms(self) -> float:
        if self.total_jobs == 0:
            return 0.0
        return self.total_ms / self.total_jobs
