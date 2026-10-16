"""
main.py — Entry point for the ANPR real-time pipeline
======================================================

Run:
    python main.py                      # default camera (index 0)
    python main.py --camera 1           # specify camera index
    python main.py --list-cameras       # scan and list all available cameras
    python main.py --imgsz 480          # faster but slightly less accurate
    python main.py --conf 0.40          # adjust detection threshold

Keyboard controls (while window is open):
    Q / ESC  — Quit and print performance summary
    S        — Save snapshot to captures/
    P        — Print live profiling to terminal
"""

import argparse
import os
import sys
import time
from collections import deque

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np

# Ensure this file can be run from any working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from anpr_pipeline import config
from anpr_pipeline.camera import ThreadedCamera, list_available_cameras
from anpr_pipeline.detector import PlateDetector
from anpr_pipeline.tracker import Tracker
from anpr_pipeline.ocr import AsyncOCRWorker, OCRJob


# ── Colour palette ────────────────────────────────────────────────────────────
COL_GREEN  = (0, 255, 128)
COL_CYAN   = (56, 189, 248)
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0, 0, 0)
COL_AMBER  = (0, 200, 255)
COL_RED    = (0, 0, 255)
COL_DARK   = (15, 20, 30)


# ── Profiler ──────────────────────────────────────────────────────────────────
class Profiler:
    def __init__(self, n: int = 30):
        self._n = n
        self.capture  = deque(maxlen=n)
        self.yolo     = deque(maxlen=n)
        self.tracking = deque(maxlen=n)
        self.render   = deque(maxlen=n)
        self.total    = deque(maxlen=n)

    def _avg(self, d): return sum(d) / len(d) if d else 0.0
    @property
    def fps(self): return 1000.0 / self._avg(self.total) if self._avg(self.total) else 0.0
    @property
    def min_fps(self): return 1000.0 / max(self.total) if self.total else 0.0
    @property
    def max_fps(self): return 1000.0 / min(self.total) if self.total else 0.0

    def print_summary(self, ocr_worker: AsyncOCRWorker):
        print("\n" + "="*62)
        print("         ANPR CPU PIPELINE  -  PERFORMANCE REPORT")
        print("="*62)
        print(f"  Avg FPS          : {self.fps:.1f}")
        print(f"  Min FPS (worst)  : {self.min_fps:.1f}")
        print(f"  Max FPS (best)   : {self.max_fps:.1f}")
        print("-" * 62)
        print(f"  Frame Capture    : {self._avg(self.capture):.1f} ms")
        print(f"  YOLO Inference   : {self._avg(self.yolo):.1f} ms   "
              f"(~{1000/self._avg(self.yolo):.0f} FPS pure)" if self._avg(self.yolo) else "  YOLO Inference   : n/a")
        print(f"  Tracking         : {self._avg(self.tracking):.2f} ms")
        print(f"  Render/Display   : {self._avg(self.render):.1f} ms")
        print(f"  Total per frame  : {self._avg(self.total):.1f} ms")
        print("-" * 62)
        print(f"  OCR Jobs Done    : {ocr_worker.total_jobs}")
        print(f"  OCR Avg Time     : {ocr_worker.avg_ocr_ms:.0f} ms  (background thread)")
        print("="*62 + "\n")


# ── HUD drawing ───────────────────────────────────────────────────────────────
def _draw_track(frame, track):
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 2, track.bbox[0]))
    y1 = max(0, min(h - 2, track.bbox[1]))
    x2 = max(x1 + 2, min(w, track.bbox[2]))
    y2 = max(y1 + 2, min(h, track.bbox[3]))

    state_col = COL_GREEN if track.state == "TRACKING" else COL_AMBER

    cv2.rectangle(frame, (x1, y1), (x2, y2), state_col, 3)

    # Corner accents
    cl = min(16, (x2-x1)//4, (y2-y1)//4)
    for (px, py, dx, dy) in [(x1,y1,1,0),(x1,y1,0,1),(x2,y2,-1,0),(x2,y2,0,-1)]:
        cv2.line(frame, (px, py), (px+dx*cl, py+dy*cl), COL_CYAN, 4)

    # Info badge
    ocr_line  = track.display_text
    conf_line = f"conf {track.confidence:.2f}"
    id_line   = f"ID:{track.track_id}"
    lines     = [id_line, ocr_line, conf_line]

    line_h = 18
    pad    = 6
    badge_h = line_h * len(lines) + pad * 2
    badge_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                  for l in lines) + pad * 2

    bx1 = x1
    by2 = y1
    by1 = max(0, by2 - badge_h)
    bx2 = min(w, bx1 + badge_w)

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), COL_DARK, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), state_col, 1)

    for i, line in enumerate(lines):
        col = COL_CYAN if i == 0 else (COL_GREEN if i == 1 else (200, 200, 200))
        cv2.putText(frame, line,
                    (bx1 + pad, by1 + pad + (i+1)*line_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def _draw_hud(frame, profiler: Profiler, ocr_worker: AsyncOCRWorker, n_tracks: int):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 52), COL_DARK, -1)
    cv2.line(frame, (0, 52), (w, 52), COL_CYAN, 2)

    title = f"ANPR Real-Time  |  Tracks: {n_tracks}  |  OCR done: {ocr_worker.total_jobs}"
    cv2.putText(frame, title, (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE, 2, cv2.LINE_AA)

    fps = profiler.fps
    fps_col = COL_GREEN if fps >= 24 else (COL_AMBER if fps >= 16 else COL_RED)
    cv2.putText(frame, f"FPS:{fps:.1f}", (w - 130, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, fps_col, 2, cv2.LINE_AA)

    # Mini diagnostics panel
    panel_lines = [
        f"YOLO : {sum(profiler.yolo)/len(profiler.yolo):.0f} ms" if profiler.yolo else "YOLO : --",
        f"Track: {sum(profiler.tracking)/len(profiler.tracking):.1f} ms" if profiler.tracking else "Track: --",
        f"OCR  : {ocr_worker.avg_ocr_ms:.0f} ms (bg)" if ocr_worker.total_jobs else "OCR  : waiting",
    ]
    px, py = 16, 66
    cv2.rectangle(frame, (px-4, py-4), (px+220, py + 16*len(panel_lines)+4), COL_DARK, -1)
    cv2.rectangle(frame, (px-4, py-4), (px+220, py + 16*len(panel_lines)+4), COL_CYAN, 1)
    for i, pl in enumerate(panel_lines):
        cv2.putText(frame, pl, (px, py + 13 + i*16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 230, 255), 1, cv2.LINE_AA)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(args):
    # ── Init ─────────────────────────────────────────────────────────────────
    os.makedirs("captures", exist_ok=True)

    if not os.path.exists(args.model):
        print(f"[Error] Model file not found: {args.model}")
        sys.exit(1)

    config.IMAGE_SIZE = args.imgsz
    config.CONFIDENCE_THRESHOLD = args.conf

    detector   = PlateDetector(model_path=args.model, imgsz=args.imgsz, conf=args.conf)
    tracker    = Tracker()
    ocr_worker = AsyncOCRWorker().start()

    try:
        camera = ThreadedCamera(index=args.camera,
                                width=config.FRAME_WIDTH,
                                height=config.FRAME_HEIGHT,
                                fps=config.TARGET_FPS).start()
    except RuntimeError as e:
        print(f"[Error] {e}")
        print("  Tip: run with --list-cameras to see available cameras.")
        sys.exit(1)

    profiler   = Profiler()
    frame_no   = 0
    last_crop: dict[int, np.ndarray] = {}   # track_id → last crop for PIP

    window = "ANPR Live Pipeline  (Q=quit  S=snap  P=profile)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    print("\n" + "="*60)
    print("  [*] ANPR PIPELINE RUNNING")
    print(f"     Model   : {args.model}")
    print(f"     Camera  : {args.camera}")
    print(f"     ImgSz   : {args.imgsz}")
    print(f"     Conf    : {args.conf}")
    print("  Controls: Q/ESC=quit  S=snapshot  P=profiling report")
    print("="*60 + "\n")

    try:
        while True:
            t_frame = time.perf_counter()

            # ── 1. Capture ────────────────────────────────────────────────
            t0 = time.perf_counter()
            ok, frame = camera.read()
            profiler.capture.append((time.perf_counter()-t0)*1000)

            if not ok or frame is None:
                time.sleep(0.01)
                continue

            frame_no += 1
            h, w = frame.shape[:2]
            display = frame.copy()

            # ── 2. Detection & Tracking (Configurable Periodic Detection) ─
            is_detection_frame = (frame_no % args.interval == 0) or (len(tracker.tracks) == 0)

            if is_detection_frame:
                t0 = time.perf_counter()
                detections, yolo_ms = detector.detect(frame)
                profiler.yolo.append(yolo_ms)

                # Extract crops for each detection
                crops = []
                for det in detections:
                    # 5% padding
                    pw = max(2, int((det.x2 - det.x1) * 0.05))
                    ph = max(2, int((det.y2 - det.y1) * 0.05))
                    cx1 = max(0, det.x1 - pw)
                    cy1 = max(0, det.y1 - ph)
                    cx2 = min(w, det.x2 + pw)
                    cy2 = min(h, det.y2 + ph)
                    crop = frame[cy1:cy2, cx1:cx2]
                    crops.append(crop if crop.size > 0 else np.zeros((10,10,3), np.uint8))

                # Update tracks with new detections + motion estimation
                t0 = time.perf_counter()
                tracks, track_ms = tracker.update(detections, crops, frame_no)
                profiler.tracking.append(track_ms)
            else:
                # Inter-detection frames: Advance tracks smoothly using velocity prediction
                t0 = time.perf_counter()
                tracks = tracker.step_prediction()
                track_ms = (time.perf_counter() - t0) * 1000.0
                profiler.tracking.append(track_ms)

            # ── 4. OCR scheduling ─────────────────────────────────────────
            for track in tracks:
                if track.should_trigger_ocr():
                    best = track.get_best_candidate()
                    if best is not None:
                        job = OCRJob(
                            track_id  = track.track_id,
                            best_crop = best.crop_bgr,
                            yolo_conf = best.yolo_conf,
                            track_ref = track,
                        )
                        accepted = ocr_worker.submit(job)
                        if accepted:
                            track.ocr_queued = True   # prevent re-submission

            # ── 5. Render ─────────────────────────────────────────────────
            t0 = time.perf_counter()

            for track in tracks:
                if track.should_draw:
                    _draw_track(display, track)
                    # Update last-seen crop for PIP
                    best = track.get_best_candidate()
                    if best is not None:
                        last_crop[track.track_id] = best.crop_bgr

            # Purge PIP cache for removed tracks
            alive_ids = {t.track_id for t in tracks}
            for tid in list(last_crop.keys()):
                if tid not in alive_ids:
                    del last_crop[tid]

            # PIP: show largest available crop in bottom-right
            if last_crop:
                best_tid  = max(last_crop, key=lambda tid:
                                last_crop[tid].shape[0] * last_crop[tid].shape[1])
                pip_src = last_crop[best_tid]
                pip_w, pip_h = 240, 85
                pip = cv2.resize(pip_src, (pip_w, pip_h))
                px, py = w - pip_w - 20, h - pip_h - 20
                cv2.rectangle(display, (px-4, py-22), (px+pip_w+4, py+pip_h+4), COL_CYAN, 2)
                cv2.rectangle(display, (px-4, py-22), (px+pip_w+4, py), COL_DARK, -1)
                cv2.putText(display, f"Best crop  ID:{best_tid}",
                            (px+6, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_CYAN, 1)
                display[py:py+pip_h, px:px+pip_w] = pip

            _draw_hud(display, profiler, ocr_worker, len(tracks))

            profiler.render.append((time.perf_counter()-t0)*1000)
            profiler.total.append((time.perf_counter()-t_frame)*1000)

            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('s'), ord('S')):
                snap = f"captures/anpr_{int(time.time())}.jpg"
                cv2.imwrite(snap, display)
                print(f"[Snapshot] Saved → {snap}")
            elif key in (ord('p'), ord('P')):
                profiler.print_summary(ocr_worker)

    finally:
        ocr_worker.stop()
        camera.stop()
        cv2.destroyAllWindows()
        profiler.print_summary(ocr_worker)
        print("[Done] All resources released.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Real-Time ANPR Pipeline  (YOLO + Tracking + PaddleOCR)")

    parser.add_argument("--model",         type=str,   default="anpr.pt",
                        help="Path to YOLO model weights (default: anpr.pt)")
    parser.add_argument("--camera",        type=int,   default=config.CAMERA_INDEX,
                        help="Camera index to use (default: 0)")
    parser.add_argument("--list-cameras",  action="store_true",
                        help="Scan and list all available camera indices, then exit")
    parser.add_argument("--imgsz",         type=int,   default=config.IMAGE_SIZE,
                        help=f"YOLO inference size (default: {config.IMAGE_SIZE})")
    parser.add_argument("--conf",          type=float, default=config.CONFIDENCE_THRESHOLD,
                        help=f"Detection threshold (default: {config.CONFIDENCE_THRESHOLD})")
    parser.add_argument("--interval",      type=int,   default=config.DETECTION_INTERVAL,
                        help=f"Detection interval (1 = detect every frame, >1 = track in between, default: {config.DETECTION_INTERVAL})")

    args = parser.parse_args()

    if args.list_cameras:
        print("[*] Scanning for available cameras ...")
        cams = list_available_cameras()
        if cams:
            print(f"[+] Available cameras: {cams}")
            print(f"    Use: python main.py --camera <index>")
        else:
            print("[!] No cameras found.")
        sys.exit(0)

    run(args)


if __name__ == "__main__":
    main()
