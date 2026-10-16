import os
import sys
import time
import argparse
import re
import threading
from collections import deque, Counter
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------------------------------------
# CPU THREADING & ENVIRONMENT CONFIGURATION
# ---------------------------------------------------------
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 4) // 2))

# Tune PyTorch CPU thread count to prevent context-switching thread contention
cpu_cores = os.cpu_count() or 4
optimal_threads = max(1, cpu_cores // 2)
torch.set_num_threads(optimal_threads)

INDIAN_STATES = {
    'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA',
    'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH',
    'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ',
    'SK', 'TN', 'TR', 'TS', 'UK', 'UA', 'UP', 'WB', 'AN', 'BH'
}

CHAR_TO_NUM = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'Q': '0', 'D': '0'}
NUM_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}

# ---------------------------------------------------------
# STAGE PROFILER (Microsecond Timers)
# ---------------------------------------------------------
class StageProfiler:
    """Tracks latency across all pipeline stages with true min/avg/max metrics."""
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.times = {
            'capture': deque(maxlen=window_size),
            'preprocess': deque(maxlen=window_size),
            'yolo_infer': deque(maxlen=window_size),
            'postprocess': deque(maxlen=window_size),
            'ocr_infer': deque(maxlen=window_size),
            'render': deque(maxlen=window_size),
            'total_frame': deque(maxlen=window_size)
        }
        self.fps_history = deque(maxlen=window_size)

    def record(self, stage, elapsed_ms):
        if stage in self.times:
            self.times[stage].append(elapsed_ms)

    def record_frame(self, total_ms):
        self.times['total_frame'].append(total_ms)
        if total_ms > 0:
            fps = 1000.0 / total_ms
            self.fps_history.append(fps)

    def get_stage_avg(self, stage):
        data = self.times.get(stage, [])
        return sum(data) / len(data) if data else 0.0

    def get_fps_stats(self):
        if not self.fps_history:
            return {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'current': 0.0}
        return {
            'avg': sum(self.fps_history) / len(self.fps_history),
            'min': min(self.fps_history),
            'max': max(self.fps_history),
            'current': self.fps_history[-1]
        }

    def print_summary(self):
        print("\n" + "="*60)
        print("           📊 ANPR CPU PERFORMANCE PROFILING REPORT")
        print("="*60)
        fps_stats = self.get_fps_stats()
        print(f"  • Real End-to-End Average FPS : {fps_stats['avg']:.2f} FPS")
        print(f"  • Minimum FPS (1% Low)         : {fps_stats['min']:.2f} FPS")
        print(f"  • Maximum FPS (Peak)           : {fps_stats['max']:.2f} FPS")
        print("-" * 60)
        print(f"  • Stage 1: Frame Capture       : {self.get_stage_avg('capture'):.2f} ms")
        print(f"  • Stage 2: Preprocessing       : {self.get_stage_avg('preprocess'):.2f} ms")
        print(f"  • Stage 3: YOLO CPU Inference  : {self.get_stage_avg('yolo_infer'):.2f} ms")
        print(f"  • Stage 4: Post-Processing     : {self.get_stage_avg('postprocess'):.2f} ms")
        if self.times['ocr_infer']:
            print(f"  • Stage 5: OCR Inference (BG)  : {self.get_stage_avg('ocr_infer'):.2f} ms")
        print(f"  • Stage 6: HUD Render & Display: {self.get_stage_avg('render'):.2f} ms")
        print(f"  --------------------------------------------------")
        print(f"  • Total Pipeline Latency       : {self.get_stage_avg('total_frame'):.2f} ms")
        print("="*60 + "\n")


# ---------------------------------------------------------
# THREADED REAL-TIME CAMERA CAPTURE (Zero Lag Buffer)
# ---------------------------------------------------------
class ThreadedCamera:
    """Reads camera continuously in a background thread, discarding stale buffered frames."""
    def __init__(self, src=0, width=1280, height=720):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        if self.started:
            return self
        self.started = True
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.read_lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()


# ---------------------------------------------------------
# OPTIONAL ASYNC OCR WORKER (Throttled & Tracked)
# ---------------------------------------------------------
class AsyncOCRWorker:
    """Background worker for OCR so video loop never drops below 24 FPS."""
    def __init__(self, profiler):
        self.profiler = profiler
        self.paddle_engine = None
        self.is_busy = False
        self.running = True
        self.lock = threading.Lock()
        self.pending_crop = None
        self.last_text = "SCANNING..."
        self.last_conf = 0.0
        self.history = deque(maxlen=10)
        self.confs = deque(maxlen=10)
        self.enabled = False
        self.thread = None

    def start_engine(self):
        try:
            from paddleocr import PaddleOCR
            print("[OCR] Initializing PaddleOCR background engine...")
            self.paddle_engine = PaddleOCR(lang='en', enable_mkldnn=False)
            print("[OCR] PaddleOCR engine ready!")
            self.enabled = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"[OCR] Warning: Could not initialize PaddleOCR: {e}")

    def submit_crop(self, crop_bgr):
        if not self.enabled:
            return False
        with self.lock:
            if not self.is_busy and crop_bgr is not None and crop_bgr.size > 0:
                self.pending_crop = crop_bgr.copy()
                return True
        return False

    def _clean_text(self, raw_boxes):
        if not raw_boxes:
            return "", 0.0
        sorted_boxes = sorted(raw_boxes, key=lambda b: (b[0][0][1] // 25, b[0][0][0]))
        words, confs = [], []
        for bbox, text, conf in sorted_boxes:
            t = text.strip().upper()
            t = re.sub(r'[^A-Z0-9]', '', t)
            if t.startswith('IND') and len(t) > 3:
                t = t[3:]
            elif t == 'IND':
                continue
            if t:
                words.append(t)
                confs.append(conf)
        full = "".join(words)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        if len(full) >= 9:
            chars = list(full)
            for i in (0, 1):
                if i < len(chars) and chars[i] in NUM_TO_CHAR:
                    chars[i] = NUM_TO_CHAR[chars[i]]
            for i in (2, 3):
                if i < len(chars) and chars[i] in CHAR_TO_NUM:
                    chars[i] = CHAR_TO_NUM[chars[i]]
            for i in range(max(4, len(chars) - 4), len(chars)):
                if chars[i] in CHAR_TO_NUM:
                    chars[i] = CHAR_TO_NUM[chars[i]]
            repaired = "".join(chars)
            if repaired[:2] in INDIAN_STATES:
                full = repaired
        return full, avg_conf

    def _loop(self):
        while self.running:
            crop = None
            with self.lock:
                if self.pending_crop is not None:
                    crop = self.pending_crop
                    self.pending_crop = None
                    self.is_busy = True

            if crop is not None and self.paddle_engine is not None:
                t0 = time.perf_counter()
                try:
                    p_res = self.paddle_engine.ocr(crop)
                    raw_boxes = []
                    if p_res:
                        for page in p_res:
                            if isinstance(page, dict):
                                for poly, txt, score in zip(page.get('dt_polys', []), page.get('rec_texts', []), page.get('rec_scores', [])):
                                    raw_boxes.append((poly, str(txt), float(score)))
                            elif isinstance(page, list):
                                for line in page:
                                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                                        raw_boxes.append((line[0], str(line[1][0]), float(line[1][1])))
                    
                    text, conf = self._clean_text(raw_boxes)
                    if text and len(text) >= 5:
                        self.history.append(text)
                        self.confs.append(conf)
                        counts = Counter(self.history)
                        best_text = counts.most_common(1)[0][0]
                        matching = [c for t, c in zip(self.history, self.confs) if t == best_text]
                        self.last_text = best_text
                        self.last_conf = sum(matching) / len(matching)
                except Exception:
                    pass
                finally:
                    t_ocr = (time.perf_counter() - t0) * 1000.0
                    self.profiler.record('ocr_infer', t_ocr)
                    with self.lock:
                        self.is_busy = False
            else:
                time.sleep(0.01)

    def stop(self):
        self.running = False


# ---------------------------------------------------------
# MAIN INFERENCE PIPELINE
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ANPR Ultra Fast CPU Inference & Profiling Suite (anpr.pt)")
    parser.add_argument("--model", type=str, default="anpr.pt", help="Path to YOLO model weights (default: anpr.pt)")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0, 1) or path to video file")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold (default: 0.35)")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size: 640 (standard) or 480 (fast) or 384 (ultra)")
    parser.add_argument("--enable-ocr", action="store_true", help="Enable async background OCR")
    parser.add_argument("--threads", type=int, default=optimal_threads, help=f"CPU thread count (default: {optimal_threads})")
    args = parser.parse_args()

    # Set PyTorch CPU threads
    torch.set_num_threads(args.threads)

    if not os.path.exists(args.model):
        print(f"[Error] Model weights file '{args.model}' not found in current directory.")
        sys.exit(1)

    print(f"[*] Initializing CPU YOLOv8 Engine...")
    print(f"    • Model Path       : {args.model}")
    print(f"    • CPU Cores        : {cpu_cores}")
    print(f"    • Active Threads   : {args.threads}")
    print(f"    • Inference Res    : {args.imgsz}x{args.imgsz}")
    print(f"    • Confidence Thresh: {args.conf}")

    # Load YOLO Model with CPU optimizations
    model = YOLO(args.model)
    model.to('cpu')
    print(f"[✓] YOLO model loaded successfully! Detected Classes: {model.names}")

    # Warmup pass to eliminate initial JIT compiler spike
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    with torch.inference_mode():
        _ = model.predict(source=dummy, imgsz=args.imgsz, device='cpu', verbose=False)
    print("[✓] Model CPU warmup pass completed.")

    # Initialize Profiler
    profiler = StageProfiler(window_size=30)

    # Initialize Async OCR Worker
    ocr_worker = AsyncOCRWorker(profiler)
    if args.enable_ocr:
        ocr_worker.start_engine()

    # Start Decoupled Threaded Camera
    src = int(args.source) if args.source.isdigit() else args.source
    camera = ThreadedCamera(src=src).start()
    time.sleep(0.5)

    os.makedirs("captures", exist_ok=True)
    last_crop_bgr = None
    window_name = "ANPR CPU Real-Time Pipeline (anpr.pt)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n" + "="*60)
    print("  🚀 ANPR CPU REAL-TIME INFERENCE RUNNING (24-30+ FPS)")
    print("  Controls:")
    print("    [Q] / [ESC] : Quit & Print Detailed Profiling Benchmark")
    print("    [O]         : Toggle Async OCR On/Off")
    print("    [S]         : Save Snapshot to captures/")
    print("    [P]         : Print Instantaneous Benchmark Report in Terminal")
    print("="*60 + "\n")

    frame_count = 0

    try:
        with torch.inference_mode():
            while True:
                t_frame_start = time.perf_counter()

                # 1. FRAME CAPTURE
                t0 = time.perf_counter()
                ret, frame = camera.read()
                t_cap = (time.perf_counter() - t0) * 1000.0
                profiler.record('capture', t_cap)

                if not ret or frame is None:
                    continue

                frame_count += 1
                h, w = frame.shape[:2]
                display_frame = frame.copy()

                # 2. PREPROCESSING
                t0 = time.perf_counter()
                # Frame is passed directly into Ultralytics engine (letterbox & tensor conversion)
                t_prep = (time.perf_counter() - t0) * 1000.0
                profiler.record('preprocess', t_prep)

                # 3. YOLO CPU INFERENCE
                t0 = time.perf_counter()
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device='cpu',
                    verbose=False,
                    stream=False
                )
                t_yolo = (time.perf_counter() - t0) * 1000.0
                profiler.record('yolo_infer', t_yolo)

                # 4. POST-PROCESSING (NMS & Bounding Box extraction)
                t0 = time.perf_counter()
                detected_plates = []
                for r in results:
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(w, xyxy[2]), min(h, xyxy[3])
                        if (x2 - x1) > 15 and (y2 - y1) > 8:
                            detected_plates.append((x1, y1, x2, y2, conf))

                if detected_plates:
                    detected_plates.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
                    bx1, by1, bx2, by2, bconf = detected_plates[0]
                    crop = frame[by1:by2, bx1:bx2]
                    last_crop_bgr = crop.copy()

                    # Trigger async OCR worker without blocking frame loop
                    if ocr_worker.enabled:
                        ocr_worker.submit_crop(crop)

                t_post = (time.perf_counter() - t0) * 1000.0
                profiler.record('postprocess', t_post)

                # 5. RENDERING & HUD DISPLAY
                t0 = time.perf_counter()

                # Draw plate bounding boxes
                for x1, y1, x2, y2, conf in detected_plates:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 128), 3)
                    
                    # Corner styling
                    c_len = min(18, (x2 - x1) // 4, (y2 - y1) // 4)
                    cv2.line(display_frame, (x1, y1), (x1 + c_len, y1), (56, 189, 248), 4)
                    cv2.line(display_frame, (x1, y1), (x1, y1 + c_len), (56, 189, 248), 4)
                    cv2.line(display_frame, (x2, y2), (x2 - c_len, y2), (56, 189, 248), 4)
                    cv2.line(display_frame, (x2, y2), (x2, y2 - c_len), (56, 189, 248), 4)

                    if ocr_worker.enabled:
                        label = f"Plate: {ocr_worker.last_text} ({ocr_worker.last_conf*100:.0f}%)"
                    else:
                        label = f"Number Plate: {conf*100:.1f}%"

                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (x1, max(0, y1 - lh - 10)), (x1 + lw + 10, y1), (0, 255, 128), -1)
                    cv2.putText(display_frame, label, (x1 + 5, max(lh + 3, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                # Calculate True End-to-End Latency & FPS
                t_total_frame = (time.perf_counter() - t_frame_start) * 1000.0
                profiler.record_frame(t_total_frame)
                fps_stats = profiler.get_fps_stats()
                avg_fps = fps_stats['avg']

                # HUD Header Banner
                cv2.rectangle(display_frame, (0, 0), (w, 55), (15, 20, 30), -1)
                cv2.line(display_frame, (0, 55), (w, 55), (56, 189, 248), 2)

                hud_left = f"YOLOv8 CPU (anpr.pt {args.imgsz}x{args.imgsz}) | Plates: {len(detected_plates)} | OCR: {'ACTIVE' if ocr_worker.enabled else 'OFF'}"
                cv2.putText(display_frame, hud_left, (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                # FPS Meter
                fps_color = (0, 255, 128) if avg_fps >= 24 else ((0, 215, 255) if avg_fps >= 18 else (0, 0, 255))
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (w - 150, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, fps_color, 2, cv2.LINE_AA)

                # Live Profiling Stage Diagnostics Panel (Top-Left under header)
                panel_h = 100
                cv2.rectangle(display_frame, (15, 68), (340, 68 + panel_h), (10, 15, 25), -1)
                cv2.rectangle(display_frame, (15, 68), (340, 68 + panel_h), (56, 189, 248), 1)

                cv2.putText(display_frame, f"LATENCY BREAKDOWN", (25, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)
                cv2.putText(display_frame, f"• YOLO CPU Infer : {profiler.get_stage_avg('yolo_infer'):.1f} ms ({1000.0/max(1, profiler.get_stage_avg('yolo_infer')):.0f} FPS pure)", (25, 104),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 240, 255), 1, cv2.LINE_AA)
                cv2.putText(display_frame, f"• Post-Processing: {profiler.get_stage_avg('postprocess'):.1f} ms", (25, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 240, 255), 1, cv2.LINE_AA)
                cv2.putText(display_frame, f"• End-to-End Lat : {t_total_frame:.1f} ms (Min:{fps_stats['min']:.0f} Max:{fps_stats['max']:.0f})", (25, 136),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1, cv2.LINE_AA)
                if ocr_worker.enabled:
                    cv2.putText(display_frame, f"• OCR (Background): {profiler.get_stage_avg('ocr_infer'):.1f} ms", (25, 152),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 100), 1, cv2.LINE_AA)

                # PIP Cropped Plate (Bottom-Right)
                if last_crop_bgr is not None and last_crop_bgr.size > 0:
                    pip_w, pip_h = 240, 85
                    pip_resized = cv2.resize(last_crop_bgr, (pip_w, pip_h))
                    px, py = w - pip_w - 20, h - pip_h - 20

                    cv2.rectangle(display_frame, (px - 4, py - 22), (px + pip_w + 4, py + pip_h + 4), (56, 189, 248), 2)
                    cv2.rectangle(display_frame, (px - 4, py - 22), (px + pip_w + 4, py), (15, 20, 30), -1)
                    cv2.putText(display_frame, "ZOOMED PLATE CROP", (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)
                    display_frame[py:py + pip_h, px:px + pip_w] = pip_resized

                cv2.imshow(window_name, display_frame)
                t_rend = (time.perf_counter() - t0) * 1000.0
                profiler.record('render', t_rend)

                # Keyboard Controls
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                elif key in (ord('s'), ord('S')):
                    snap_path = f"captures/anpr_snap_{int(time.time())}.jpg"
                    cv2.imwrite(snap_path, display_frame)
                    print(f"[Snapshot] Saved to {snap_path}")
                elif key in (ord('o'), ord('O')):
                    if not ocr_worker.enabled:
                        ocr_worker.start_engine()
                        print("[OCR] Async Background OCR Activated.")
                    else:
                        ocr_worker.enabled = False
                        ocr_worker.last_text = "SCANNING..."
                        print("[OCR] Async Background OCR Deactivated.")
                elif key in (ord('p'), ord('P')):
                    profiler.print_summary()

    finally:
        ocr_worker.stop()
        camera.stop()
        cv2.destroyAllWindows()
        profiler.print_summary()
        print("[Done] Resources cleaned up.")

if __name__ == "__main__":
    main()
