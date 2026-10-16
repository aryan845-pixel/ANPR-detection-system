import os
import sys
import time
import cv2
import numpy as np

# Set Paddle environment flags before import
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

print("=" * 70)
print(" ANPR DETAILED BENCHMARK PROFILER")
print("=" * 70)

# Step 0: Imports timing
t_import_start = time.perf_counter()
from ultralytics import YOLO
from paddleocr import PaddleOCR
t_import_end = time.perf_counter()
print(f"[Profiling] Imports time: {(t_import_end - t_import_start)*1000:.2f} ms")

# Step 1: YOLO Initialization
t_yolo_init_start = time.perf_counter()
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anpr.pt")
yolo_model = YOLO(model_path)
t_yolo_init_end = time.perf_counter()
print(f"[Profiling] YOLO model load time: {(t_yolo_init_end - t_yolo_init_start)*1000:.2f} ms")

# Step 2: PaddleOCR Initialization
t_paddle_init_start = time.perf_counter()
paddle_ocr = PaddleOCR(lang='en', enable_mkldnn=False)
t_paddle_init_end = time.perf_counter()
print(f"[Profiling] PaddleOCR model init time: {(t_paddle_init_end - t_paddle_init_start)*1000:.2f} ms")

# Step 3: Create synthetic license plate test image
dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
cv2.rectangle(dummy_img, (200, 200), (440, 280), (255, 255, 255), -1)
cv2.rectangle(dummy_img, (200, 200), (440, 280), (0, 0, 0), 3)
cv2.putText(dummy_img, "MH12PQ4589", (210, 255), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

_, img_encoded = cv2.imencode('.jpg', dummy_img)
img_bytes = img_encoded.tobytes()

def run_single_pass(pass_name=""):
    t_start = time.perf_counter()

    # 1. Image decode
    t0 = time.perf_counter()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    t_decode = (time.perf_counter() - t0) * 1000.0

    # 2. YOLO inference
    t0 = time.perf_counter()
    results = yolo_model.predict(source=img, conf=0.25, imgsz=640, verbose=False, device="cpu")
    boxes = results[0].boxes
    t_yolo_inf = (time.perf_counter() - t0) * 1000.0

    # 3. Crop
    t0 = time.perf_counter()
    h_orig, w_orig = img.shape[:2]
    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0].xyxy[0].tolist())
        plate_crop = img[y1:y2, x1:x2]
    else:
        plate_crop = dummy_img[200:280, 200:440]
    t_crop = (time.perf_counter() - t0) * 1000.0

    # 4. Preprocessing
    t0 = time.perf_counter()
    h, w = plate_crop.shape[:2]
    target_height = 160
    if h < target_height:
        scale = target_height / float(h)
        new_w = int(w * scale)
        processed_crop = cv2.resize(plate_crop, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
    else:
        processed_crop = plate_crop.copy()
    t_prep = (time.perf_counter() - t0) * 1000.0

    # 5. PaddleOCR inference
    t0 = time.perf_counter()
    res_full = paddle_ocr.ocr(processed_crop)
    t_paddle_full = (time.perf_counter() - t0) * 1000.0

    # 6. Encoding / UI prep
    t0 = time.perf_counter()
    _, a_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    t_encode = (time.perf_counter() - t0) * 1000.0

    t_total = (time.perf_counter() - t_start) * 1000.0

    print(f"\n--- RESULTS FOR: {pass_name} ---")
    print(f" Image decode       : {t_decode:.2f} ms")
    print(f" YOLO inference     : {t_yolo_inf:.2f} ms")
    print(f" Plate cropping     : {t_crop:.2f} ms")
    print(f" Preprocessing      : {t_prep:.2f} ms")
    print(f" PaddleOCR inference: {t_paddle_full:.2f} ms")
    print(f" Image encoding     : {t_encode:.2f} ms")
    print(f" TOTAL REQUEST TIME : {t_total:.2f} ms ({t_total/1000.0:.2f} sec)")

print("\nRunning Pass 1 (Cold / First Request)...")
run_single_pass("Pass 1 (Cold Warmup Request)")

print("\nRunning Pass 2 (Warm Request)...")
run_single_pass("Pass 2 (Warm Request)")

print("\nRunning Pass 3 (Warm Request)...")
run_single_pass("Pass 3 (Warm Request)")

print("=" * 70)
