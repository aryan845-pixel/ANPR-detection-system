"""
Deep performance profiling to find remaining bottlenecks.
Tests:
1. YOLO at imgsz=640 vs 320
2. PaddleOCR det+rec vs rec-only
3. Image decode + resize overhead
4. Base64 encoding overhead
"""
import os
import time
import cv2
import numpy as np
import base64

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

print("=" * 70)
print(" DEEP PERFORMANCE PROFILING")
print("=" * 70)

# Create synthetic test image (1280x720 with plate-like text)
dummy_img = np.ones((720, 1280, 3), dtype=np.uint8) * 128
cv2.rectangle(dummy_img, (400, 300), (800, 420), (255, 255, 255), -1)
cv2.rectangle(dummy_img, (400, 300), (800, 420), (0, 0, 0), 4)
cv2.putText(dummy_img, "MH12PQ4589", (430, 385), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)

# Encode as JPEG (simulates uploaded file)
_, encoded = cv2.imencode('.jpg', dummy_img)
img_bytes = encoded.tobytes()

# ── Test 1: Image decode ──
print("\n--- Test 1: Image Decode ---")
for _ in range(3):
    t0 = time.perf_counter()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(f"  Decode: {(time.perf_counter() - t0)*1000:.2f} ms | Shape: {img.shape}")

# ── Test 2: YOLO at different imgsz ──
print("\n--- Test 2: YOLO Detection Speed ---")
from ultralytics import YOLO
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anpr.pt")
yolo = YOLO(model_path)

# Warmup
yolo.predict(source=dummy_img, conf=0.25, imgsz=640, verbose=False, device="cpu")

for imgsz in [640, 416, 320]:
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        res = yolo.predict(source=img, conf=0.25, imgsz=imgsz, verbose=False, device="cpu")
        times.append((time.perf_counter() - t0) * 1000)
    boxes_found = len(res[0].boxes) if res else 0
    avg = sum(times) / len(times)
    print(f"  imgsz={imgsz}: Avg={avg:.1f} ms | Detections={boxes_found} | All: {[f'{t:.0f}' for t in times]}")

# ── Test 3: PaddleOCR Det+Rec vs Rec-Only ──
print("\n--- Test 3: PaddleOCR Modes ---")
from paddleocr import PaddleOCR

# Full OCR (det + rec) 
ocr_full = PaddleOCR(
    lang='en',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_limit_side_len=320,
    text_det_limit_type="max",
    enable_mkldnn=False
)

# Create plate crop at 96px height
plate_crop = np.ones((40, 150, 3), dtype=np.uint8) * 200
cv2.putText(plate_crop, "MH12PQ4589", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
crop_96 = cv2.resize(plate_crop, (int(150 * (96/40)), 96), interpolation=cv2.INTER_CUBIC)

# Warmup
ocr_full.predict(crop_96, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, text_det_limit_side_len=320, text_det_limit_type="max")

print("  Full Det+Rec on 96px crop:")
for i in range(5):
    t0 = time.perf_counter()
    res = ocr_full.predict(
        crop_96,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=320,
        text_det_limit_type="max"
    )
    lat = (time.perf_counter() - t0) * 1000
    texts = res[0].get('rec_texts', []) if res and isinstance(res[0], dict) else str(res)[:60]
    print(f"    Run {i+1}: {lat:.1f} ms | Text: {texts}")

# Try smaller crop sizes
for h_target in [64, 48]:
    crop_small = cv2.resize(plate_crop, (int(150 * (h_target/40)), h_target), interpolation=cv2.INTER_CUBIC)
    ocr_full.predict(crop_small, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, text_det_limit_side_len=320, text_det_limit_type="max")
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        res = ocr_full.predict(
            crop_small,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=320,
            text_det_limit_type="max"
        )
        times.append((time.perf_counter() - t0) * 1000)
    texts = res[0].get('rec_texts', []) if res and isinstance(res[0], dict) else str(res)[:60]
    avg = sum(times) / len(times)
    print(f"  {h_target}px crop: Avg={avg:.1f} ms | Text: {texts}")

# ── Test 4: Base64 encoding overhead ──
print("\n--- Test 4: Base64 Encoding ---")
_, a_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
times = []
for _ in range(100):
    t0 = time.perf_counter()
    b64 = "data:image/jpeg;base64," + base64.b64encode(a_buf).decode('utf-8')
    times.append((time.perf_counter() - t0) * 1000)
avg = sum(times) / len(times)
print(f"  Base64 encode (annotated img): Avg={avg:.3f} ms (100 runs)")

# ── Test 5: YOLO at imgsz=320 + OCR at 64px crop ──
print("\n--- Test 5: Full Pipeline Simulation (imgsz=320 + 64px crop) ---")
for i in range(5):
    t_total = time.perf_counter()
    
    t0 = time.perf_counter()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    t_decode = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    res = yolo.predict(source=img2, conf=0.25, imgsz=320, verbose=False, device="cpu")
    t_yolo = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    crop_test = cv2.resize(plate_crop, (int(150 * (64/40)), 64), interpolation=cv2.INTER_CUBIC)
    res_ocr = ocr_full.predict(
        crop_test,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=320,
        text_det_limit_type="max"
    )
    t_ocr = (time.perf_counter() - t0) * 1000
    
    t_total_ms = (time.perf_counter() - t_total) * 1000
    print(f"  Run {i+1}: Total={t_total_ms:.0f}ms (Decode={t_decode:.0f} YOLO={t_yolo:.0f} OCR={t_ocr:.0f})")

print("\n" + "=" * 70)
