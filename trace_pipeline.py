"""
Precise in-process benchmark that mimics the EXACT Flask pipeline.
No HTTP overhead — just process_image() directly.
"""
import os, time, sys
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
import numpy as np

# Create the EXACT same test image as benchmark_app.py
dummy_img = np.ones((720, 1280, 3), dtype=np.uint8) * 128
cv2.rectangle(dummy_img, (400, 300), (800, 420), (255, 255, 255), -1)
cv2.rectangle(dummy_img, (400, 300), (800, 420), (0, 0, 0), 4)
cv2.putText(dummy_img, "MH12PQ4589", (430, 385), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
_, encoded = cv2.imencode('.jpg', dummy_img)
img_bytes = encoded.tobytes()

# Import app module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import process_image

print("=" * 70)
print(" IN-PROCESS process_image() BENCHMARK (no HTTP)")
print("=" * 70)

for i in range(1, 6):
    t0 = time.perf_counter()
    res = process_image(img_bytes)
    lat = (time.perf_counter() - t0) * 1000
    print(f" Run #{i}: {lat:.0f} ms | Plate='{res.get('text')}' | latency_ms={res.get('latency_ms')}")

print("=" * 70)

# Now trace each sub-step manually
print("\n DETAILED STEP-BY-STEP TRACE:")
print("-" * 70)

nparr = np.frombuffer(img_bytes, np.uint8)

t0 = time.perf_counter()
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
print(f"  1. Image decode:    {(time.perf_counter()-t0)*1000:.1f} ms")

from app import yolo_model, paddle_ocr, extract_plate_text_with_paddleocr
import base64

t0 = time.perf_counter()
results = yolo_model.predict(source=img, conf=0.25, imgsz=320, verbose=False, device="cpu")
print(f"  2. YOLO predict:    {(time.perf_counter()-t0)*1000:.1f} ms | Boxes: {len(results[0].boxes)}")

boxes = results[0].boxes
h_orig, w_orig = img.shape[:2]

if len(boxes) > 0:
    box = boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = int(bw * 0.05), int(bh * 0.05)
    cx1, cy1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    cx2, cy2 = min(w_orig, x2 + pad_w), min(h_orig, y2 + pad_h)
    plate_crop = img[cy1:cy2, cx1:cx2]
    
    t0 = time.perf_counter()
    extracted_text, ocr_conf, best_crop, quality_data = extract_plate_text_with_paddleocr(plate_crop)
    print(f"  3. OCR pipeline:    {(time.perf_counter()-t0)*1000:.1f} ms | Text: '{extracted_text}'")
    
    t0 = time.perf_counter()
    _, c_buf = cv2.imencode('.jpg', best_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    crop_b64 = "data:image/jpeg;base64," + base64.b64encode(c_buf).decode('utf-8')
    print(f"  4. Crop encode:     {(time.perf_counter()-t0)*1000:.1f} ms")
    
    t0 = time.perf_counter()
    annotated = img.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (220, 85, 30), 2)
    print(f"  5. Annotate draw:   {(time.perf_counter()-t0)*1000:.1f} ms")
    
    t0 = time.perf_counter()
    _, a_buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ann_b64 = "data:image/jpeg;base64," + base64.b64encode(a_buf).decode('utf-8')
    print(f"  6. Annotated enc:   {(time.perf_counter()-t0)*1000:.1f} ms | B64 len: {len(ann_b64)}")

print("=" * 70)
