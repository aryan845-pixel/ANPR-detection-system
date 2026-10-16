import os
import sys
import time
import cv2
import numpy as np

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

# Create synthetic license plate test image
dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
cv2.rectangle(dummy_img, (200, 200), (440, 280), (255, 255, 255), -1)
cv2.rectangle(dummy_img, (200, 200), (440, 280), (0, 0, 0), 3)
cv2.putText(dummy_img, "MH12PQ4589", (210, 255), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

crop = dummy_img[200:280, 200:440]
crop_resized = cv2.resize(crop, (480, 160), interpolation=cv2.INTER_CUBIC)

print("=" * 70)
print(" PADDLEX DOCUMENT DISABLE TEST (OPTIMIZED SPEED)")
print("=" * 70)

from paddleocr import PaddleOCR

params = {
    "lang": "en",
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "enable_mkldnn": False
}

print(f"Initializing PaddleOCR with params: {params}")
t0 = time.perf_counter()
ocr_engine = PaddleOCR(**params)
init_ms = (time.perf_counter() - t0) * 1000.0
print(f" Init time: {init_ms:.2f} ms")

print("\nRunning Warmup Pass...")
t0 = time.perf_counter()
res = ocr_engine.ocr(crop_resized)
warmup_ms = (time.perf_counter() - t0) * 1000.0
print(f" Warmup Pass Time: {warmup_ms:.2f} ms")

latencies = []
for i in range(5):
    t0 = time.perf_counter()
    res = ocr_engine.ocr(crop_resized)
    latencies.append((time.perf_counter() - t0) * 1000.0)

avg_lat = sum(latencies) / len(latencies)
print(f"\nWarm Latencies: {[round(l, 2) for l in latencies]} ms")
print(f"AVERAGE INFERENCE LATENCY: {avg_lat:.2f} ms ({avg_lat/1000.0:.3f} sec)")
print(f"Sample Result Output: {res}")
print("=" * 70)
