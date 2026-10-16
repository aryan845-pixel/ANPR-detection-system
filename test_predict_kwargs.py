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
print(" PADDLEX PREDICT KWARGS LATENCY TEST")
print("=" * 70)

from paddleocr import PaddleOCR

params = {
    "lang": "en",
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "enable_mkldnn": False
}

t0 = time.perf_counter()
ocr_engine = PaddleOCR(**params)
print(f" Init time: {(time.perf_counter() - t0)*1000:.2f} ms")

predict_kwargs = {
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
}

# Warmup
ocr_engine.predict(crop_resized, **predict_kwargs)

latencies = []
for i in range(10):
    t0 = time.perf_counter()
    res = ocr_engine.predict(crop_resized, **predict_kwargs)
    lat = (time.perf_counter() - t0) * 1000.0
    latencies.append(lat)

avg_lat = sum(latencies) / len(latencies)
print(f"\n10 Consecutive Pass Latencies (ms):\n{[round(l, 2) for l in latencies]}")
print(f"\nAVERAGE INFERENCE LATENCY: {avg_lat:.2f} ms ({avg_lat/1000.0:.3f} sec)")
print("=" * 70)
