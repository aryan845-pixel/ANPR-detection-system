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

print("=" * 70)
print(" PADDLEOCR TUNED BENCHMARK TEST")
print("=" * 70)

from paddleocr import PaddleOCR

params = {
    "lang": "en",
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "text_det_limit_side_len": 320,
    "text_det_limit_type": "max",
    "enable_mkldnn": False
}

t0 = time.perf_counter()
ocr_engine = PaddleOCR(**params)
print(f" Init time: {(time.perf_counter() - t0)*1000:.2f} ms")

predict_kwargs = {
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "text_det_limit_side_len": 320,
    "text_det_limit_type": "max",
}

# Warmup
ocr_engine.predict(crop, **predict_kwargs)

latencies = []
for i in range(5):
    t0 = time.perf_counter()
    res = ocr_engine.predict(crop, **predict_kwargs)
    lat = (time.perf_counter() - t0) * 1000.0
    latencies.append(lat)

avg_lat = sum(latencies) / len(latencies)
print(f"\n5 Consecutive Pass Latencies (ms):\n{[round(l, 2) for l in latencies]}")
print(f"\nAVERAGE INFERENCE LATENCY: {avg_lat:.2f} ms ({avg_lat/1000.0:.3f} sec)")
print(f"Sample Output: {res}")
print("=" * 70)
