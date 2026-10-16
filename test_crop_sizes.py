import os
import sys
import time
import cv2
import numpy as np

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

# Create synthetic license plate test image
dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
cv2.rectangle(dummy_img, (200, 200), (440, 280), (255, 255, 255), -1)
cv2.rectangle(dummy_img, (200, 200), (440, 280), (0, 0, 0), 3)
cv2.putText(dummy_img, "MH12PQ4589", (210, 255), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

crop = dummy_img[200:280, 200:440]

print("=" * 70)
print(" PADDLEOCR INFERENCE LATENCY vs CROP HEIGHT BENCHMARK")
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

ocr_engine = PaddleOCR(**params)

predict_kwargs = {
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "text_det_limit_side_len": 320,
    "text_det_limit_type": "max",
}

for target_h in [60, 80, 100, 120, 160]:
    h, w = crop.shape[:2]
    scale = target_h / float(h)
    new_w = int(w * scale)
    proc = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Warmup
    ocr_engine.predict(proc, **predict_kwargs)

    latencies = []
    text_read = ""
    for _ in range(3):
        t0 = time.perf_counter()
        res = ocr_engine.predict(proc, **predict_kwargs)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if res and isinstance(res, list) and len(res) > 0 and 'rec_texts' in res[0]:
            text_read = res[0]['rec_texts']

    avg = sum(latencies) / len(latencies)
    print(f" Target H: {target_h}px (Crop Size: {new_w}x{target_h}) | Avg Latency: {avg:.2f} ms ({avg/1000.0:.3f} s) | Text: {text_read}")

print("=" * 70)
