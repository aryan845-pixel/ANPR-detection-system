import os
import sys
import time
import cv2
import numpy as np

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR

# Synthetic license plate crop (already cropped by YOLO)
dummy_crop = np.ones((40, 150, 3), dtype=np.uint8) * 128
cv2.putText(dummy_crop, "MH12PQ4589", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

print("=" * 70)
print(" EXPERT OPTIMIZATION TEST: RECOGNITION ONLY vs FULL DET+REC")
print("=" * 70)

# Option A: Normal PaddleOCR with det+rec (det_limit_side_len=320)
print("\n--- Option A: Full PaddleOCR (Det + Rec) ---")
ocr_a = PaddleOCR(
    lang='en',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_limit_side_len=320,
    text_det_limit_type="max",
    enable_mkldnn=False
)

# Resize to 80px height vs 160px height
h, w = dummy_crop.shape[:2]
crop_80 = cv2.resize(dummy_crop, (int(w * (80/h)), 80), interpolation=cv2.INTER_CUBIC)
crop_160 = cv2.resize(dummy_crop, (int(w * (160/h)), 160), interpolation=cv2.INTER_CUBIC)

t0 = time.perf_counter()
res_a_160 = ocr_a.predict(crop_160)
lat_a_160 = (time.perf_counter() - t0) * 1000.0
print(f" 160px Crop (Det+Rec) Latency: {lat_a_160:.2f} ms | Text: {res_a_160[0].get('rec_texts') if res_a_160 else ''}")

t0 = time.perf_counter()
res_a_80 = ocr_a.predict(crop_80)
lat_a_80 = (time.perf_counter() - t0) * 1000.0
print(f" 80px Crop (Det+Rec) Latency : {lat_a_80:.2f} ms | Text: {res_a_80[0].get('rec_texts') if res_a_80 else ''}")


# Option B: Recognition ONLY (disable text_detection_model)
print("\n--- Option B: Recognition ONLY (text_detection_model_name=None) ---")
try:
    ocr_b = PaddleOCR(
        lang='en',
        text_detection_model_name=None,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False
    )
    t0 = time.perf_counter()
    res_b_80 = ocr_b.predict(crop_80)
    lat_b_80 = (time.perf_counter() - t0) * 1000.0
    print(f" Recognition ONLY Latency: {lat_b_80:.2f} ms | Output: {res_b_80}")
except Exception as e:
    print(f" Option B Error: {e}")

print("=" * 70)
