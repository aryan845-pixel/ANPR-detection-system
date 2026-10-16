"""Test PaddleOCR mobile vs medium models for speed + accuracy."""
import os, time, cv2, numpy as np

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR

# Create plate crop
plate = np.ones((40, 150, 3), dtype=np.uint8) * 200
cv2.putText(plate, "MH12PQ4589", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
crop_64 = cv2.resize(plate, (int(150 * (64/40)), 64), interpolation=cv2.INTER_CUBIC)

print("=" * 70)

# Test A: Current medium models
print("\n--- A: PP-OCRv6_medium (current) ---")
ocr_med = PaddleOCR(
    lang='en',
    use_doc_orientation_classify=False, use_doc_unwarping=False,
    use_textline_orientation=False, text_det_limit_side_len=320,
    text_det_limit_type="max", enable_mkldnn=False
)
ocr_med.predict(crop_64, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, text_det_limit_side_len=320, text_det_limit_type="max")
times = []
for _ in range(5):
    t0 = time.perf_counter()
    r = ocr_med.predict(crop_64, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, text_det_limit_side_len=320, text_det_limit_type="max")
    times.append((time.perf_counter() - t0) * 1000)
txt = r[0].get('rec_texts', []) if r and isinstance(r[0], dict) else str(r)[:80]
print(f"  Avg: {sum(times)/len(times):.0f}ms | Text: {txt} | All: {[f'{t:.0f}' for t in times]}")

# Test B: Mobile models
print("\n--- B: PP-OCRv4_mobile ---")
try:
    ocr_mob = PaddleOCR(
        lang='en',
        text_detection_model_name="PP-OCRv4_mobile_det",
        text_recognition_model_name="PP-OCRv4_mobile_rec",
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, text_det_limit_side_len=320,
        text_det_limit_type="max", enable_mkldnn=False
    )
    ocr_mob.predict(crop_64, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, text_det_limit_side_len=320, text_det_limit_type="max")
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = ocr_mob.predict(crop_64, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, text_det_limit_side_len=320, text_det_limit_type="max")
        times.append((time.perf_counter() - t0) * 1000)
    txt = r[0].get('rec_texts', []) if r and isinstance(r[0], dict) else str(r)[:80]
    print(f"  Avg: {sum(times)/len(times):.0f}ms | Text: {txt} | All: {[f'{t:.0f}' for t in times]}")
except Exception as e:
    print(f"  Error: {e}")

# Test C: Server model (lighter)
print("\n--- C: PP-OCRv4_server ---")
try:
    ocr_srv = PaddleOCR(
        lang='en',
        text_detection_model_name="PP-OCRv4_server_det",
        text_recognition_model_name="PP-OCRv4_server_rec",
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, text_det_limit_side_len=320,
        text_det_limit_type="max", enable_mkldnn=False
    )
    ocr_srv.predict(crop_64, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, text_det_limit_side_len=320, text_det_limit_type="max")
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = ocr_srv.predict(crop_64, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, text_det_limit_side_len=320, text_det_limit_type="max")
        times.append((time.perf_counter() - t0) * 1000)
    txt = r[0].get('rec_texts', []) if r and isinstance(r[0], dict) else str(r)[:80]
    print(f"  Avg: {sum(times)/len(times):.0f}ms | Text: {txt} | All: {[f'{t:.0f}' for t in times]}")
except Exception as e:
    print(f"  Error: {e}")

# Test D: det_limit_side_len=160 (even smaller detection)
print("\n--- D: Medium models + det_limit=160 ---")
times = []
for _ in range(5):
    t0 = time.perf_counter()
    r = ocr_med.predict(crop_64, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, text_det_limit_side_len=160, text_det_limit_type="max")
    times.append((time.perf_counter() - t0) * 1000)
txt = r[0].get('rec_texts', []) if r and isinstance(r[0], dict) else str(r)[:80]
print(f"  Avg: {sum(times)/len(times):.0f}ms | Text: {txt} | All: {[f'{t:.0f}' for t in times]}")

print("\n" + "=" * 70)
