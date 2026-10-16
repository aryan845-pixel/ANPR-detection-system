import os
import sys
import time
import cv2
import numpy as np

# Create synthetic license plate test image
dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
cv2.rectangle(dummy_img, (200, 200), (440, 280), (255, 255, 255), -1)
cv2.rectangle(dummy_img, (200, 200), (440, 280), (0, 0, 0), 3)
cv2.putText(dummy_img, "MH12PQ4589", (210, 255), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

crop = dummy_img[200:280, 200:440]
crop_resized = cv2.resize(crop, (480, 160), interpolation=cv2.INTER_CUBIC)

print("=" * 70)
print(" PADDLEOCR PERFORMANCE OPTIMIZATION BENCHMARK")
print("=" * 70)

def test_paddle(name, env_flags, paddle_kwargs):
    print(f"\n---> Testing Config: {name}")
    for k, v in env_flags.items():
        os.environ[k] = str(v)
    
    try:
        from paddleocr import PaddleOCR
        t0 = time.perf_counter()
        ocr_engine = PaddleOCR(**paddle_kwargs)
        init_ms = (time.perf_counter() - t0) * 1000.0
        print(f" Init time: {init_ms:.2f} ms")

        # Warmup pass
        ocr_engine.ocr(crop_resized)

        # Timed passes
        latencies = []
        res = None
        for i in range(3):
            t0 = time.perf_counter()
            res = ocr_engine.ocr(crop_resized)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_latency = sum(latencies) / len(latencies)
        print(f" Warm Latencies: {[round(l, 2) for l in latencies]} ms")
        print(f" Average Latency: {avg_latency:.2f} ms ({avg_latency/1000.0:.2f} s)")
        print(f" Sample Result: {res}")
        return avg_latency
    except Exception as e:
        print(f" Error testing {name}: {e}")
        return None

# Test 1: Baseline (current flags)
test_paddle(
    "1. Baseline (use_mkldnn=0)",
    {"FLAGS_enable_pir_api": "0", "FLAGS_use_mkldnn": "0"},
    {"lang": "en", "enable_mkldnn": False}
)

# Test 2: Enable OneDNN / MKL-DNN
test_paddle(
    "2. Enable MKL-DNN (enable_mkldnn=True)",
    {"FLAGS_enable_pir_api": "0", "FLAGS_use_mkldnn": "1"},
    {"lang": "en", "enable_mkldnn": True, "cpu_threads": 4}
)

# Test 3: Disable doc orientation & unwarping if supported in PaddleOCR 3.x
test_paddle(
    "3. Optimized Flags (no angle cls, mkldnn, 4 threads)",
    {"FLAGS_enable_pir_api": "0", "FLAGS_use_mkldnn": "1"},
    {"lang": "en", "enable_mkldnn": True, "cpu_threads": 4, "use_angle_cls": False}
)

print("=" * 70)
