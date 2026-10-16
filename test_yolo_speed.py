import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Create synthetic test image
dummy_img = np.ones((720, 1280, 3), dtype=np.uint8) * 128
cv2.rectangle(dummy_img, (400, 300), (600, 360), (255, 255, 255), -1)

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anpr.pt")
model = YOLO(model_path)

# Warmup
model.predict(source=dummy_img, conf=0.25, imgsz=640, verbose=False, device="cpu")

print("=" * 70)
print(" YOLOv8 SPEED BENCHMARK BY IMGSZ")
print("=" * 70)

for sz in [640, 480, 320]:
    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        res = model.predict(source=dummy_img, conf=0.25, imgsz=sz, verbose=False, device="cpu")
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg = sum(latencies) / len(latencies)
    print(f" ImgSz: {sz}px  |  Average Latency: {avg:.2f} ms ({avg/1000.0:.3f} s)")

print("=" * 70)
