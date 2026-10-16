import os
import time
import requests
import cv2
import numpy as np

# Create synthetic test image file
dummy_img = np.ones((720, 1280, 3), dtype=np.uint8) * 128
cv2.rectangle(dummy_img, (400, 300), (800, 420), (255, 255, 255), -1)
cv2.rectangle(dummy_img, (400, 300), (800, 420), (0, 0, 0), 4)
cv2.putText(dummy_img, "MH12PQ4589", (430, 385), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)

_, encoded = cv2.imencode('.jpg', dummy_img)
img_bytes = encoded.tobytes()

url = "http://127.0.0.1:5000/api/plates/recognize"

print("=" * 70)
print(" FLASK END-TO-END HTTP API BENCHMARK (UPLOADED IMAGE ANPR)")
print("=" * 70)

latencies = []
for i in range(1, 6):
    t0 = time.perf_counter()
    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    resp = requests.post(url, files=files)
    lat_ms = (time.perf_counter() - t0) * 1000.0
    latencies.append(lat_ms)
    
    if resp.status_code == 200:
        data = resp.json()
        print(f" Request #{i}: Latency = {lat_ms:.2f} ms | Plate = '{data.get('text')}' | Conf = {data.get('confidence')} | Status = {data.get('status')}")
    else:
        print(f" Request #{i}: HTTP {resp.status_code} Error - {resp.text}")

avg_lat = sum(latencies) / len(latencies)
print("-" * 70)
print(f" AVERAGE HTTP RESPONSE LATENCY: {avg_lat:.2f} ms ({avg_lat/1000.0:.3f} sec)")
print("=" * 70)
