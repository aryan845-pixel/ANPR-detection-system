import os
import cv2
import time
import numpy as np
from app import extract_plate_text_with_easyocr, assess_plate_quality, paddle_ocr, ocr_reader

def run_evaluation():
    print("=" * 70)
    print(" ANPR OCR EVALUATION BENCHMARK (PADDLEOCR + EASYOCR)")
    print("=" * 70)

    test_images = [
        os.path.join("major", "sample_number_plate.png"),
        os.path.join("major", "sample_test.png")
    ]

    print(f"[*] PaddleOCR Active: {paddle_ocr is not None}")
    print(f"[*] EasyOCR Active  : {ocr_reader is not None}\n")

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"[!] Warning: Test image not found at '{img_path}'")
            continue

        print("-" * 60)
        print(f" TESTING IMAGE: {img_path}")
        print("-" * 60)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Could not read image file: {img_path}")
            continue

        # Assess plate quality
        quality = assess_plate_quality(img)
        print(f" RAW CROP SIZE    : {quality['width']}x{quality['height']} px")
        print(f" ASPECT RATIO     : {quality['aspect_ratio']}")
        print(f" SHARPNESS (LAP)  : {quality['sharpness']}")
        print(f" QUALITY SCORE    : {quality['score']} / 100 ({quality['status']})")

        t0 = time.time()
        plate_text, conf, proc_crop, qual = extract_plate_text_with_easyocr(ocr_reader, img)
        latency = int((time.time() - t0) * 1000)

        proc_h, proc_w = proc_crop.shape[:2]
        print(f" PROCESSED SIZE   : {proc_w}x{proc_h} px")
        print(f" FINAL OCR TEXT   : \"{plate_text}\"")
        print(f" OCR CONFIDENCE   : {conf*100:.1f}%")
        print(f" LATENCY          : {latency} ms")
        print("=" * 70)

if __name__ == "__main__":
    run_evaluation()
