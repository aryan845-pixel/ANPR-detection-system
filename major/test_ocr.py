import os
import cv2
import numpy as np
from ocr_service import OCRService

def create_sample_image(filename="sample_test.png"):
    """
    Creates a synthetic test image with text for verification.
    """
    # Create a clean dark slate image
    img = np.ones((300, 700, 3), dtype=np.uint8) * 240
    
    # Add title and text
    cv2.putText(img, "EasyOCR Reader Setup", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 30, 30), 3)
    cv2.putText(img, "Status: Model Active 100%", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 130, 20), 2)
    cv2.putText(img, "Invoice ID: #8942-OCR", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 200), 2)
    
    cv2.imwrite(filename, img)
    print(f"[Test] Sample test image created at: {filename}")
    return filename

if __name__ == "__main__":
    test_img_path = "sample_test.png"
    create_sample_image(test_img_path)
    
    print("\n--- Running EasyOCR Extraction ---")
    result = OCRService.read_text(test_img_path, languages=['en'])
    
    if result["success"]:
        print(f"\n[SUCCESS] Extracted Full Text:\n--> \"{result['full_text']}\"\n")
        print("Detected Words Breakdown:")
        for idx, item in enumerate(result["results"], 1):
            print(f" {idx}. '{item['text']}' | Confidence: {item['confidence']*100:.1f}%")
        print(f"\nAnnotated visualization generated successfully!")
    else:
        print(f"\n[ERROR] OCR Failed: {result.get('error')}")
