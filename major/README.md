# 🚀 EasyOCR Model & Image Text Reader Setup

An end-to-end Python OCR service powered by **EasyOCR**, **PyTorch**, and **OpenCV** to extract text from any input image automatically.

---

## 📁 Project Structure

- `ocr_service.py`: Core EasyOCR module with cached model readers, GPU auto-detection, multi-language support, and bounding box image annotation.
- `app.py`: Flask Web Application & REST API (`POST /api/ocr`) with dark-mode Web UI.
- `test_ocr.py`: Verification script to test image text extraction.
- `requirements.txt`: Python package dependencies.
- `venv/`: Virtual environment containing Python 3.11 and PyTorch/EasyOCR dependencies.

---

## 🛠️ How to Use

### 1️⃣ Integration in Python Code / ML Models
Import `OCRService` from `ocr_service.py` in your model code:

```python
from ocr_service import OCRService

# 1. From an image file path
result = OCRService.read_text("sample_test.png", languages=['en'])

# 2. From OpenCV numpy array / PIL Image / Bytes / Base64
# result = OCRService.read_text(image_np, languages=['en', 'hi'])

if result["success"]:
    print("Full Extracted Text:", result["full_text"])
    for item in result["results"]:
        print(f"Text: {item['text']} | Confidence: {item['confidence']*100:.1f}%")
```

### 2️⃣ Run Web App & REST API
Start the Flask Web Server:

```bash
.\venv\Scripts\python.exe app.py
```

Open your browser at `http://127.0.0.1:5000`:
- **Drag & drop** any image to visualize detected text boxes and copy text.
- **REST API Endpoint**: Send POST requests to `http://127.0.0.1:5000/api/ocr`

#### cURL API Example:
```bash
curl -X POST -F "file=@your_image.png" -F "languages=en,hi" http://127.0.0.1:5000/api/ocr
```

---

## 🌍 Supported Languages
EasyOCR supports 80+ languages! Pass language codes like:
- `languages=['en']` (English)
- `languages=['en', 'hi']` (English + Hindi)
- `languages=['en', 'es']` (English + Spanish)
- `languages=['en', 'fr']` (English + French)
