import os
import io
import base64
import cv2
import re
import numpy as np
from PIL import Image

# Disable PIR and OneDNN issues on Windows CPU for PaddleOCR
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

class OCRService:
    """
    Singleton / Cached PaddleOCR + EasyOCR Service for high-performance ANPR reading.
    Supports English ('en'), Hindi ('hi'), and multiple languages.
    """
    _paddle_ocr = None
    _easyocr_readers = {}

    @classmethod
    def get_paddle_reader(cls):
        """
        Get or initialize the PaddleOCR instance (PP-OCRv6 / PP-OCRv4).
        """
        if cls._paddle_ocr is None:
            try:
                from paddleocr import PaddleOCR
                print("[PaddleOCR] Initializing PaddleOCR engine...")
                cls._paddle_ocr = PaddleOCR(
                    lang='en',
                    enable_mkldnn=False
                )
                print("[PaddleOCR] Engine ready!")
            except Exception as e:
                print(f"[PaddleOCR] Failed to initialize PaddleOCR: {e}")
                cls._paddle_ocr = False
        return cls._paddle_ocr if cls._paddle_ocr is not False else None

    @classmethod
    def get_easyocr_reader(cls, lang_list=('en',), gpu=None):
        """
        Get or initialize an EasyOCR Reader instance cached by language configuration.
        """
        key = tuple(sorted(lang_list))
        if key not in cls._easyocr_readers:
            import easyocr
            import torch
            use_gpu = gpu if gpu is not None else torch.cuda.is_available()
            print(f"[EasyOCR] Initializing reader for languages {key} (GPU enabled: {use_gpu})...")
            cls._easyocr_readers[key] = easyocr.Reader(list(key), gpu=use_gpu, verbose=False)
            print(f"[EasyOCR] Reader for {key} ready!")
        return cls._easyocr_readers[key]

    @classmethod
    def process_image_input(cls, image_input):
        """
        Convert various image inputs (filepath, bytes, PIL Image, numpy array, base64)
        into an OpenCV RGB numpy array.
        """
        if isinstance(image_input, str):
            if image_input.startswith("data:image") or len(image_input) > 500:
                if "," in image_input:
                    image_input = image_input.split(",", 1)[1]
                img_bytes = base64.b64decode(image_input)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                return np.array(pil_img)
            else:
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image path not found: {image_input}")
                pil_img = Image.open(image_input).convert('RGB')
                return np.array(pil_img)
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert('RGB')
            return np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            elif image_input.shape[2] == 3:
                return cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            return image_input
        else:
            raise ValueError("Unsupported image input type")

    @classmethod
    def clean_plate_text(cls, raw_boxes):
        """
        Clean and validate Indian plate text from OCR bounding boxes.
        """
        if not raw_boxes:
            return "", 0.0, []

        sorted_boxes = sorted(raw_boxes, key=lambda b: (b[0][0][1] // 25, b[0][0][0]))
        cleaned_words = []
        conf_sum = 0.0
        conf_count = 0

        for bbox, text, conf in sorted_boxes:
            t = text.strip().upper()
            t = re.sub(r'[^A-Z0-9]', '', t)
            if t.startswith('IND') and len(t) > 3:
                t = t[3:]
            elif t == 'IND':
                continue
            if t:
                cleaned_words.append(t)
                conf_sum += conf
                conf_count += 1

        full_text = "".join(cleaned_words)
        avg_conf = (conf_sum / max(1, conf_count)) if conf_count else 0.0
        return full_text, avg_conf, sorted_boxes

    @classmethod
    def read_text(cls, image_input, languages=('en',), detail=1, paragraph=False, preprocess=True):
        """
        Main method to read text from an image using PaddleOCR (fallback: EasyOCR).
        """
        try:
            rgb_image = cls.process_image_input(image_input)
            target_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            raw_results = []
            paddle_engine = cls.get_paddle_reader()

            if paddle_engine is not None and 'en' in languages:
                try:
                    p_res = paddle_engine.ocr(target_bgr)
                    if p_res:
                        for page in p_res:
                            if page is None:
                                continue
                            if isinstance(page, dict):
                                texts = page.get('rec_texts', [])
                                scores = page.get('rec_scores', [])
                                polys = page.get('dt_polys', [])
                                for poly, txt, score in zip(polys, texts, scores):
                                    pts = poly.tolist() if hasattr(poly, 'tolist') else poly
                                    raw_results.append((pts, str(txt), float(score)))
                            elif isinstance(page, list):
                                for line in page:
                                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                                        bbox, (txt, score) = line[0], line[1]
                                        raw_results.append((bbox, str(txt), float(score)))
                except Exception as p_err:
                    print(f"[PaddleOCR] Error during inference: {p_err}, falling back to EasyOCR")

            # Fallback to EasyOCR if PaddleOCR yielded no results or was unavailable
            if not raw_results:
                reader = cls.get_easyocr_reader(lang_list=languages)
                raw_easy = reader.readtext(
                    rgb_image,
                    detail=1,
                    paragraph=paragraph,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
                for bbox, txt, score in raw_easy:
                    raw_results.append((bbox, str(txt), float(score)))

            full_text, avg_conf, sorted_boxes = cls.clean_plate_text(raw_results)

            formatted_results = []
            for bbox, text, prob in raw_results:
                formatted_results.append({
                    "text": text,
                    "confidence": float(round(prob, 4)),
                    "bbox": [[int(pt[0]), int(pt[1])] for pt in bbox] if isinstance(bbox[0], (list, tuple, np.ndarray)) else []
                })

            # Copy preprocessed image for UI visualization
            _, prep_buf = cv2.imencode('.png', target_bgr)
            preprocessed_base64 = "data:image/png;base64," + base64.b64encode(prep_buf).decode('utf-8')

            # Draw annotations on RGB image
            annotated_bgr = target_bgr.copy()
            for bbox, text, prob in raw_results:
                try:
                    pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_bgr, [pts], isClosed=True, color=(0, 215, 255), thickness=2)
                    x, y = pts[0][0]
                    cv2.putText(annotated_bgr, f"{text} ({prob:.2f})", (int(x), max(15, int(y) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                except Exception:
                    pass

            _, buffer = cv2.imencode('.png', annotated_bgr)
            annotated_base64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

            return {
                "success": True,
                "full_text": full_text or " ".join([r['text'] for r in formatted_results]),
                "count": len(formatted_results),
                "results": formatted_results,
                "annotated_image_base64": annotated_base64,
                "preprocessed_image_base64": preprocessed_base64
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

def read_image_text(image_path, languages=('en',)):
    return OCRService.read_text(image_path, languages=languages)
