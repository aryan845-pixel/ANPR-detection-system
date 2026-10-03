import os
import io
import base64
import cv2
import numpy as np
from PIL import Image
import easyocr

class OCRService:
    """
    Singleton / Cached EasyOCR Service for high-performance OCR reading.
    Supports English ('en'), Hindi ('hi'), and multiple languages.
    """
    _readers = {}

    @classmethod
    def get_reader(cls, lang_list=('en',), gpu=None):
        """
        Get or initialize an EasyOCR Reader instance cached by language configuration.
        """
        key = tuple(sorted(lang_list))
        if key not in cls._readers:
            import torch
            use_gpu = gpu if gpu is not None else torch.cuda.is_available()
            print(f"[EasyOCR] Initializing reader for languages {key} (GPU enabled: {use_gpu})...")
            cls._readers[key] = easyocr.Reader(list(key), gpu=use_gpu, verbose=False)
            print(f"[EasyOCR] Reader for {key} ready!")
        return cls._readers[key]

    @classmethod
    def process_image_input(cls, image_input):
        """
        Convert various image inputs (filepath, bytes, PIL Image, numpy array, base64)
        into an OpenCV RGB numpy array.
        """
        if isinstance(image_input, str):
            if image_input.startswith("data:image") or len(image_input) > 500:
                # Base64 string
                if "," in image_input:
                    image_input = image_input.split(",", 1)[1]
                img_bytes = base64.b64decode(image_input)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                return np.array(pil_img)
            else:
                # File path
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
                # Assume OpenCV BGR if array passed from cv2.imread
                return cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            return image_input
        else:
            raise ValueError("Unsupported image input type")

    @classmethod
    def read_text(cls, image_input, languages=('en',), detail=1, paragraph=False):
        """
        Main method to read text from an image.
        
        Args:
            image_input: Filepath, bytes, PIL Image, numpy array, or Base64 string.
            languages: Tuple or list of language codes (e.g. ['en'], ['en', 'hi']).
            detail: 1 to return bounding box & confidence, 0 for simple text.
            paragraph: True to combine text into paragraphs.

        Returns:
            dict containing:
                - success (bool)
                - full_text (str)
                - results (list of dicts with text, confidence, bbox)
                - count (int)
                - annotated_image_base64 (str)
        """
        try:
            rgb_image = cls.process_image_input(image_input)
            reader = cls.get_reader(lang_list=languages)
            
            raw_results = reader.readtext(rgb_image, detail=detail, paragraph=paragraph)
            
            formatted_results = []
            full_text_list = []
            
            # Copy image for visualization (convert RGB back to BGR for OpenCV drawing)
            annotated_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            if detail == 1:
                for bbox, text, prob in raw_results:
                    # Convert bbox float points to int tuples
                    pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
                    
                    # Draw polygon around text
                    cv2.polylines(annotated_bgr, [pts], isClosed=True, color=(0, 215, 255), thickness=2)
                    
                    # Draw label background box
                    x, y = pts[0][0]
                    cv2.putText(
                        annotated_bgr,
                        f"{text} ({prob:.2f})",
                        (int(x), max(15, int(y) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

                    formatted_results.append({
                        "text": text,
                        "confidence": float(round(prob, 4)),
                        "bbox": [[int(pt[0]), int(pt[1])] for pt in bbox]
                    })
                    full_text_list.append(text)
            else:
                full_text_list = raw_results
                formatted_results = [{"text": t} for t in raw_results]

            # Encode annotated image to Base64
            _, buffer = cv2.imencode('.png', annotated_bgr)
            annotated_base64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

            full_text = " ".join(full_text_list)

            return {
                "success": True,
                "full_text": full_text,
                "count": len(formatted_results),
                "results": formatted_results,
                "annotated_image_base64": annotated_base64
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Quick test helper function if imported directly
def read_image_text(image_path, languages=('en',)):
    return OCRService.read_text(image_path, languages=languages)
