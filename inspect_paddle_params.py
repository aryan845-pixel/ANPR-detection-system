import os
import inspect
from paddleocr import PaddleOCR
import paddleocr._pipelines.ocr as ocr_module

print("=" * 70)
print("INSPECTING PADDLEOCR CLASS & PIPELINE ARGS")
print("=" * 70)

print("\n--- PaddleOCR.__init__ signature ---")
try:
    sig = inspect.signature(PaddleOCR.__init__)
    print(sig)
except Exception as e:
    print(e)

print("\n--- PaddleOCR.predict signature ---")
try:
    sig = inspect.signature(PaddleOCR.predict)
    print(sig)
except Exception as e:
    print(e)

print("\n--- Common args parser ---")
try:
    import paddleocr._common_args as common_args
    print(dir(common_args))
except Exception as e:
    print(e)

print("=" * 70)
