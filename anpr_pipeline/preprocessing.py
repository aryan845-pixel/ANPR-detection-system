"""
preprocessing.py  — IMMUTABLE BLACK BOX
========================================
This module contains the existing plate preprocessing pipeline exactly
as tuned and tested.  DO NOT MODIFY ANY PARAMETER, STEP, OR ORDERING.

External callers must call:
    preprocess_plate(bgr_crop) -> preprocessed_bgr_or_gray
"""

import cv2
import numpy as np


# ── Internal quality constants (read-only) ────────────────────────────────────
_MIN_CROP_HEIGHT = 18
_MIN_CROP_WIDTH  = 45
_MIN_SHARPNESS   = 15.0
_TARGET_HEIGHT   = 160       # 4× upscale target (Bicubic)


def _assess_quality(crop):
    """Returns a dict with resolution/sharpness flags. Internal use only."""
    if crop is None or crop.size == 0:
        return {"acceptable": False, "status": "INVALID_CROP",
                "width": 0, "height": 0, "sharpness": 0.0, "score": 0.0}

    h, w  = crop.shape[:2]
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(np.mean(gray))

    score = 100.0
    status = "HIGH_QUALITY"
    ok = True

    if h < _MIN_CROP_HEIGHT or w < _MIN_CROP_WIDTH:
        ok = False; status = "LOW_QUALITY_TOO_SMALL"; score -= 50.0
    elif h < 32 or w < 90:
        score -= 20.0; status = "ACCEPTABLE_LOW_RES"

    if sharp < _MIN_SHARPNESS:
        ok = False; status = "LOW_QUALITY_BLURRY"; score -= 30.0

    return {
        "acceptable": ok,
        "status":     status,
        "score":      max(0.0, min(100.0, score)),
        "width":      w,
        "height":     h,
        "sharpness":  round(sharp, 2),
        "brightness": round(bright, 2),
    }


def preprocess_plate(bgr_crop):
    """
    Existing preprocessing pipeline — TREAT AS IMMUTABLE.

    Steps (unchanged):
        1. Quality assessment gate.
        2. 4× Bicubic / Cubic upscale to target height 160 px.
        3. Returns the upscaled BGR crop ready for PaddleOCR.

    Returns:
        (preprocessed_bgr, quality_dict)
        If quality is unacceptable, preprocessed_bgr is None.
    """
    quality = _assess_quality(bgr_crop)

    if not quality["acceptable"]:
        return None, quality

    h, w = bgr_crop.shape[:2]
    if h < _TARGET_HEIGHT:
        scale = _TARGET_HEIGHT / float(h)
        new_w = int(w * scale)
        processed = cv2.resize(bgr_crop, (new_w, _TARGET_HEIGHT),
                               interpolation=cv2.INTER_CUBIC)
    else:
        processed = bgr_crop.copy()

    return processed, quality
