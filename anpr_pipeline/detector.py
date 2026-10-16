"""
detector.py — YOLOv8 plate detection (CPU, single-load, warmup)
================================================================
Wraps the Ultralytics YOLO model.  Loaded once at startup, then called
per-frame via detect(). Returns a list of Detection namedtuples.
"""

import time
import torch
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional

from . import config


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    cx: int           # centre-x
    cy: int           # centre-y

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def iou(self, other: "Detection") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class PlateDetector:
    """
    Single-instance YOLO detector.
    Call detect(frame_bgr) → list[Detection]
    """

    def __init__(self, model_path: str = config.MODEL_PATH,
                 imgsz: int = config.IMAGE_SIZE,
                 conf: float = config.CONFIDENCE_THRESHOLD,
                 iou: float = config.IOU_THRESHOLD):

        from ultralytics import YOLO
        print(f"[Detector] Loading YOLO model: {model_path}")
        self._model = YOLO(model_path)
        self._model.to(config.DEVICE)
        self._imgsz = imgsz
        self._conf  = conf
        self._iou   = iou

        # CPU thread tuning
        import os
        optimal = max(1, (os.cpu_count() or 4) // 2)
        torch.set_num_threads(optimal)

        self._warmup()
        print(f"[Detector] Ready. Classes: {self._model.names}")

    def _warmup(self):
        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        with torch.inference_mode():
            self._model.predict(source=dummy, imgsz=self._imgsz,
                                device=config.DEVICE, verbose=False)
        print("[Detector] CPU warmup complete.")

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[Detection], float]:
        """
        Returns (detections, inference_time_ms).
        Runs under torch.inference_mode() for maximum CPU throughput.
        """
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()

        with torch.inference_mode():
            results = self._model.predict(
                source=frame_bgr,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._imgsz,
                device=config.DEVICE,
                verbose=False,
                stream=False,
            )

        infer_ms = (time.perf_counter() - t0) * 1000.0
        detections: list[Detection] = []

        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1 = max(0, int(xyxy[0]))
                y1 = max(0, int(xyxy[1]))
                x2 = min(w, int(xyxy[2]))
                y2 = min(h, int(xyxy[3]))
                conf = float(box.conf[0].cpu().numpy())

                if (x2 - x1) > 15 and (y2 - y1) > 8:
                    detections.append(Detection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=conf,
                        cx=(x1 + x2) // 2,
                        cy=(y1 + y2) // 2,
                    ))

        return detections, infer_ms
