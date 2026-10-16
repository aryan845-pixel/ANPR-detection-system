"""
camera.py — Threaded zero-lag camera capture
=============================================
Keeps the latest frame in a shared buffer. The main loop always
reads the freshest frame instead of the OpenCV internal queue.
"""

import threading
import time
import cv2


def list_available_cameras(max_check: int = 6) -> list[int]:
    """
    Scans camera indices 0 … max_check-1 and returns those that open
    successfully. Tries CAP_DSHOW first on Windows for low latency.
    """
    available = []
    for idx in range(max_check):
        # Try DirectShow backend (Windows) then default
        for backend in (cv2.CAP_DSHOW, 0):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    available.append(idx)
                    break
    return available


class ThreadedCamera:
    """
    Grabs frames continuously in a daemon thread.

    • Keeps ONLY the latest frame (no backlog).
    • Raises RuntimeError if the camera cannot be opened.
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30):
        self.index = index
        self._lock  = threading.Lock()
        self._frame = None
        self._ok    = False
        self._running = False

        # Try DirectShow first (Windows), fall back to default
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {index}. "
                               "Run list_available_cameras() to see valid indices.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)     # keep buffer tiny
        self._cap = cap

    def start(self) -> "ThreadedCamera":
        self._running = True
        t = threading.Thread(target=self._grab_loop, daemon=True)
        t.start()
        # warm-up: wait until first frame arrives
        deadline = time.time() + 3.0
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    break
            time.sleep(0.05)
        return self

    def _grab_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._ok    = True
            else:
                with self._lock:
                    self._ok = False
                time.sleep(0.01)

    def read(self):
        """Returns (True, frame) or (False, None)."""
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    def stop(self):
        self._running = False
        time.sleep(0.1)
        try:
            self._cap.release()
        except Exception:
            pass
