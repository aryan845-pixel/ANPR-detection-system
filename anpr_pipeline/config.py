import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "..", "anpr.pt")   # relative to this file

# ──────────────────────────────────────────────
# CAMERA
# ──────────────────────────────────────────────
CAMERA_INDEX   = 0      # set at runtime via --camera flag or camera_selector
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
TARGET_FPS     = 30

# ──────────────────────────────────────────────
# YOLO INFERENCE
# ──────────────────────────────────────────────
IMAGE_SIZE            = 640     # lower = faster (480 trades ~15% mAP for +30% FPS)
CONFIDENCE_THRESHOLD  = 0.35
IOU_THRESHOLD         = 0.45
DEVICE                = "cpu"

# ──────────────────────────────────────────────
# TRACKER & MOTION PREDICTION
# ──────────────────────────────────────────────
MAX_LOST_FRAMES             = 15   # frames before a track is deleted
IOU_MATCH_THRESHOLD         = 0.25 # min IoU to re-associate a detection to an existing track
CENTROID_DISTANCE_THRESHOLD = 90.0 # max centroid displacement in pixels for motion matching
DETECTION_INTERVAL          = 1    # 1 = run detector every frame; >1 = step tracker in between

# ──────────────────────────────────────────────
# CANDIDATE / BEST-FRAME SELECTION
# ──────────────────────────────────────────────
MAX_CANDIDATES_PER_TRACK = 10    # max stored crops per track (bounded memory)
CAPTURE_WINDOW_SECONDS   = 2.0   # how long to collect candidates before OCR fires

# ──────────────────────────────────────────────
# OCR SCHEDULING
# ──────────────────────────────────────────────
MAX_OCR_QUEUE_SIZE    = 4    # pending OCR jobs queue cap (drops excess to stay real-time)
OCR_RERUN_INTERVAL    = 60   # re-run OCR after N frames if plate still visible (0 = disabled)

# ──────────────────────────────────────────────
# PLATE QUALITY THRESHOLDS (used by preprocessing.py)
# ──────────────────────────────────────────────
MIN_CROP_HEIGHT  = 18
MIN_CROP_WIDTH   = 45
MIN_SHARPNESS    = 15.0
