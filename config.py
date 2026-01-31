import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create directories
for directory in [UPLOAD_DIR, OUTPUT_DIR, SNAPSHOT_DIR, LOG_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
MAX_TRACK_AGE = 30

# Alert thresholds (in seconds)
THEFT_THRESHOLD = 3.0
SUSPICIOUS_THRESHOLD = 10.0
LOST_THRESHOLD = 15.0

# Entry/Exit zones
ENTRY_ZONE_Y = 0.1
EXIT_ZONE_Y = 0.9

# Tracked object classes
TRACKED_OBJECTS = {
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    28: 'suitcase',
    67: 'cell phone',
    73: 'laptop',
    76: 'keyboard',
    77: 'mouse',
    84: 'book',
    63: 'laptop'
}

# Video processing
TARGET_FPS = 3
VIDEO_FORMATS = ['mp4', 'avi', 'mov', 'mkv']
