"""
Configuration and constants for voting-based ensemble methods.

Top-3 models (by mAP@0.5): YOLOv11, YOLOv5, YOLOv8
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Dataset paths
DATASET_DIR = PROJECT_ROOT / "datasets" / "pcb-filtered-yolov8"
TEST_IMAGES_DIR = DATASET_DIR / "test" / "images"
TEST_LABELS_DIR = DATASET_DIR / "test" / "labels"

# Model paths (top-3 by mAP@0.5)
RUNS_DIR = PROJECT_ROOT / "runs"
MODEL_PATHS = {
    "yolov11": RUNS_DIR / "yolov11" / "pcb-filtered" / "weights" / "best.pt",
    "yolov5": RUNS_DIR / "yolov5_ultralytics" / "yolov5s-ultralytics2" / "weights" / "best.pt",
    "yolov8": RUNS_DIR / "yolov8" / "pcb-filtered" / "weights" / "best.pt",
}

# Model order (ranked by mAP@0.5, descending)
MODEL_ORDER = ["yolov11", "yolov5", "yolov8"]

# Model performance (from individual evaluations)
MODEL_PERFORMANCE = {
    "yolov11": {"mAP50": 0.551, "rank": 1},
    "yolov5": {"mAP50": 0.548, "rank": 2},
    "yolov8": {"mAP50": 0.544, "rank": 3},
}

# Class configuration
CLASS_NAMES = ["IC", "Capacitor", "Connector", "Electrolytic Capacitor"]
NUM_CLASSES = 4

# Detection parameters
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25

# Method 2: Static weight configurations
TOP3_WEIGHTS = {
    "performance_based": {
        "yolov11": 0.50,
        "yolov5": 0.30,
        "yolov8": 0.20,
    },
    "moderate_decay": {
        "yolov11": 0.40,
        "yolov5": 0.35,
        "yolov8": 0.25,
    },
    "equal": {
        "yolov11": 0.34,
        "yolov5": 0.33,
        "yolov8": 0.33,
    },
}

# Method 3: Grid search parameters
GRID_SEARCH_STEP = 0.1
GRID_SEARCH_FINE_STEP = 0.05
GRID_SEARCH_METRIC = "mAP50"

# Output directories
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "voting_results"
CACHE_DIR = OUTPUT_DIR / "cache"
