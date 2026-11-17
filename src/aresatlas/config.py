import os
from pathlib import Path

# Root paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src" / "aresatlas"
CONFIGS_DIR = ROOT_DIR / "configs"
REPORTS_DIR = ROOT_DIR / "reports"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
MODELS_DIR = ROOT_DIR / "backend" / "models"
DATA_DIR = ROOT_DIR / "backend" / "data" / "ai4mars"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
SPLITS_DIR = DATA_DIR / "splits"
BENCHMARK_FILE = ROOT_DIR / "backend" / "data" / "benchmark_results.json"

# Ensure runtime directories exist
for p in [REPORTS_DIR, ARTIFACTS_DIR, FIGURES_DIR, CHECKPOINTS_DIR, IMAGES_DIR, LABELS_DIR, SPLITS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# AI4Mars Taxonomy
CLASS_NAMES = {
    0: "Soil",
    1: "Bedrock",
    2: "Sand",
    3: "Big Rock",
    255: "Unlabeled / Ignore"
}

NUM_CLASSES = 4
IGNORE_INDEX = 255

CLASS_COLORS = {
    0: {"name": "Soil", "color": "#10B981", "rgb": (16, 185, 129), "risk": "SAFE"},
    1: {"name": "Bedrock", "color": "#06B6D4", "rgb": (6, 182, 212), "risk": "MODERATE"},
    2: {"name": "Sand", "color": "#F59E0B", "rgb": (245, 158, 11), "risk": "HIGH_SLIP"},
    3: {"name": "Big Rock", "color": "#EF4444", "rgb": (239, 68, 68), "risk": "LETHAL_STRIKE"},
    255: {"name": "Unlabeled", "color": "#64748B", "rgb": (100, 116, 139), "risk": "IGNORE"}
}
