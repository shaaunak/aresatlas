import os
from pydantic import BaseModel
from typing import Dict, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data", "ai4mars")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BENCHMARK_FILE = os.path.join(BASE_DIR, "data", "benchmark_results.json")

# Terrain class mappings for NASA JPL AI4Mars
# 0: Soil - fine-grained, smooth, safe wheel traction
# 1: Bedrock - flat outcrop, solid substrate, low-to-moderate roughness
# 2: Sand - loose aeolian dune/ripple, severe sinkage/wheel-slip hazard (e.g. Curiosity Pahrump Hills / Opportunity Purgatory Dune)
# 3: Big Rock - boulders > 15-20cm, chassis-strike risk, collision obstacle
# 255: Unlabeled / Null / Rover frame / Horizon sky
CLASS_NAMES = {
    0: "Soil",
    1: "Bedrock",
    2: "Sand",
    3: "Big Rock",
    255: "Unlabeled / Sky"
}

# JPL Mission Palette (hex and RGB for visual mask synthesis)
CLASS_COLORS = {
    0: {"name": "Soil", "color": "#10B981", "rgb": (16, 185, 129), "risk_base": 1.0, "desc": "Compacted regolith; optimal traction"},
    1: {"name": "Bedrock", "color": "#06B6D4", "rgb": (6, 182, 212), "risk_base": 1.3, "desc": "Exposed bedrock; stable but uneven"},
    2: {"name": "Sand", "color": "#F59E0B", "rgb": (245, 158, 11), "risk_base": 6.0, "desc": "Loose sand dune; severe slip hazard"},
    3: {"name": "Big Rock", "color": "#EF4444", "rgb": (239, 68, 68), "risk_base": 999.0, "desc": "Obstacle / boulder; chassis strike risk"},
    255: {"name": "Unlabeled", "color": "#64748B", "rgb": (100, 116, 139), "risk_base": 3.0, "desc": "Unmapped / occluded / sky"}
}

NUM_CLASSES = 4
IGNORE_INDEX = 255

class CostConfig(BaseModel):
    soil_weight: float = 1.0
    bedrock_weight: float = 1.3
    sand_weight: float = 6.0
    rock_weight: float = 999.0  # Impassable barrier
    uncertainty_penalty: float = 2.5
    safety_buffer_radius: int = 3  # pixel dilation around big rocks
