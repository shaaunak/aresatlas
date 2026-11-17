import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from typing import Dict, Any, Tuple
from ..config import CostConfig

def build_terrain_cost_map(
    seg_mask: np.ndarray,
    uncertainty_map: np.ndarray = None,
    config: CostConfig = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Constructs a traversability cost grid from terrain class segmentation
    and epistemic uncertainty maps.

    Parameters:
      seg_mask: (H, W) array with classes {0: Soil, 1: Bedrock, 2: Sand, 3: Big Rock, 255: Unlabeled}
      uncertainty_map: (H, W) normalized entropy/uncertainty [0.0, 1.0]
      config: CostConfig weighting parameters

    Returns:
      cost_map: (H, W) float32 traversal costs
      traversable_mask: (H, W) boolean mask of valid driving nodes
      telemetry: terrain cost distribution telemetry
    """
    if config is None:
        config = CostConfig()

    H, W = seg_mask.shape
    cost_map = np.ones((H, W), dtype=np.float32) * config.soil_weight

    # Class base costs
    cost_map[seg_mask == 0] = config.soil_weight
    cost_map[seg_mask == 1] = config.bedrock_weight
    cost_map[seg_mask == 2] = config.sand_weight
    cost_map[seg_mask == 255] = 3.5  # Unlabeled/occluded caution cost

    # Add uncertainty penalty
    if uncertainty_map is not None:
        cost_map += (uncertainty_map * config.uncertainty_penalty).astype(np.float32)

    # Big Rocks are hard impassable obstacles
    rock_mask = (seg_mask == 3)
    
    # Apply rover footprint safety dilation around rocks
    if config.safety_buffer_radius > 0 and np.any(rock_mask):
        structure = np.ones((config.safety_buffer_radius * 2 + 1, config.safety_buffer_radius * 2 + 1))
        inflated_rocks = binary_dilation(rock_mask, structure=structure)
        
        # Smooth proximity gradient using Euclidean Distance Transform
        dist_from_rocks = distance_transform_edt(~rock_mask)
        buffer_zone = (dist_from_rocks <= config.safety_buffer_radius) & (~rock_mask)
        # Closer to rock -> higher penalty
        rock_proximity_penalty = (1.0 - (dist_from_rocks[buffer_zone] / (config.safety_buffer_radius + 1e-5))) * 15.0
        cost_map[buffer_zone] += rock_proximity_penalty.astype(np.float32)
    else:
        inflated_rocks = rock_mask

    # Mark rocks as impassable
    cost_map[rock_mask] = config.rock_weight
    traversable_mask = ~inflated_rocks

    telemetry = {
        "mean_traversable_cost": float(np.mean(cost_map[traversable_mask])) if np.any(traversable_mask) else 0.0,
        "max_cost": float(np.max(cost_map)),
        "rock_obstacle_pixels": int(np.sum(rock_mask)),
        "safety_buffer_pixels": int(np.sum(inflated_rocks) - np.sum(rock_mask)),
        "sand_pixels": int(np.sum(seg_mask == 2)),
        "soil_pixels": int(np.sum(seg_mask == 0)),
        "bedrock_pixels": int(np.sum(seg_mask == 1))
    }

    return cost_map, traversable_mask, telemetry
