import pytest
import numpy as np
from backend.app.config import CostConfig
from backend.app.planner.cost_map import build_terrain_cost_map
from backend.app.planner.astar import plan_path

def test_cost_map_generation():
    # 50x50 test grid
    seg_mask = np.zeros((50, 50), dtype=np.uint8) # Soil (0)
    seg_mask[20:30, 20:30] = 2                    # Sand (2)
    seg_mask[35:40, 35:40] = 3                    # Big Rock (3)

    config = CostConfig(soil_weight=1.0, sand_weight=5.0, rock_weight=999.0, safety_buffer_radius=2)
    cost_map, traversable, telemetry = build_terrain_cost_map(seg_mask, config=config)

    assert cost_map.shape == (50, 50)
    assert traversable.shape == (50, 50)
    assert cost_map[5, 5] == 1.0       # Soil base cost
    assert cost_map[25, 25] == 5.0     # Sand cost
    assert cost_map[37, 37] == 999.0   # Rock barrier
    assert not traversable[37, 37]     # Rock is not traversable
    assert telemetry["sand_pixels"] == 100
    assert telemetry["rock_obstacle_pixels"] == 25

def test_astar_navigates_around_hazards():
    # Grid where middle has a solid row of sand and rock with an open soil gap on the left
    H, W = 40, 40
    seg_mask = np.zeros((H, W), dtype=np.uint8) # Soil
    # Put sand and rock barrier in row 20 from x=10 to x=39
    seg_mask[20, 10:30] = 2  # Sand
    seg_mask[20, 30:40] = 3  # Rock

    cost_map, traversable, _ = build_terrain_cost_map(seg_mask)

    # Plan from top (20, 5) to bottom (20, 35)
    start = (20, 5)
    goal = (20, 35)

    astar_res = plan_path(cost_map, traversable, seg_mask, start, goal, algorithm="astar")
    assert astar_res["success"] is True
    assert len(astar_res["path"]) > 0

    # Shortest path should pass directly through sand, while A* minimizes sand
    shortest_res = plan_path(cost_map, traversable, seg_mask, start, goal, algorithm="shortest")
    assert shortest_res["success"] is True
    assert shortest_res["sand_exposure_pct"] >= astar_res["sand_exposure_pct"]

def test_out_of_bounds_handling():
    cost_map = np.ones((20, 20), dtype=np.float32)
    traversable = np.ones((20, 20), dtype=bool)
    seg = np.zeros((20, 20), dtype=np.uint8)

    res = plan_path(cost_map, traversable, seg, (-1, 5), (10, 10))
    assert res["success"] is False
    assert "out of bounds" in res["error"]
