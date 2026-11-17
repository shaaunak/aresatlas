import heapq
import math
import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# 8-directional neighbor movements with Euclidean step lengths
# (dy, dx, distance)
DIRECTIONS = [
    (-1, 0, 1.0),       # North
    (1, 0, 1.0),        # South
    (0, -1, 1.0),       # West
    (0, 1, 1.0),        # East
    (-1, -1, math.sqrt(2)),  # North-West
    (-1, 1, math.sqrt(2)),   # North-East
    (1, -1, math.sqrt(2)),   # South-West
    (1, 1, math.sqrt(2))     # South-East
]

def plan_path(
    cost_map: np.ndarray,
    traversable_mask: np.ndarray,
    seg_mask: np.ndarray,
    start: Tuple[int, int],  # (x, y)
    goal: Tuple[int, int],   # (x, y)
    algorithm: str = "astar",  # "astar", "dijkstra", or "shortest"
    uncertainty_map: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Plans path between start and goal on the Martian terrain cost grid.

    Parameters:
      cost_map: (H, W) float cost grid
      traversable_mask: (H, W) boolean mask
      seg_mask: (H, W) terrain class labels
      start: (x, y) coordinates
      goal: (x, y) coordinates
      algorithm: "astar" (terrain-aware), "dijkstra", or "shortest" (unweighted baseline)
    """
    start_time = time.perf_counter()
    H, W = cost_map.shape

    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])

    # Validate bounds
    if not (0 <= sx < W and 0 <= sy < H):
        return {"success": False, "error": f"Start coordinate ({sx}, {sy}) out of bounds"}
    if not (0 <= gx < W and 0 <= gy < H):
        return {"success": False, "error": f"Goal coordinate ({gx}, {gy}) out of bounds"}

    # Check if start or goal is inside hard obstacle
    if algorithm != "shortest" and not traversable_mask[sy, sx]:
        # Try to find nearest traversable neighbor
        found = False
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < H and 0 <= nx < W and traversable_mask[ny, nx]:
                    sy, sx = ny, nx
                    found = True
                    break
            if found:
                break
        if not found:
            return {"success": False, "error": "Start point is located in impassable rock hazard"}

    if algorithm != "shortest" and not traversable_mask[gy, gx]:
        found = False
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < H and 0 <= nx < W and traversable_mask[ny, nx]:
                    gy, gx = ny, nx
                    found = True
                    break
            if found:
                break
        if not found:
            return {"success": False, "error": "Goal point is located in impassable rock hazard"}

    # Grid for pathfinding
    if algorithm == "shortest":
        # Unweighted shortest path: ignores terrain weights except absolute rocks
        working_costs = np.ones((H, W), dtype=np.float32)
        # Still can't drive through rocks in basic reality
        working_costs[seg_mask == 3] = 9999.0
    else:
        working_costs = cost_map

    # Minimum step cost for admissible A* heuristic
    min_step_cost = float(np.min(working_costs[traversable_mask])) if np.any(traversable_mask) else 1.0

    # Priority queue: (priority, g_score, (x, y))
    frontier = []
    heapq.heappush(frontier, (0.0, 0.0, (sx, sy)))

    came_from = {}
    g_score = { (sx, sy): 0.0 }
    explored_nodes = 0

    success = False

    while frontier:
        _, current_g, current = heapq.heappop(frontier)
        cx, cy = current
        explored_nodes += 1

        if (cx, cy) == (gx, gy):
            success = True
            break

        if current_g > g_score.get(current, float('inf')):
            continue

        for dy, dx, step_dist in DIRECTIONS:
            nx, ny = cx + dx, cy + dy

            if 0 <= nx < W and 0 <= ny < H:
                # Impassable check
                if algorithm != "shortest" and not traversable_mask[ny, nx]:
                    continue
                if algorithm == "shortest" and seg_mask[ny, nx] == 3:
                    continue

                # Node cost is average of current and neighbor cell multiplied by Euclidean step
                cell_cost = (working_costs[cy, cx] + working_costs[ny, nx]) * 0.5
                tentative_g = current_g + (cell_cost * step_dist)

                neighbor = (nx, ny)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current

                    # Heuristic
                    if algorithm == "astar" or algorithm == "shortest":
                        # Euclidean distance heuristic
                        h = math.hypot(gx - nx, gy - ny) * min_step_cost
                        priority = tentative_g + h
                    else:  # dijkstra
                        priority = tentative_g

                    heapq.heappush(frontier, (priority, tentative_g, neighbor))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    if not success:
        return {
            "success": False,
            "error": "No viable path found between waypoints (path obstructed by rock barriers)",
            "elapsed_ms": round(elapsed_ms, 2),
            "explored_nodes": explored_nodes
        }

    # Reconstruct path
    curr = (gx, gy)
    path = [curr]
    while curr in came_from:
        curr = came_from[curr]
        path.append(curr)
    path.reverse()

    # Compute path telemetry
    total_length = 0.0
    total_cost = 0.0
    sand_steps = 0
    bedrock_steps = 0
    soil_steps = 0
    rock_steps = 0
    uncertainties = []

    rock_y, rock_x = np.where(seg_mask == 3)
    has_rocks = len(rock_y) > 0

    min_rock_clearance = float('inf')

    for i in range(len(path)):
        px, py = path[i]
        c = seg_mask[py, px]
        if c == 0:
            soil_steps += 1
        elif c == 1:
            bedrock_steps += 1
        elif c == 2:
            sand_steps += 1
        elif c == 3:
            rock_steps += 1

        if uncertainty_map is not None:
            uncertainties.append(float(uncertainty_map[py, px]))

        if i > 0:
            prev_x, prev_y = path[i - 1]
            dist = math.hypot(px - prev_x, py - prev_y)
            total_length += dist
            step_cost = (cost_map[prev_y, prev_x] + cost_map[py, px]) * 0.5 * dist
            total_cost += step_cost

        # Approximate rock clearance
        if has_rocks and i % 5 == 0:  # sample every 5 steps for speed
            dists = np.hypot(rock_x - px, rock_y - py)
            min_d = np.min(dists)
            if min_d < min_rock_clearance:
                min_rock_clearance = float(min_d)

    total_steps = len(path)
    sand_exposure_pct = float(round((sand_steps / total_steps) * 100.0, 1)) if total_steps > 0 else 0.0
    avg_uncertainty = float(round(float(np.mean(uncertainties)), 4)) if uncertainties else 0.0
    if min_rock_clearance == float('inf'):
        min_rock_clearance = 50.0

    return {
        "success": True,
        "algorithm": algorithm,
        "path": [[int(p[0]), int(p[1])] for p in path],
        "total_steps": int(total_steps),
        "path_length_units": float(round(total_length, 2)),
        "total_cost": float(round(total_cost, 2)),
        "sand_exposure_pct": float(sand_exposure_pct),
        "min_rock_clearance_units": float(round(min_rock_clearance, 1)),
        "terrain_breakdown": {
            "soil_steps": int(soil_steps),
            "bedrock_steps": int(bedrock_steps),
            "sand_steps": int(sand_steps),
            "rock_steps": int(rock_steps)
        },
        "average_uncertainty": float(avg_uncertainty),
        "elapsed_ms": float(round(elapsed_ms, 2)),
        "explored_nodes": int(explored_nodes)
    }
