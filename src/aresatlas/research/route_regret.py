import math
import heapq
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ..config import NUM_CLASSES

# Movement directions (dy, dx, Euclidean step distance)
DIRECTIONS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (1, 1, math.sqrt(2))
]

TERRAIN_WEIGHTS = {
    0: 1.0,     # Soil
    1: 1.3,     # Bedrock
    2: 6.0,     # Sand (slip hazard)
    3: 999.0,   # Big Rock (impassable obstacle)
    255: 3.5    # Unlabeled / Caution
}

def build_cost_grid(
    class_map: np.ndarray,
    terrain_weights: Dict[int, float],
    uncertainty_map: Optional[np.ndarray] = None,
    uncertainty_lambda: float = 0.0,
    ignore_index: int = 255
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a traversability cost grid:
    cost(cell) = terrain_penalty(class) + lambda * uncertainty(cell)
    Returns: (cost_grid, traversable_mask)
    """
    H, W = class_map.shape
    cost_grid = np.ones((H, W), dtype=np.float32) * terrain_weights.get(0, 1.0)
    traversable_mask = np.ones((H, W), dtype=bool)

    for c, w in terrain_weights.items():
        mask = (class_map == c)
        cost_grid[mask] = w
        if w >= 900.0: # Impassable obstacle (Big Rock)
            traversable_mask[mask] = False

    # Unlabeled / ignore index
    unlabeled = (class_map == ignore_index)
    cost_grid[unlabeled] = terrain_weights.get(ignore_index, 3.5)

    if uncertainty_map is not None and uncertainty_lambda > 0.0:
        cost_grid += (uncertainty_map * uncertainty_lambda).astype(np.float32)

    return cost_grid, traversable_mask


def run_astar(
    cost_grid: np.ndarray,
    traversable_mask: np.ndarray,
    start: Tuple[int, int], # (x, y)
    goal: Tuple[int, int],  # (x, y)
    use_heuristic: bool = True
) -> Dict[str, Any]:
    """
    Runs A* (or Dijkstra when use_heuristic=False) on cost_grid.
    Returns path, cost on this planning grid, explored nodes, elapsed time.
    """
    t0 = time.perf_counter()
    H, W = cost_grid.shape
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])

    if not (0 <= sx < W and 0 <= sy < H) or not (0 <= gx < W and 0 <= gy < H):
        return {"success": False, "error": "Coordinates out of bounds"}

    if not traversable_mask[sy, sx] or not traversable_mask[gy, gx]:
        return {"success": False, "error": "Start or goal is on impassable obstacle"}

    min_step = float(np.min(cost_grid[traversable_mask])) if np.any(traversable_mask) else 1.0

    frontier = []
    heapq.heappush(frontier, (0.0, 0.0, (sx, sy)))
    came_from = {}
    g_score = {(sx, sy): 0.0}
    explored = 0
    found = False

    while frontier:
        _, current_g, current = heapq.heappop(frontier)
        cx, cy = current
        explored += 1

        if (cx, cy) == (gx, gy):
            found = True
            break

        if current_g > g_score.get(current, float('inf')):
            continue

        for dy, dx, step_dist in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and traversable_mask[ny, nx]:
                cell_cost = (cost_grid[cy, cx] + cost_grid[ny, nx]) * 0.5
                tentative_g = current_g + (cell_cost * step_dist)
                neighbor = (nx, ny)

                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    h = math.hypot(gx - nx, gy - ny) * min_step if use_heuristic else 0.0
                    heapq.heappush(frontier, (tentative_g + h, tentative_g, neighbor))

    elapsed = (time.perf_counter() - t0) * 1000.0
    if not found:
        return {"success": False, "error": "No route found", "elapsed_ms": elapsed, "explored": explored}

    curr = (gx, gy)
    path = [curr]
    while curr in came_from:
        curr = came_from[curr]
        path.append(curr)
    path.reverse()

    return {
        "success": True,
        "path": path,
        "planned_cost": float(g_score[(gx, gy)]),
        "steps": len(path),
        "explored": explored,
        "elapsed_ms": elapsed
    }


def evaluate_route_on_ground_truth(
    path: List[Tuple[int, int]],
    gt_cost_grid: np.ndarray,
    gt_class_map: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates a proposed route on the true ground truth cost grid.
    Computes total true cost, true length, hazard exposure.
    """
    if not path:
        return {"true_cost": float('inf'), "valid": False}

    total_cost = 0.0
    total_len = 0.0
    sand_steps = 0
    rock_steps = 0
    soil_steps = 0
    bedrock_steps = 0

    for i in range(len(path)):
        px, py = path[i]
        c = gt_class_map[py, px]
        if c == 0:
            soil_steps += 1
        elif c == 1:
            bedrock_steps += 1
        elif c == 2:
            sand_steps += 1
        elif c == 3:
            rock_steps += 1

        if i > 0:
            prev_x, prev_y = path[i - 1]
            dist = math.hypot(px - prev_x, py - prev_y)
            total_len += dist
            step_cost = (gt_cost_grid[prev_y, prev_x] + gt_cost_grid[py, px]) * 0.5 * dist
            total_cost += step_cost

    total_steps = len(path)
    return {
        "valid": True,
        "true_cost": float(total_cost),
        "length_units": float(total_len),
        "steps": total_steps,
        "sand_exposure_pct": float(round((sand_steps / total_steps * 100.0), 2)) if total_steps > 0 else 0.0,
        "rock_steps": int(rock_steps),
        "sand_steps": int(sand_steps)
    }


def compute_route_regret_comparison(
    gt_class_map: np.ndarray,
    pred_class_map: np.ndarray,
    uncertainty_map: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    terrain_weights: Dict[int, float],
    uncertainty_lambda: float = 2.5
) -> Dict[str, Any]:
    """
    Core Route Regret Experiment:
    Compares:
      1. Oracle Route (R*): Optimal route planned on Ground Truth classes.
      2. Predicted Route (R_pred): Route planned on Model Predicted classes.
      3. Uncertainty-Aware Route (R_unc): Route planned on Predicted classes + lambda * uncertainty.
    Evaluates all three on the IDENTICAL Ground Truth Cost Grid.
    Computes Regret: TrueCost(R) - TrueCost(R*).
    """
    # 1. Build Ground Truth Cost Grid (Evaluation Objective)
    gt_costs, gt_traversable = build_cost_grid(gt_class_map, terrain_weights)
    
    # 2. Build Predicted Cost Grid
    pred_costs, pred_traversable = build_cost_grid(pred_class_map, terrain_weights)
    
    # 3. Build Uncertainty-Aware Cost Grid
    unc_costs, unc_traversable = build_cost_grid(
        pred_class_map, terrain_weights,
        uncertainty_map=uncertainty_map,
        uncertainty_lambda=uncertainty_lambda
    )

    # Compute Oracle Route R*
    oracle_res = run_astar(gt_costs, gt_traversable, start, goal)
    if not oracle_res["success"]:
        return {"status": "ineligible", "reason": "No feasible oracle route on ground truth"}

    oracle_eval = evaluate_route_on_ground_truth(oracle_res["path"], gt_costs, gt_class_map)
    oracle_true_cost = oracle_eval["true_cost"]

    # Compute Predicted Route R_pred
    pred_res = run_astar(pred_costs, pred_traversable, start, goal)
    pred_eval = evaluate_route_on_ground_truth(pred_res["path"], gt_costs, gt_class_map) if pred_res["success"] else None

    # Compute Uncertainty-Aware Route R_unc
    unc_res = run_astar(unc_costs, unc_traversable, start, goal)
    unc_eval = evaluate_route_on_ground_truth(unc_res["path"], gt_costs, gt_class_map) if unc_res["success"] else None

    # Calculate Regrets
    pred_regret = max(0.0, float(pred_eval["true_cost"] - oracle_true_cost)) if pred_eval else None
    pred_norm_regret = float(pred_regret / (oracle_true_cost + 1e-7)) if pred_regret is not None else None

    unc_regret = max(0.0, float(unc_eval["true_cost"] - oracle_true_cost)) if unc_eval else None
    unc_norm_regret = float(unc_regret / (oracle_true_cost + 1e-7)) if unc_regret is not None else None

    regret_savings = float(pred_regret - unc_regret) if (pred_regret is not None and unc_regret is not None) else None

    return {
        "status": "success",
        "start": list(start),
        "goal": list(goal),
        "oracle": {
            "path": oracle_res["path"],
            "true_cost": round(oracle_true_cost, 2),
            "length": round(oracle_eval["length_units"], 2),
            "sand_exposure_pct": oracle_eval["sand_exposure_pct"]
        },
        "predicted": {
            "success": pred_res["success"],
            "path": pred_res.get("path", []),
            "true_cost": round(pred_eval["true_cost"], 2) if pred_eval else None,
            "regret": round(pred_regret, 2) if pred_regret is not None else None,
            "normalized_regret": round(pred_norm_regret, 4) if pred_norm_regret is not None else None,
            "sand_exposure_pct": pred_eval["sand_exposure_pct"] if pred_eval else None
        },
        "uncertainty_aware": {
            "success": unc_res["success"],
            "path": unc_res.get("path", []),
            "true_cost": round(unc_eval["true_cost"], 2) if unc_eval else None,
            "regret": round(unc_regret, 2) if unc_regret is not None else None,
            "normalized_regret": round(unc_norm_regret, 4) if unc_norm_regret is not None else None,
            "sand_exposure_pct": unc_eval["sand_exposure_pct"] if unc_eval else None
        },
        "regret_savings_via_uncertainty": round(regret_savings, 2) if regret_savings is not None else None
    }
