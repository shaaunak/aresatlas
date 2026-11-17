from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import base64
import io
from PIL import Image
import os

from ..config import CostConfig, DATA_DIR
from ..planner.cost_map import build_terrain_cost_map
from ..planner.astar import plan_path

router = APIRouter(prefix="/api/route", tags=["route-planning"])

class RouteRequest(BaseModel):
    sample_id: Optional[str] = None
    start: List[int] = Field(..., description="[x, y] start coordinates")
    goal: List[int] = Field(..., description="[x, y] goal coordinates")
    soil_weight: float = Field(1.0, ge=0.5, le=10.0)
    bedrock_weight: float = Field(1.3, ge=0.5, le=10.0)
    sand_weight: float = Field(6.0, ge=1.0, le=20.0)
    rock_weight: float = Field(999.0, ge=50.0, le=9999.0)
    uncertainty_penalty: float = Field(2.5, ge=0.0, le=10.0)
    safety_buffer_radius: int = Field(3, ge=0, le=10)
    algorithms: List[str] = Field(["astar", "shortest"], description="List of algorithms to compare")

class RouteComparisonResponse(BaseModel):
    sample_id: Optional[str]
    start: List[int]
    goal: List[int]
    cost_config: Dict[str, Any]
    terrain_telemetry: Dict[str, Any]
    routes: Dict[str, Any]
    comparison_summary: Dict[str, Any]
    scientific_disclaimer: str

@router.post("", response_model=RouteComparisonResponse)
async def compute_route(req: RouteRequest):
    if len(req.start) != 2 or len(req.goal) != 2:
        raise HTTPException(status_code=400, detail="Start and goal must be [x, y] coordinates.")

    labels_dir = os.path.join(DATA_DIR, "labels")
    
    # Load mask for requested sample or default to first available
    seg_mask = None
    if req.sample_id:
        mask_path = os.path.join(labels_dir, f"{req.sample_id}_mask.png")
        if os.path.exists(mask_path):
            seg_mask = np.array(Image.open(mask_path))

    if seg_mask is None:
        # Check if there are any label masks in labels_dir
        if os.path.exists(labels_dir):
            masks = [f for f in os.listdir(labels_dir) if f.endswith(".png")]
            if masks:
                seg_mask = np.array(Image.open(os.path.join(labels_dir, masks[0])))

    if seg_mask is None:
        # Fallback synthetic Martian terrain grid for simulation tests
        H, W = 256, 256
        seg_mask = np.zeros((H, W), dtype=np.uint8)  # Soil
        seg_mask[100:150, 40:220] = 2                 # Sand dune hazard in middle
        seg_mask[60:85, 120:145] = 3                  # Big Rock boulder obstacle
        seg_mask[170:190, 80:105] = 3                 # Big Rock obstacle 2
        seg_mask[180:240, 160:240] = 1                # Bedrock zone

    # Synthetic uncertainty based on class borders if not provided
    H, W = seg_mask.shape
    uncertainty_map = np.zeros((H, W), dtype=np.float32)
    # Class boundaries have higher uncertainty
    gy, gx = np.gradient(seg_mask.astype(float))
    edge_mag = np.hypot(gx, gy)
    uncertainty_map = np.clip(edge_mag / (np.max(edge_mag) + 1e-5), 0.0, 1.0).astype(np.float32) * 0.8

    config = CostConfig(
        soil_weight=req.soil_weight,
        bedrock_weight=req.bedrock_weight,
        sand_weight=req.sand_weight,
        rock_weight=req.rock_weight,
        uncertainty_penalty=req.uncertainty_penalty,
        safety_buffer_radius=req.safety_buffer_radius
    )

    cost_map, traversable_mask, telemetry = build_terrain_cost_map(seg_mask, uncertainty_map, config)

    routes = {}
    for algo in req.algorithms:
        plan_res = plan_path(
            cost_map=cost_map,
            traversable_mask=traversable_mask,
            seg_mask=seg_mask,
            start=(req.start[0], req.start[1]),
            goal=(req.goal[0], req.goal[1]),
            algorithm=algo,
            uncertainty_map=uncertainty_map
        )
        routes[algo] = plan_res

    # Build comparison summary
    astar_res = routes.get("astar", {})
    shortest_res = routes.get("shortest", {})

    astar_success = astar_res.get("success", False)
    shortest_success = shortest_res.get("success", False)

    summary = {
        "astar_viable": astar_success,
        "shortest_viable": shortest_success,
    }

    if astar_success and shortest_success:
        astar_cost = float(astar_res.get("total_cost", 0.0))
        shortest_cost = float(shortest_res.get("total_cost", 0.0))
        astar_sand = float(astar_res.get("sand_exposure_pct", 0.0))
        shortest_sand = float(shortest_res.get("sand_exposure_pct", 0.0))
        astar_len = float(astar_res.get("path_length_units", 0.0))
        shortest_len = float(shortest_res.get("path_length_units", 0.0))
        
        cost_savings_pct = round(((shortest_cost - astar_cost) / (shortest_cost + 1e-5)) * 100.0, 1) if shortest_cost > 0 else 0.0
        sand_reduction_pct = round(shortest_sand - astar_sand, 1)

        summary.update({
            "length_delta_units": float(round(astar_len - shortest_len, 2)),
            "cost_delta": float(round(astar_cost - shortest_cost, 2)),
            "cost_savings_pct": float(cost_savings_pct),
            "sand_exposure_reduction_pct": float(sand_reduction_pct),
            "tactical_recommendation": (
                "Terrain-Aware A* path bypasses high-risk sand traps and rock buffers, "
                f"reducing hazardous sand exposure by {sand_reduction_pct}% at the cost of a {round(astar_len - shortest_len, 1)} unit detour."
                if sand_reduction_pct > 0 else
                "Terrain-Aware path coincides with direct heading as minimal hazard obstacles were detected."
            )
        })

    disclaimer = (
        "RESEARCH SIMULATION LIMITATION: This 2D path planner operates on monocular rover imagery "
        "and demonstrates cost-weighted algorithmic path selection. Operational Mars rover navigation "
        "(e.g., JPL GESTALT / AutoNav on Curiosity & Perseverance) requires calibrated stereo 3D digital "
        "elevation models (DEM), pitch/roll tilt inclinometry, slip odometry, and physical chassis suspension models."
    )

    return RouteComparisonResponse(
        sample_id=req.sample_id,
        start=req.start,
        goal=req.goal,
        cost_config=config.model_dump(),
        terrain_telemetry={k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in telemetry.items()},
        routes=routes,
        comparison_summary=summary,
        scientific_disclaimer=disclaimer
    )
