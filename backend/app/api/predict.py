import os
import io
import time
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from ..config import DATA_DIR, MODELS_DIR, CLASS_NAMES, CLASS_COLORS, CostConfig
from ..ml.models import DeepLabV3Plus, SegFormerB0, AllSoilBaseline
from ..ml.uncertainty import compute_predictive_entropy, compute_margin_uncertainty, compute_confidence_map
from ..planner.cost_map import build_terrain_cost_map

router = APIRouter(prefix="/api/predict", tags=["inference"])

# Global model cache to avoid reallocating weights on every request
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LOADED_MODELS = {}

def get_or_load_model(model_name: str = "deeplabv3plus"):
    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    if model_name == "all_soil_baseline":
        model = AllSoilBaseline(num_classes=4)
        LOADED_MODELS[model_name] = model
        return model

    elif model_name == "deeplabv3plus":
        model = DeepLabV3Plus(num_classes=4).to(DEVICE)
        ckpt_path = os.path.join(MODELS_DIR, "aresatlas_deeplabv3plus.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"[+] Loaded DeepLabV3+ checkpoint from {ckpt_path}")
        else:
            print(f"[-] No checkpoint at {ckpt_path}, initialized DeepLabV3+ architecture.")
        model.eval()
        LOADED_MODELS[model_name] = model
        return model

    elif model_name == "segformer_b0":
        model = SegFormerB0(num_classes=4).to(DEVICE)
        ckpt_path = os.path.join(MODELS_DIR, "aresatlas_segformer_b0.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"[+] Loaded SegFormer-B0 checkpoint from {ckpt_path}")
        model.eval()
        LOADED_MODELS[model_name] = model
        return model

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Creates an RGB visualization of the 4-class terrain mask"""
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for c, meta in CLASS_COLORS.items():
        if c in [0, 1, 2, 3]:
            rgb[mask == c] = meta["rgb"]
        elif c == 255:
            rgb[mask == 255] = (40, 45, 55)
    return rgb


def array_to_base64_png(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@router.post("")
async def predict_terrain(
    sample_id: Optional[str] = Form(None),
    model_name: str = Form("deeplabv3plus"),
    uncertainty_threshold: float = Form(0.65),
    file: Optional[UploadFile] = File(None)
):
    """
    Executes deep learning terrain segmentation, epistemic uncertainty quantification,
    and hazard cost mapping on real Mars rover imagery.
    """
    image = None
    gt_mask = None

    if sample_id:
        img_path = os.path.join(DATA_DIR, "images", f"{sample_id}.jpg")
        if not os.path.exists(img_path):
            # check with other extensions
            for ext in [".png", ".jpeg", ".JPG"]:
                p = os.path.join(DATA_DIR, "images", f"{sample_id}{ext}")
                if os.path.exists(p):
                    img_path = p
                    break
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            
        mask_path = os.path.join(DATA_DIR, "labels", f"{sample_id}_mask.png")
        if os.path.exists(mask_path):
            gt_mask = np.array(Image.open(mask_path))

    elif file:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

    if image is None:
        raise HTTPException(status_code=400, detail="Must provide either sample_id or image file upload.")

    orig_w, orig_h = image.size
    
    # Process at standard model resolution (512x512) for speed and consistency
    proc_w, proc_h = 512, 512
    img_resized = image.resize((proc_w, proc_h), Image.Resampling.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    # Shape: (3, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    model = get_or_load_model(model_name)

    start_t = time.perf_counter()
    with torch.no_grad():
        if model_name == "all_soil_baseline":
            probs = model.predict(img_tensor).cpu().numpy()[0]
            logits = np.log(probs + 1e-7)
        else:
            logits_tensor = model(img_tensor)
            probs_tensor = F.softmax(logits_tensor, dim=1)
            probs = probs_tensor.cpu().numpy()[0]
            logits = logits_tensor.cpu().numpy()[0]
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    pred_mask = np.argmax(probs, axis=0).astype(np.uint8) # (H, W)
    
    # Compute uncertainty maps
    entropy_map = compute_predictive_entropy(probs)
    margin_map = compute_margin_uncertainty(probs)
    confidence_map = compute_confidence_map(probs)

    # Rejection mask: high uncertainty triggers autonomous teleoperation request
    high_uncertainty_mask = (entropy_map > uncertainty_threshold)
    high_uncertainty_pct = round(float(np.mean(high_uncertainty_mask)) * 100.0, 2)

    # Class distribution breakdown
    total_pixels = proc_w * proc_h
    class_breakdown = {}
    for c in [0, 1, 2, 3]:
        c_pixels = int(np.sum(pred_mask == c))
        c_pct = round((c_pixels / total_pixels) * 100.0, 2)
        class_breakdown[CLASS_NAMES[c]] = {
            "class_id": c,
            "pixels": c_pixels,
            "percentage": c_pct,
            "hazard_level": "CRITICAL" if c == 3 else ("HIGH" if c == 2 else ("MODERATE" if c == 1 else "SAFE"))
        }

    # Colorize preview masks
    color_mask_rgb = colorize_mask(pred_mask)
    
    # Build entropy heatmap (0=dark purple, 1=bright yellow)
    entropy_uint8 = (entropy_map * 255).astype(np.uint8)
    # Simple false-color mapping for entropy: red/amber for high uncertainty
    entropy_rgb = np.zeros((proc_h, proc_w, 3), dtype=np.uint8)
    entropy_rgb[:, :, 0] = entropy_uint8
    entropy_rgb[:, :, 1] = (entropy_uint8 * 0.4).astype(np.uint8)
    entropy_rgb[:, :, 2] = (255 - entropy_uint8)

    # Build cost map
    cost_config = CostConfig()
    cost_grid, traversable_mask, cost_telemetry = build_terrain_cost_map(pred_mask, entropy_map, cost_config)

    # Normalize cost grid for visualization
    norm_cost = np.clip(cost_grid / 15.0, 0.0, 1.0)
    cost_uint8 = (norm_cost * 255).astype(np.uint8)
    cost_rgb = np.zeros((proc_h, proc_w, 3), dtype=np.uint8)
    cost_rgb[:, :, 0] = cost_uint8
    cost_rgb[:, :, 1] = (255 - cost_uint8)
    cost_rgb[cost_grid >= 900.0] = [255, 0, 0] # impassable rocks in pure crimson

    # If ground truth mask exists, compute sample-level agreement
    gt_agreement = None
    if gt_mask is not None:
        gt_resized = Image.fromarray(gt_mask).resize((proc_w, proc_h), Image.Resampling.NEAREST)
        gt_arr = np.array(gt_resized)
        valid = (gt_arr != 255)
        if np.any(valid):
            match = (pred_mask[valid] == gt_arr[valid])
            gt_agreement = {
                "pixel_accuracy": round(float(np.mean(match)), 4),
                "evaluated_pixels": int(np.sum(valid)),
                "unlabeled_ignored_pixels": int(np.sum(~valid))
            }

    mask_b64 = array_to_base64_png(color_mask_rgb)
    entropy_b64 = array_to_base64_png(entropy_rgb)
    cost_b64 = array_to_base64_png(cost_rgb)

    return {
        "sample_id": sample_id,
        "model_name": model_name,
        "device": str(DEVICE),
        "inference_latency_ms": round(elapsed_ms, 2),
        "resolution": {"width": proc_w, "height": proc_h, "original_width": orig_w, "original_height": orig_h},
        "class_breakdown": class_breakdown,
        "uncertainty_telemetry": {
            "mean_entropy": round(float(np.mean(entropy_map)), 4),
            "mean_confidence": round(float(np.mean(confidence_map)), 4),
            "mean_margin": round(float(np.mean(margin_map)), 4),
            "high_uncertainty_area_pct": high_uncertainty_pct,
            "autonomous_safety_state": "PASS" if high_uncertainty_pct < 15.0 else "CAUTION_FLAGGED"
        },
        "ground_truth_agreement": gt_agreement,
        "cost_telemetry": cost_telemetry,
        "overlay_png_base64": mask_b64,
        "overlay_base64": mask_b64,
        "overlay": mask_b64,
        "visualizations": {
            "pred_mask_b64": mask_b64,
            "entropy_heatmap_b64": entropy_b64,
            "cost_heatmap_b64": cost_b64
        }
    }
