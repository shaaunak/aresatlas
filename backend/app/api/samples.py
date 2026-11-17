import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict, Any

from ..config import DATA_DIR, SPLITS_DIR, CLASS_NAMES

router = APIRouter(prefix="/api/samples", tags=["samples"])

@router.get("", response_model=List[Dict[str, Any]])
async def list_samples(limit: int = 50, split: str = "all"):
    """
    Returns available real NASA AI4Mars rover images and metadata.
    """
    splits_dir = SPLITS_DIR
    records = []

    # Read expert test split first, then train/val
    split_files = ["test_expert.json", "val.json", "train.json"]
    for sf in split_files:
        path = os.path.join(splits_dir, sf)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                for item in data:
                    if split == "all" or item.get("split") == split:
                        records.append(item)

    if not records:
        # If dataset not yet extracted, check images directory directly
        images_dir = os.path.join(DATA_DIR, "images")
        if os.path.exists(images_dir):
            for fname in sorted(os.listdir(images_dir))[:limit]:
                if fname.endswith((".jpg", ".png", ".jpeg")):
                    sample_id = os.path.splitext(fname)[0]
                    records.append({
                        "id": sample_id,
                        "split": "test_expert" if "test" in fname else "train",
                        "image_file": fname,
                        "label_file": f"{sample_id}_mask.png",
                        "width": 1024,
                        "height": 1024,
                        "rover": "Curiosity (MSL)",
                        "mission": "Mars Science Laboratory",
                        "instrument": "Left Navcam / Mastcam",
                        "expert_annotated": "test" in fname
                    })

    return records[:limit]


@router.get("/{sample_id}/image")
async def get_sample_image(sample_id: str):
    images_dir = os.path.join(DATA_DIR, "images")
    for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
        path = os.path.join(images_dir, f"{sample_id}{ext}")
        if os.path.exists(path):
            return FileResponse(path, media_type="image/jpeg")
            
    # Try direct filename
    path = os.path.join(images_dir, sample_id)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail=f"Image for {sample_id} not found")


@router.get("/{sample_id}/mask")
async def get_sample_mask(sample_id: str):
    labels_dir = os.path.join(DATA_DIR, "labels")
    path = os.path.join(labels_dir, f"{sample_id}_mask.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
        
    path = os.path.join(labels_dir, f"{sample_id}.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")

    raise HTTPException(status_code=404, detail=f"Mask for {sample_id} not found")
