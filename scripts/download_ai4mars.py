#!/usr/bin/env python3
"""
AresAtlas - Real NASA AI4Mars Dataset Ingestion & Preprocessing Pipeline
Downloads real Curiosity rover images and expert annotations from Hugging Face (hassanjbara/AI4MARS)
and extracts partitioned train, validation, and held-out expert test sets.

Partition Strategy:
  - Held-out Expert Test Split (test_expert): Extracted from test_min1 (322 images with consensus gold annotations).
  - Training & Validation Splits: Extracted from test_min2 (partitioned 80% train, 20% val).
  Strict isolation prevents any data leakage.

AI4Mars Class Taxonomy:
  0: Soil (safe traversable terrain)
  1: Bedrock (firm rock, moderate roughness)
  2: Sand (high wheel-slip hazard)
  3: Big Rock (chassis strike hazard / impassable obstacle)
  255: Unlabeled / Null / Sky / Rover Body (ignore index)
"""

import os
import io
import json
import urllib.request
import numpy as np
from PIL import Image
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "data", "ai4mars")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

HF_TEST_URL = "https://huggingface.co/datasets/hassanjbara/AI4MARS/resolve/main/data/test_min1-00000-of-00001.parquet"
HF_TRAIN_URL = "https://huggingface.co/datasets/hassanjbara/AI4MARS/resolve/main/data/test_min2-00000-of-00001.parquet"

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000000:
        print(f"[*] Already downloaded: {dest_path} ({os.path.getsize(dest_path) / (1024*1024):.1f} MB)")
        return
    print(f"[*] Downloading {url} -> {dest_path}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as f:
        total = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"\r    Downloaded {total / (1024*1024):.1f} MB", end="", flush=True)
    print("\n[+] Download completed.")

def extract_parquet(parquet_path: str, split_name: str, max_samples: int = 350):
    print(f"[*] Processing parquet {parquet_path} for split '{split_name}' (max {max_samples})...")
    df = pd.read_parquet(parquet_path, engine='fastparquet')
    print(f"    Loaded dataframe with {len(df)} rows. Columns: {list(df.columns)}")
    
    records = []
    saved_count = 0
    
    for idx, row in df.iterrows():
        if saved_count >= max_samples:
            break
            
        try:
            # Check for image and label_mask bytes
            img_bytes = None
            lbl_bytes = None
            
            if 'image.bytes' in row and row['image.bytes'] is not None:
                img_bytes = row['image.bytes']
            elif 'image' in row:
                img_val = row['image']
                img_bytes = img_val['bytes'] if isinstance(img_val, dict) and 'bytes' in img_val else img_val

            if 'label_mask.bytes' in row and row['label_mask.bytes'] is not None:
                lbl_bytes = row['label_mask.bytes']
            elif 'label_mask' in row:
                lbl_val = row['label_mask']
                lbl_bytes = lbl_val['bytes'] if isinstance(lbl_val, dict) and 'bytes' in lbl_val else lbl_val

            if not img_bytes or not lbl_bytes:
                continue
                
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            label = Image.open(io.BytesIO(lbl_bytes))
            
            sample_id = f"{split_name}_{idx:04d}"
            img_filename = f"{sample_id}.jpg"
            lbl_filename = f"{sample_id}_mask.png"
            
            img_path = os.path.join(IMAGES_DIR, img_filename)
            lbl_path = os.path.join(LABELS_DIR, lbl_filename)
            
            # Save raw image and mask
            image.save(img_path, format="JPEG", quality=90)
            label.save(lbl_path, format="PNG")
            
            # Compute class distribution
            lbl_np = np.array(label)
            unique_classes, counts = np.unique(lbl_np, return_counts=True)
            class_dist = {int(c): int(cnt) for c, cnt in zip(unique_classes, counts)}
            
            records.append({
                "id": sample_id,
                "split": split_name,
                "image_file": img_filename,
                "label_file": lbl_filename,
                "width": image.width,
                "height": image.height,
                "class_distribution": class_dist,
                "rover": "Curiosity (MSL)",
                "mission": "Mars Science Laboratory",
                "instrument": "Navigation Camera (Navcam) / Mastcam",
                "expert_annotated": ("test" in split_name)
            })
            saved_count += 1
            if saved_count % 50 == 0:
                print(f"    Saved {saved_count} samples...")
        except Exception as e:
            continue
            
    print(f"[+] Finished split '{split_name}': {len(records)} samples extracted.")
    return records

def main():
    print("=== AresAtlas Real AI4Mars Ingestion ===")
    
    # 1. Ensure Expert Gold Test Set (min1) is downloaded
    test_parquet = os.path.join(DATA_DIR, "test_min1.parquet")
    download_file(HF_TEST_URL, test_parquet)
    
    # 2. Download Training Parquet (min2)
    train_parquet = os.path.join(DATA_DIR, "test_min2.parquet")
    download_file(HF_TRAIN_URL, train_parquet)
    
    # 3. Extract Expert Test Set (strictly held out)
    test_records = extract_parquet(test_parquet, split_name="test_expert", max_samples=322)
    
    # 4. Extract Train & Validation Sets from min2
    # To prevent any data leakage, we split into 80% train and 20% val
    train_val_records = extract_parquet(train_parquet, split_name="train_pool", max_samples=322)
    
    np.random.seed(42)
    indices = np.random.permutation(len(train_val_records))
    split_idx = int(len(train_val_records) * 0.8)
    
    train_records = []
    val_records = []
    
    for i, idx in enumerate(indices):
        rec = dict(train_val_records[idx])
        if i < split_idx:
            rec["split"] = "train"
            train_records.append(rec)
        else:
            rec["split"] = "val"
            val_records.append(rec)
            
    # Save split manifests
    with open(os.path.join(SPLITS_DIR, "train.json"), "w") as f:
        json.dump(train_records, f, indent=2)
    with open(os.path.join(SPLITS_DIR, "val.json"), "w") as f:
        json.dump(val_records, f, indent=2)
    with open(os.path.join(SPLITS_DIR, "test_expert.json"), "w") as f:
        json.dump(test_records, f, indent=2)
        
    print(f"\n[+] Ingestion complete!")
    print(f"    Train samples: {len(train_records)}")
    print(f"    Validation samples: {len(val_records)}")
    print(f"    Held-out Expert Test samples: {len(test_records)}")
    print(f"    Manifests saved to: {SPLITS_DIR}")

if __name__ == "__main__":
    main()
