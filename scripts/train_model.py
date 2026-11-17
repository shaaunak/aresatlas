#!/usr/bin/env python3
"""
AresAtlas - Real Model Training Pipeline
Trains DeepLabV3+ and SegFormer-B0 on NASA JPL AI4Mars Curiosity rover dataset.
Uses genuine cross-entropy loss ignoring unlabeled pixels (class 255),
computes validation mIoU and per-class hazard recall, and persists weights.
"""

import os
import json
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from backend.app.config import DATA_DIR, MODELS_DIR, SPLITS_DIR, NUM_CLASSES, IGNORE_INDEX
from backend.app.ml.models import DeepLabV3Plus, SegFormerB0
from backend.app.ml.metrics import compute_confusion_matrix, evaluate_segmentation_metrics

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
os.makedirs(MODELS_DIR, exist_ok=True)

class AI4MarsDataset(Dataset):
    def __init__(self, manifest_file: str, img_size: int = 256, is_train: bool = True):
        self.img_size = img_size
        self.is_train = is_train
        self.images_dir = os.path.join(DATA_DIR, "images")
        self.labels_dir = os.path.join(DATA_DIR, "labels")
        
        with open(manifest_file, "r") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.images_dir, item["image_file"])
        lbl_path = os.path.join(self.labels_dir, item["label_file"])

        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path)

        # Resize
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        label = label.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)

        # Augmentation for training
        if self.is_train and np.random.rand() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        img_arr = np.array(image, dtype=np.float32) / 255.0
        # Normalize to standard ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_arr = (img_arr - mean) / std
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()

        lbl_arr = np.array(label, dtype=np.int64)
        lbl_tensor = torch.from_numpy(lbl_arr).long()

        return img_tensor, lbl_tensor


def train_model(model_name: str = "deeplabv3plus", epochs: int = 5, batch_size: int = 8, lr: float = 3e-4):
    print(f"\n==========================================")
    print(f"[*] Training {model_name.upper()} on device: {DEVICE}")
    print(f"==========================================")

    train_manifest = os.path.join(SPLITS_DIR, "train.json")
    val_manifest = os.path.join(SPLITS_DIR, "val.json")

    if not os.path.exists(train_manifest) or not os.path.exists(val_manifest):
        print(f"[-] Manifests not found at {SPLITS_DIR}. Please run scripts/download_ai4mars.py first.")
        return

    train_ds = AI4MarsDataset(train_manifest, img_size=256, is_train=True)
    val_ds = AI4MarsDataset(val_manifest, img_size=256, is_train=False)

    print(f"    Train size: {len(train_ds)} | Validation size: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    if model_name == "deeplabv3plus":
        model = DeepLabV3Plus(num_classes=NUM_CLASSES).to(DEVICE)
    elif model_name == "segformer_b0":
        model = SegFormerB0(num_classes=NUM_CLASSES).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Loss function ignoring unlabeled pixels (class 255)
    # Give slight weight to rare hazards (Sand=2, Big Rock=3)
    class_weights = torch.tensor([1.0, 1.2, 2.0, 2.5], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_miou = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_t = time.perf_counter()

        for step, (images, targets) in enumerate(train_loader):
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, targets)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                tgts = targets.cpu().numpy()
                cm += compute_confusion_matrix(preds, tgts, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

        val_loss /= len(val_loader)
        metrics = evaluate_segmentation_metrics(cm)
        val_miou = metrics["mean_iou"]
        sand_rec = metrics["hazard_recall"]["sand_recall"]
        rock_rec = metrics["hazard_recall"]["rock_recall"]
        elapsed = time.perf_counter() - start_t

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | Sand Recall: {sand_rec:.4f} | Rock Recall: {rock_rec:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_mean_iou": round(val_miou, 4),
            "sand_recall": round(sand_rec, 4),
            "rock_recall": round(rock_rec, 4)
        })

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            save_path = os.path.join(MODELS_DIR, f"aresatlas_{model_name}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"    [+] Saved new best model checkpoint to {save_path} (mIoU: {best_val_miou:.4f})")

    # Save training history
    with open(os.path.join(MODELS_DIR, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[+] Training complete for {model_name}!")
    return model

if __name__ == "__main__":
    train_model(model_name="deeplabv3plus", epochs=4, batch_size=8, lr=4e-4)
