#!/usr/bin/env python3
"""
AresAtlas - Comprehensive Empirical Benchmark Suite (Version 2.0)
Evaluates:
  1. Conventional Semantic Segmentation: Pixel Acc, Mean IoU, Per-Class IoU
  2. Hazard-First Evaluation (Signature 1): Cost-Weighted Error (CWE), Sand/Rock recall,
     lethal sinkage rate (Sand->Soil), chassis strike rate (Rock->Bedrock/Soil).
  3. Uncertainty & Selective Prediction (Signature 2): ECE, Brier Score,
     Risk-Coverage Curves (100%, 95%, 90%, 80%, 70%, 50%), AURC vs Random Abstention.
  4. Latency & Memory Footprint on Apple Silicon MPS.
"""

import os
import json
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.aresatlas.config import DATA_DIR, SPLITS_DIR, REPORTS_DIR, BENCHMARK_FILE, NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES
from src.aresatlas.models.deeplabv3plus import DeepLabV3Plus, AllSoilBaseline
from src.aresatlas.evaluation.hazard import compute_hazard_metrics
from src.aresatlas.uncertainty.quantification import calculate_expected_calibration_error
from src.aresatlas.uncertainty.selective import evaluate_selective_prediction

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def evaluate_model_full(model, model_name: str, test_samples: list, img_size: int = 256):
    print(f"[*] Comprehensive Evaluation of '{model_name}' on {len(test_samples)} held-out expert test images...")
    images_dir = DATA_DIR / "images"
    labels_dir = DATA_DIR / "labels"

    is_baseline = isinstance(model, AllSoilBaseline)
    if not is_baseline:
        model.eval()

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    latencies = []

    confidences_list = []
    accuracies_list = []
    errors_list = []

    for item in test_samples:
        img_path = images_dir / item["image_file"]
        lbl_path = labels_dir / item["label_file"]
        if not img_path.exists() or not lbl_path.exists():
            continue

        raw_img = Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
        raw_lbl = Image.open(lbl_path).resize((img_size, img_size), Image.Resampling.NEAREST)

        img_arr = np.array(raw_img, dtype=np.float32) / 255.0
        if not is_baseline:
            img_arr = (img_arr - mean) / std
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        target = np.array(raw_lbl, dtype=np.int64)

        t0 = time.perf_counter()
        with torch.no_grad():
            if is_baseline:
                probs = model.predict(img_t).cpu().numpy()[0]
            else:
                logits = model(img_t)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        pred_mask = np.argmax(probs, axis=0)

        # Update confusion matrix
        t_flat = target.reshape(-1)
        p_flat = pred_mask.reshape(-1)
        valid = (t_flat != IGNORE_INDEX) & (t_flat >= 0) & (t_flat < NUM_CLASSES)
        if np.any(valid):
            t_v = t_flat[valid]
            p_v = p_flat[valid]
            sample_cm = np.bincount(NUM_CLASSES * t_v + p_v, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
            total_cm += sample_cm

            # Calibration & Selective Prediction Sampling
            confs = np.max(probs, axis=0).reshape(-1)[valid]
            accs = (p_v == t_v).astype(float)
            errs = 1.0 - accs

            if len(confs) > 100:
                sub = np.random.choice(len(confs), 100, replace=False)
                confidences_list.extend(confs[sub])
                accuracies_list.extend(accs[sub])
                errors_list.extend(errs[sub])
            else:
                confidences_list.extend(confs)
                accuracies_list.extend(accs)
                errors_list.extend(errs)

    # 1. Standard Metrics
    tp = np.diag(total_cm).astype(float)
    fp = np.sum(total_cm, axis=0) - tp
    fn = np.sum(total_cm, axis=1) - tp
    denom = tp + fp + fn
    iou = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        iou[c] = tp[c] / denom[c] if denom[c] > 0 else 0.0
    mean_iou = float(np.mean(iou))
    pixel_acc = float(np.sum(tp) / np.sum(total_cm)) if np.sum(total_cm) > 0 else 0.0

    # 2. Hazard-First Metrics
    hazard_data = compute_hazard_metrics(total_cm)

    # 3. Calibration Metrics
    conf_arr = np.array(confidences_list, dtype=float)
    acc_arr = np.array(accuracies_list, dtype=float)
    err_arr = np.array(errors_list, dtype=float)
    calibration = calculate_expected_calibration_error(conf_arr, acc_arr)
    brier_score = float(np.mean((conf_arr - acc_arr) ** 2)) if len(conf_arr) > 0 else 0.0

    # 4. Selective Prediction Risk-Coverage Curves
    selective_res = evaluate_selective_prediction(conf_arr, err_arr)

    # Per-class summary
    per_class = {}
    gt_totals = np.sum(total_cm, axis=1)
    for c in range(NUM_CLASSES):
        c_recall = float(tp[c] / gt_totals[c]) if gt_totals[c] > 0 else 0.0
        c_pred = np.sum(total_cm[:, c])
        c_prec = float(tp[c] / c_pred) if c_pred > 0 else 0.0
        per_class[CLASS_NAMES[c]] = {
            "iou": round(float(iou[c]), 4),
            "recall": round(c_recall, 4),
            "precision": round(c_prec, 4),
            "ground_truth_pixels": int(gt_totals[c])
        }

    return {
        "model_name": model_name,
        "mean_iou": round(mean_iou, 4),
        "pixel_accuracy": round(pixel_acc, 4),
        "per_class": per_class,
        "hazard_evaluation": hazard_data["hazard_evaluation"],
        "cost_weighted_error": hazard_data["cost_weighted_error"],
        "calibration": {
            "ece": calibration["ece"],
            "brier_score": round(brier_score, 4),
            "bins": calibration["bins"]
        },
        "selective_prediction": selective_res,
        "confusion_matrix": total_cm.tolist(),
        "mean_latency_ms": round(float(np.mean(latencies[5:]) if len(latencies) > 5 else np.mean(latencies)), 2),
        "total_evaluated_pixels": int(np.sum(total_cm))
    }

def main():
    print("=== AresAtlas Comprehensive Benchmark Evaluation ===")
    manifest_file = SPLITS_DIR / "test_expert.json"
    with open(manifest_file, "r") as f:
        test_samples = json.load(f)

    print(f"[+] Loaded {len(test_samples)} held-out gold expert test images.")

    # 1. Baseline
    baseline = AllSoilBaseline(num_classes=NUM_CLASSES)
    base_res = evaluate_model_full(baseline, "All-Soil Baseline", test_samples)

    # 2. DeepLabV3+
    deeplab = DeepLabV3Plus(num_classes=NUM_CLASSES).to(DEVICE)
    ckpt_path = DATA_DIR.parent.parent / "backend" / "models" / "aresatlas_deeplabv3plus.pt"
    if ckpt_path.exists():
        deeplab.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"[+] Loaded trained DeepLabV3+ checkpoint from {ckpt_path}")
    deeplab_res = evaluate_model_full(deeplab, "DeepLabV3+ (ASPP + Decoder)", test_samples)

    full_report = {
        "evaluation_name": "AresAtlas V2.0 Empirical Evaluation",
        "dataset": "NASA JPL AI4Mars (MSL Curiosity)",
        "test_partition": "test_min1 (Expert consensus gold standard)",
        "evaluated_samples": len(test_samples),
        "device": str(DEVICE),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {
            "all_soil_baseline": base_res,
            "deeplabv3plus": deeplab_res
        },
        "comparative_insights": {
            "mIoU_gain": round(deeplab_res["mean_iou"] - base_res["mean_iou"], 4),
            "cost_weighted_error_reduction": round(base_res["cost_weighted_error"] - deeplab_res["cost_weighted_error"], 4),
            "sand_recall_gain": round(deeplab_res["hazard_evaluation"]["sand"]["recall"] - base_res["hazard_evaluation"]["sand"]["recall"], 4),
            "rock_recall_gain": round(deeplab_res["hazard_evaluation"]["big_rock"]["recall"] - base_res["hazard_evaluation"]["big_rock"]["recall"], 4),
            "selective_prediction_aurc_gain": round(deeplab_res["selective_prediction"]["aurc_gain"], 4)
        }
    }

    report_path = REPORTS_DIR / "empirical_benchmarks.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)

    # Also update backend benchmark file
    with open(BENCHMARK_FILE, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n[+] Full benchmark report saved to {report_path} and {BENCHMARK_FILE}!")
    print(f"    Baseline CWE: {base_res['cost_weighted_error']} | DeepLabV3+ CWE: {deeplab_res['cost_weighted_error']}")
    print(f"    Baseline Sand Recall: {base_res['hazard_evaluation']['sand']['recall']} | DeepLabV3+ Sand Recall: {deeplab_res['hazard_evaluation']['sand']['recall']}")
    print(f"    Selective Prediction AURC Gain: {deeplab_res['selective_prediction']['aurc_gain']}")

if __name__ == "__main__":
    main()
