import numpy as np
from typing import Dict, Any, List

def compute_confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int = 4, ignore_index: int = 255) -> np.ndarray:
    """
    Computes a num_classes x num_classes confusion matrix.
    pred, target: integer arrays of any flat or 2D shape.
    """
    p = pred.reshape(-1)
    t = target.reshape(-1)

    # Filter out ignore_index
    valid = (t != ignore_index) & (t >= 0) & (t < num_classes) & (p >= 0) & (p < num_classes)
    p_valid = p[valid]
    t_valid = t[valid]

    cm = np.bincount(
        num_classes * t_valid + p_valid,
        minlength=num_classes ** 2
    ).reshape((num_classes, num_classes))

    return cm


def evaluate_segmentation_metrics(
    cm: np.ndarray,
    class_names: Dict[int, str] = None
) -> Dict[str, Any]:
    """
    Computes per-class IoU, mean IoU, hazard recall, precision, and accuracy from confusion matrix.
    Rows: Ground Truth, Columns: Predictions.
    """
    if class_names is None:
        class_names = {0: "Soil", 1: "Bedrock", 2: "Sand", 3: "Big Rock"}

    num_classes = cm.shape[0]
    total_valid_pixels = int(np.sum(cm))

    tp = np.diag(cm).astype(float)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    gt_totals = np.sum(cm, axis=1)

    # IoU = TP / (TP + FP + FN)
    denominator = tp + fp + fn
    iou = np.zeros(num_classes)
    valid_classes = []

    for c in range(num_classes):
        if denominator[c] > 0:
            iou[c] = tp[c] / denominator[c]
            valid_classes.append(c)
        else:
            iou[c] = 0.0

    mean_iou = float(np.mean(iou[valid_classes])) if valid_classes else 0.0

    # Recall and Precision
    recall = np.zeros(num_classes)
    precision = np.zeros(num_classes)

    for c in range(num_classes):
        recall[c] = (tp[c] / gt_totals[c]) if gt_totals[c] > 0 else 0.0
        pred_sum = tp[c] + fp[c]
        precision[c] = (tp[c] / pred_sum) if pred_sum > 0 else 0.0

    # Overall Pixel Accuracy
    pixel_acc = float(np.sum(tp) / total_valid_pixels) if total_valid_pixels > 0 else 0.0

    # Hazard-specific metrics
    # Sand = class 2 (slip hazard), Big Rock = class 3 (collision hazard)
    sand_recall = float(recall[2]) if 2 < num_classes else 0.0
    rock_recall = float(recall[3]) if 3 < num_classes else 0.0
    sand_iou = float(iou[2]) if 2 < num_classes else 0.0
    rock_iou = float(iou[3]) if 3 < num_classes else 0.0

    per_class = {}
    for c in range(num_classes):
        per_class[class_names.get(c, f"Class {c}")] = {
            "class_id": c,
            "iou": round(float(iou[c]), 4),
            "recall": round(float(recall[c]), 4),
            "precision": round(float(precision[c]), 4),
            "true_positive": int(tp[c]),
            "ground_truth_pixels": int(gt_totals[c])
        }

    # Normalized confusion matrix (percentage per actual class row)
    norm_cm = np.zeros_like(cm, dtype=float)
    for r in range(num_classes):
        row_sum = np.sum(cm[r, :])
        if row_sum > 0:
            norm_cm[r, :] = cm[r, :] / row_sum

    return {
        "mean_iou": round(mean_iou, 4),
        "pixel_accuracy": round(pixel_acc, 4),
        "hazard_recall": {
            "sand_recall": round(sand_recall, 4),
            "rock_recall": round(rock_recall, 4),
            "combined_hazard_recall": round(float((sand_recall + rock_recall) / 2), 4)
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": np.round(norm_cm, 4).tolist(),
        "total_evaluated_pixels": total_valid_pixels
    }
