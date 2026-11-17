import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List

from ..config import CONFIGS_DIR, CLASS_NAMES

def load_error_cost_matrix() -> np.ndarray:
    cost_file = CONFIGS_DIR / "hazard_cost_matrix.json"
    if cost_file.exists():
        with open(cost_file, "r") as f:
            data = json.load(f)
            return np.array(data["cost_matrix"], dtype=float)
    # Default fallback
    return np.array([
        [0,  1,  5,  8],
        [1,  0,  4,  8],
        [25, 20, 0,  5],
        [50, 40, 15, 0]
    ], dtype=float)


def compute_cost_weighted_error(cm: np.ndarray, cost_matrix: np.ndarray = None) -> Tuple[float, np.ndarray]:
    """
    Computes Cost-Weighted Error (CWE):
    CWE = sum(N_{i, j} * C_{i, j}) / sum(N_{i, j})
    where i = Ground Truth Class, j = Predicted Class.
    """
    if cost_matrix is None:
        cost_matrix = load_error_cost_matrix()

    total_valid = np.sum(cm)
    if total_valid == 0:
        return 0.0, np.zeros_like(cm, dtype=float)

    weighted_errors = cm * cost_matrix
    total_cost = float(np.sum(weighted_errors))
    cwe = total_cost / float(total_valid)

    return round(cwe, 4), weighted_errors


def compute_hazard_metrics(cm: np.ndarray) -> Dict[str, Any]:
    """
    Computes hazard-first evaluation metrics on real confusion matrix:
    - Sand recall/precision/F1
    - Big Rock recall/precision/F1
    - Lethal confusion rates: Sand->Soil (wheel sinkage), BigRock->Bedrock/Soil (chassis strike)
    - Cost-Weighted Error
    """
    cost_matrix = load_error_cost_matrix()
    cwe, weighted_breakdown = compute_cost_weighted_error(cm, cost_matrix)

    # Class indices: 0: Soil, 1: Bedrock, 2: Sand, 3: Big Rock
    tp = np.diag(cm).astype(float)
    gt_totals = np.sum(cm, axis=1).astype(float)
    pred_totals = np.sum(cm, axis=0).astype(float)

    # Sand (index 2)
    sand_gt = gt_totals[2]
    sand_pred = pred_totals[2]
    sand_tp = tp[2]
    sand_recall = (sand_tp / sand_gt) if sand_gt > 0 else 0.0
    sand_precision = (sand_tp / sand_pred) if sand_pred > 0 else 0.0
    sand_f1 = (2 * sand_precision * sand_recall / (sand_precision + sand_recall)) if (sand_precision + sand_recall) > 0 else 0.0

    # Sand -> Soil lethal sinkage confusion rate
    sand_misclassified_as_soil = cm[2, 0]
    sand_to_soil_rate = (sand_misclassified_as_soil / sand_gt) if sand_gt > 0 else 0.0

    # Big Rock (index 3)
    rock_gt = gt_totals[3]
    rock_pred = pred_totals[3]
    rock_tp = tp[3]
    rock_recall = (rock_tp / rock_gt) if rock_gt > 0 else 0.0
    rock_precision = (rock_tp / rock_pred) if rock_pred > 0 else 0.0
    rock_f1 = (2 * rock_precision * rock_recall / (rock_precision + rock_recall)) if (rock_precision + rock_recall) > 0 else 0.0

    # Rock -> Soil/Bedrock collision confusion rate
    rock_misclassified_as_soil = cm[3, 0]
    rock_misclassified_as_bedrock = cm[3, 1]
    rock_strike_rate = ((rock_misclassified_as_soil + rock_misclassified_as_bedrock) / rock_gt) if rock_gt > 0 else 0.0

    # Combined Hazard Metrics (Classes 2 and 3 vs Non-Hazards 0 and 1)
    hazard_tp = tp[2] + tp[3]
    hazard_gt = sand_gt + rock_gt
    hazard_pred = sand_pred + rock_pred
    hazard_recall = (hazard_tp / hazard_gt) if hazard_gt > 0 else 0.0
    hazard_precision = (hazard_tp / hazard_pred) if hazard_pred > 0 else 0.0

    return {
        "cost_weighted_error": cwe,
        "hazard_evaluation": {
            "combined_hazard_recall": round(float(hazard_recall), 4),
            "combined_hazard_precision": round(float(hazard_precision), 4),
            "sand": {
                "recall": round(float(sand_recall), 4),
                "precision": round(float(sand_precision), 4),
                "f1": round(float(sand_f1), 4),
                "ground_truth_pixels": int(sand_gt),
                "lethal_sand_as_soil_confusion_rate": round(float(sand_to_soil_rate), 4),
                "lethal_sinkage_pixels": int(sand_misclassified_as_soil)
            },
            "big_rock": {
                "recall": round(float(rock_recall), 4),
                "precision": round(float(rock_precision), 4),
                "f1": round(float(rock_f1), 4),
                "ground_truth_pixels": int(rock_gt),
                "catastrophic_collision_confusion_rate": round(float(rock_strike_rate), 4),
                "chassis_strike_pixels": int(rock_misclassified_as_soil + rock_misclassified_as_bedrock)
            }
        },
        "cost_weighted_error_breakdown": weighted_breakdown.tolist(),
        "total_evaluated_pixels": int(np.sum(cm))
    }
