import numpy as np
import torch
from typing import Dict, Any, Tuple

def compute_predictive_entropy(probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Computes Shannon entropy across class probabilities for each pixel,
    normalized by ln(C) so the range is strictly [0.0, 1.0].
    Shape: (H, W)
    """
    # probs: (C, H, W)
    num_classes = probs.shape[0]
    p = np.clip(probs, eps, 1.0)
    entropy = -np.sum(p * np.log(p), axis=0) / np.log(num_classes)
    return np.clip(entropy, 0.0, 1.0)


def compute_margin_uncertainty(probs: np.ndarray) -> np.ndarray:
    """
    Computes margin uncertainty: 1 - (p_top1 - p_top2).
    Range: [0.0, 1.0]. A value of 1.0 indicates top two classes are tied.
    Shape: (H, W)
    """
    # sort probabilities descending along channel axis
    sorted_p = np.sort(probs, axis=0)[::-1]
    top1 = sorted_p[0]
    top2 = sorted_p[1]
    margin = 1.0 - (top1 - top2)
    return np.clip(margin, 0.0, 1.0)


def compute_confidence_map(probs: np.ndarray) -> np.ndarray:
    """
    Returns max class probability for each pixel: max_c P(c).
    Shape: (H, W), Range: [0.25, 1.0] for 4 classes.
    """
    return np.max(probs, axis=0)


def calculate_expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10
) -> Dict[str, Any]:
    """
    Calculates empirical Expected Calibration Error (ECE) and reliability diagram bins.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_data = []

    total_samples = len(confidences)
    if total_samples == 0:
        return {"ece": 0.0, "bins": []}

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += abs_diff * prop_in_bin

            bin_data.append({
                "bin_index": i,
                "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "confidence": round(float(avg_confidence_in_bin), 4),
                "accuracy": round(float(accuracy_in_bin), 4),
                "sample_count": int(np.sum(in_bin))
            })
        else:
            bin_data.append({
                "bin_index": i,
                "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "confidence": round(float((bin_lower + bin_upper) / 2), 4),
                "accuracy": 0.0,
                "sample_count": 0
            })

    return {
        "ece": round(float(ece), 4),
        "bins": bin_data
    }
