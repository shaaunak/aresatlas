import numpy as np
from typing import Dict, Any, List, Tuple

def evaluate_selective_prediction(
    confidences: np.ndarray,
    errors: np.ndarray,
    coverage_thresholds: List[float] = [1.0, 0.95, 0.90, 0.80, 0.70, 0.50]
) -> Dict[str, Any]:
    """
    Evaluates Selective Prediction / Risk-Coverage trade-off:
    - Retains predictions with highest confidence (lowest uncertainty).
    - Measures error (risk) on retained predictions at specified coverage levels.
    - Computes AURC (Area Under Risk-Coverage Curve).
    - Compares against Random Abstention baseline.
    """
    N = len(confidences)
    if N == 0:
        return {"aurc": 0.0, "points": []}

    # Sort descending by confidence (most confident first)
    sort_order = np.argsort(confidences)[::-1]
    sorted_errors = errors[sort_order]

    # Cumulative risk (error rate) at each point
    cumulative_errors = np.cumsum(sorted_errors)
    n_retained = np.arange(1, N + 1)
    selective_risks = cumulative_errors / n_retained
    coverages = n_retained / float(N)

    # Sample at designated coverage milestones
    milestones = []
    for cov in coverage_thresholds:
        idx = min(int(np.ceil(cov * N)) - 1, N - 1)
        actual_cov = float(coverages[idx])
        risk = float(selective_risks[idx])
        # Random abstention expected risk at same coverage is overall baseline risk
        random_risk = float(np.mean(errors))

        milestones.append({
            "target_coverage": round(cov, 2),
            "actual_coverage": round(actual_cov, 4),
            "selective_risk": round(risk, 4),
            "random_abstention_risk": round(random_risk, 4),
            "risk_reduction_pct": round(((random_risk - risk) / (random_risk + 1e-7)) * 100.0, 2),
            "samples_retained": int(n_retained[idx])
        })

    # AURC (Area Under Risk-Coverage Curve) via trapezoidal integration
    # Evaluated from coverage=0.0 to 1.0
    if hasattr(np, 'trapezoid'):
        aurc = float(np.trapezoid(selective_risks, coverages))
    else:
        from scipy.integrate import trapezoid
        aurc = float(trapezoid(selective_risks, coverages))
    random_aurc = float(np.mean(errors)) * 1.0  # Constant horizontal line

    return {
        "aurc": round(aurc, 4),
        "random_baseline_aurc": round(random_aurc, 4),
        "aurc_gain": round(float(random_aurc - aurc), 4),
        "coverage_milestones": milestones,
        "total_evaluated_samples": N
    }
