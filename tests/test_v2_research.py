import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.aresatlas.evaluation.hazard import compute_cost_weighted_error, compute_hazard_metrics
from src.aresatlas.uncertainty.selective import evaluate_selective_prediction
from src.aresatlas.research.route_regret import (
    build_cost_grid, run_astar, compute_route_regret_comparison
)
from backend.app.main import app

client = TestClient(app)

def test_cost_weighted_error():
    # 4x4 toy confusion matrix
    cm = np.array([
        [100, 0,   0,   0],
        [0,   100, 0,   0],
        [10,  0,   90,  0],  # 10 sand pixels misclassified as soil
        [5,   0,   0,   95]  # 5 rock pixels misclassified as soil
    ], dtype=np.int64)

    # Cost matrix: Sand->Soil is 25, Rock->Soil is 50
    cwe, breakdown = compute_cost_weighted_error(cm)
    # Total cost = 10 * 25 + 5 * 50 = 250 + 250 = 500
    # Total pixels = 400
    # CWE = 500 / 400 = 1.25
    assert np.isclose(cwe, 1.25, atol=1e-2)

def test_hazard_metrics_lethal_confusion():
    cm = np.array([
        [100, 0, 0, 0],
        [0, 100, 0, 0],
        [20, 0, 80, 0],  # 20/100 sand misclassified as soil
        [10, 0, 0, 90]   # 10/100 rocks misclassified as soil
    ], dtype=np.int64)

    metrics = compute_hazard_metrics(cm)
    assert metrics["hazard_evaluation"]["sand"]["recall"] == 0.8
    assert metrics["hazard_evaluation"]["sand"]["lethal_sand_as_soil_confusion_rate"] == 0.2
    assert metrics["hazard_evaluation"]["big_rock"]["recall"] == 0.9
    assert metrics["hazard_evaluation"]["big_rock"]["catastrophic_collision_confusion_rate"] == 0.1

def test_selective_prediction_monotonic_risk():
    # 100 samples: confidences from 0.0 to 1.0, error is present only when conf < 0.5
    confs = np.linspace(0.01, 1.0, 100)
    errors = (confs < 0.5).astype(float)

    res = evaluate_selective_prediction(confs, errors)
    assert "aurc" in res
    assert res["aurc"] < res["random_baseline_aurc"]
    assert res["aurc_gain"] > 0 # Abstaining on low confidence reduces risk!

def test_route_regret_nonnegative_and_uncertainty_gain():
    # 20x20 toy grid
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[8:12, 5:15] = 2 # Sand trap in center

    # Pred: model thinks it is soil
    pred = np.zeros((20, 20), dtype=np.uint8)

    # Uncertainty: model has high uncertainty on the sand trap
    unc = np.zeros((20, 20), dtype=np.float32)
    unc[8:12, 5:15] = 1.0

    weights = {0: 1.0, 1: 1.3, 2: 6.0, 3: 999.0, 255: 3.5}
    start = (10, 2)
    goal = (10, 18)

    res = compute_route_regret_comparison(gt, pred, unc, start, goal, weights, uncertainty_lambda=4.0)
    assert res["status"] == "success"
    # Predicted regret must be non-negative
    assert res["predicted"]["regret"] >= 0.0
    # Uncertainty route should reduce true regret by navigating around high uncertainty zone
    assert res["uncertainty_aware"]["regret"] <= res["predicted"]["regret"]

def test_v2_api_endpoints():
    res_health = client.get("/health")
    assert res_health.status_code == 200

    res_model = client.get("/model/info")
    assert res_model.status_code == 200
    assert "DeepLabV3+" in res_model.json()["architecture"]

    res_dataset = client.get("/dataset/info")
    assert res_dataset.status_code == 200

    res_exp = client.get("/experiments")
    assert res_exp.status_code == 200

    res_findings = client.get("/research/findings")
    assert res_findings.status_code == 200
