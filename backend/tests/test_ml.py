import pytest
import numpy as np
import torch
from backend.app.ml.models import DeepLabV3Plus, SegFormerB0, AllSoilBaseline
from backend.app.ml.uncertainty import compute_predictive_entropy, compute_margin_uncertainty, compute_confidence_map
from backend.app.ml.metrics import compute_confusion_matrix, evaluate_segmentation_metrics

def test_deeplabv3plus_shape():
    model = DeepLabV3Plus(num_classes=4)
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (1, 4, 128, 128)

def test_segformer_b0_shape():
    model = SegFormerB0(num_classes=4)
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (1, 4, 128, 128)

def test_all_soil_baseline():
    baseline = AllSoilBaseline(num_classes=4)
    dummy_input = torch.randn(2, 3, 64, 64)
    probs = baseline.predict(dummy_input)
    assert probs.shape == (2, 4, 64, 64)
    assert torch.all(probs[:, 0, :, :] == 1.0)
    assert torch.all(probs[:, 1:, :, :] == 0.0)

def test_uncertainty_quantification():
    # Uniform probabilities -> maximal entropy
    uniform_probs = np.ones((4, 10, 10), dtype=np.float32) * 0.25
    entropy = compute_predictive_entropy(uniform_probs)
    margin = compute_margin_uncertainty(uniform_probs)
    conf = compute_confidence_map(uniform_probs)

    assert np.allclose(entropy, 1.0, atol=1e-3)
    assert np.allclose(margin, 1.0, atol=1e-3)
    assert np.allclose(conf, 0.25, atol=1e-3)

    # Deterministic certainty
    certain_probs = np.zeros((4, 10, 10), dtype=np.float32)
    certain_probs[0, :, :] = 1.0
    entropy_certain = compute_predictive_entropy(certain_probs)
    margin_certain = compute_margin_uncertainty(certain_probs)
    conf_certain = compute_confidence_map(certain_probs)

    assert np.allclose(entropy_certain, 0.0, atol=1e-3)
    assert np.allclose(margin_certain, 0.0, atol=1e-3)
    assert np.allclose(conf_certain, 1.0, atol=1e-3)

def test_metrics_calculation():
    # Target and prediction
    target = np.array([0, 0, 1, 2, 3, 255])
    pred = np.array([0, 0, 1, 2, 0, 1])

    cm = compute_confusion_matrix(pred, target, num_classes=4, ignore_index=255)
    assert cm.shape == (4, 4)
    assert np.sum(cm) == 5 # 255 was ignored

    metrics = evaluate_segmentation_metrics(cm)
    assert metrics["mean_iou"] > 0.0
    assert metrics["hazard_recall"]["sand_recall"] == 1.0
    assert metrics["hazard_recall"]["rock_recall"] == 0.0
