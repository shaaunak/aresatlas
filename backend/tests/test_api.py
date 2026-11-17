import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "operational"
    assert "AresAtlas" in data["platform"]

def test_benchmark_endpoint():
    res = client.get("/api/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert "all_soil_baseline" in data["models"]

def test_route_endpoint():
    payload = {
        "start": [10, 10],
        "goal": [50, 50],
        "soil_weight": 1.0,
        "sand_weight": 6.0,
        "rock_weight": 999.0,
        "uncertainty_penalty": 2.5,
        "safety_buffer_radius": 2,
        "algorithms": ["astar", "shortest"]
    }
    res = client.post("/api/route", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "routes" in data
    assert "astar" in data["routes"]
    assert "shortest" in data["routes"]
    assert "comparison_summary" in data
    assert "scientific_disclaimer" in data
