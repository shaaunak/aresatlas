import os
import json
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ..config import BENCHMARK_FILE

router = APIRouter(prefix="/api/benchmark", tags=["benchmarking"])

@router.get("", response_model=Dict[str, Any])
async def get_benchmark_report():
    """
    Returns empirical benchmark metrics comparing All-Soil baseline
    against trained deep learning segmentation models on NASA JPL's
    held-out expert-labeled gold test partition (test_min1).
    """
    if os.path.exists(BENCHMARK_FILE):
        with open(BENCHMARK_FILE, "r") as f:
            data = json.load(f)
            return data

    raise HTTPException(
        status_code=503,
        detail="Benchmark evaluation report not yet computed. Run 'python3 scripts/evaluate_benchmarks.py' to generate verified empirical results from the test partitions."
    )
