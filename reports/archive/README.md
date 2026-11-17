# Archived Benchmark & Study Reports

This directory preserves historical evaluation runs from AresAtlas Version 1.0/2.0 initial staging. Per research integrity protocols, inaccurate or flawed results are never deleted; they are archived with transparent post-mortem analysis.

---

## 1. `empirical_benchmarks_v1_leaked_audit.json`
- **Archived Date:** November 17, 2026
- **Reason:**
  1. **Dataset Split Leakage:** The evaluation used in this run relied on `scripts/download_ai4mars.py` where 40 training and 20 validation images were extracted from the expert consensus test split (`test_min1`).
  2. **Model Collapse Discrepancy:** The `deeplabv3plus` model in this report achieved only **0.0733 mIoU (7.33%)** and **0.2933 pixel accuracy** due to underfitting (trained for only 4 epochs on 40 samples from scratch, collapsing to predict all class 1 Bedrock). In contrast, the initial `README.md` claimed an unverified 43.2% mIoU.
  3. **SegFormer Missing:** `SegFormer-B0` was listed in README tables (41.8% mIoU) but was not evaluated or present in this empirical benchmark JSON.

## 2. `route_regret_study_v1.json`
- **Archived Date:** November 17, 2026
- **Reason:**
  1. In the API endpoint implementation associated with this phase (`src/aresatlas/api/research.py`), predicted maps in route regret were accidentally initialized as copies of ground truth (`pred_map = np.copy(gt_map)`), causing circular self-comparison.
  2. Uncertainty in that endpoint was approximated from image gradients of ground truth masks rather than true predictive entropy from model logits.
