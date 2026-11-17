# AresAtlas Research & Codebase Audit Report
**Date:** November 17, 2026  
**Auditor:** AresAtlas Independent Research Integrity Audit  
**Status:** Audit Complete — Remediation in Progress

---

## 1. Executive Summary & Critical Findings

| ID | Category | Severity | File & Location | Summary of Finding | Proposed / Applied Fix |
|---|---|---|---|---|---|
| **AUD-01** | **Dataset Leakage** | **BLOCKER** | `scripts/download_ai4mars.py:33-35`, `143-164` | `test_min1` (322 expert consensus images) was split into train (40), val (20), and test (262), causing training on expert test partitions. Furthermore, `test_min2` is a subset of `test_min1`. | Separate crowdsourced training data (`train-*.parquet`) from expert test sets (`test_min1`, `test_min2`, `test_min3`). Verify zero ID and sol overlap. |
| **AUD-02** | **Fabricated Metrics in README** | **BLOCKER** | `README.md:92-93`, `backend/app/api/benchmark.py:48-64` | README reported DeepLabV3+ 43.2% mIoU, 78.1% sand recall, 71.4% rock recall, and SegFormer-B0 41.8% mIoU. The actual saved evaluation in `reports/empirical_benchmarks.json` was 7.33% mIoU (model collapsed to class 1 Bedrock after 4 epochs on 40 samples). Fallback API also had hardcoded 68.4% mIoU. | Remove all unbacked/fabricated numbers from README and fallback API. Report exact empirical metrics with full provenance. |
| **AUD-03** | **Untrained Model Fallback** | **MAJOR** | `backend/app/api/predict.py:40-44`, `48-56` | When model checkpoints (`.pt`) were missing, inference endpoints silently initialized random uninitialized architectures and returned random masks. | Return an HTTP 503 Service Unavailable with a clear error specifying missing weights. |
| **AUD-04** | **Ground Truth Leakage in Route Regret** | **BLOCKER** | `src/aresatlas/api/research.py:161-163` | In `/research/route-regret`, prediction map was assigned via `pred_map = np.copy(gt_map)` and `entropy_map = np.zeros_like(gt_map)`. The endpoint evaluated ground truth against itself instead of model inference. | Wire genuine model inference through the feature extractor to generate `pred_map` and predictive entropy tensor $H(x, y)$. |
| **AUD-05** | **Fake Boundary Entropy** | **MAJOR** | `src/aresatlas/api/research.py:189-191`, `backend/app/api/route.py:72-74` | Uncertainty was computed by taking spatial gradients (`np.gradient`) of ground-truth label masks rather than predictive entropy from model logits. | Calculate true predictive entropy $H = -\sum_{c} p_c \log_4(p_c + \epsilon)$ from model softmax probabilities. |
| **AUD-06** | **Inconsistent Cost Semantics** | **MAJOR** | `README.md:46`, `frontend/src.js:8-11`, `configs/planning_config.json`, `configs/hazard_cost_matrix.json` | README stated sand $\times 5$ and rock $\times 10$; frontend schematic hardcoded sand 3.2 and rock 6.5; planner config had sand 6.0 and rock 999.0; hazard evaluation matrix had asymmetric costs up to 50. | Establish single documented source of truth: separate "Planner Traversal Costs" from "Evaluation Hazard Misclassification Cost Matrix" and synchronize frontend and docs. |
| **AUD-07** | **Inaccurate Rover Missions Attribution** | **MINOR** | `README.md:87`, `docs/DATASET_CARD.md`, `scripts/download_ai4mars.py` | Documentation claimed AI4Mars contains Perseverance / Mars 2020 data. AI4Mars is MSL Curiosity and MER Spirit/Opportunity; the Hugging Face subset used here is MSL Curiosity exclusively. | Correct documentation, dataset cards, citations, and roadmap to MSL Curiosity. |
| **AUD-08** | **Model Parameter Discrepancy** | **MINOR** | `README.md:92` | README claimed DeepLabV3+ had 40.5M parameters (ResNet-50 torchvision default). The implemented architecture in `backend/app/ml/models.py` has 2,477,956 parameters. | Update architecture documentation to report true parameter count (2.48M parameters). |

---

## 2. In-Depth Subsystem Audits

### 2.1 Dataset Ingestion & Partitioning
- **Hugging Face Repository:** `hassanjbara/AI4MARS`
- **Existing Parquet Files:**
  - `data/train-00000-of-00013.parquet` through `train-00012-of-00013.parquet`: Crowdsourced training images from MSL Curiosity.
  - `data/test_min1-00000-of-00001.parquet`: 322 test images with at least 1 expert consensus.
  - `data/test_min2-00000-of-00001.parquet`: Test images with at least 2 expert consensus.
  - `data/test_min3-00000-of-00001.parquet`: Test images with at least 3 expert consensus.
- **Audit Result:** `scripts/download_ai4mars.py` previously downloaded `test_min1` and partitioned it into 40 train, 20 val, and 262 test. This was a direct leakage of the expert evaluation split.
- **Remediation:** Training and validation must be derived solely from `train-*.parquet` files. The 322 expert images must remain purely held-out for benchmarking.

### 2.2 Cost Modeling & Route Regret Formula
- **Planner Traversal Costs:**
  - `Soil (Class 0)`: 1.0 (baseline traversable surface)
  - `Bedrock (Class 1)`: 1.3 (firm substrate, minor roughness)
  - `Sand (Class 2)`: 6.0 (severe wheel slip and sinkage risk)
  - `Big Rock (Class 3)`: 999.0 (impassable physical barrier / collision obstacle)
  - `Unlabeled / Occluded (Class 255)`: 3.5 (caution traversal cost)
- **Evaluation Hazard Cost Matrix ($C_{y, \hat{y}}$):**
  - Sand mistaken for Soil: Cost = 25 (lethal immobilization)
  - Big Rock mistaken for Soil: Cost = 50 (catastrophic chassis strike)
  - Big Rock mistaken for Bedrock: Cost = 40 (undercarriage damage)
  - Soil mistaken for Sand: Cost = 5 (suboptimal conservative detour)
  - Diagonal entries: Cost = 0
- **Downstream Route Regret Formula:**
  $$\text{Regret}(R_{\text{pred}}, R^*) = \frac{C_{\text{true}}(R_{\text{pred}}) - C_{\text{true}}(R^*)}{C_{\text{true}}(R^*)}$$
  Where $C_{\text{true}}(R)$ evaluates the accumulated path cost of route $R$ on the ground-truth terrain cost grid. If the predicted route is identical to the oracle route, $\text{Regret} = 0.0$. Regret is mathematically guaranteed non-negative.

### 2.3 Uncertainty Quantification & Calibration
- **Predictive Entropy:**
  $$H(x, y) = -\sum_{c=0}^3 p_c(x, y) \log_4(p_c(x, y) + \epsilon)$$
  Normalized to $[0, 1]$ where $\log_4$ scales maximum entropy to 1.0 across 4 classes.
- **Expected Calibration Error (ECE):** Evaluated with 10 uniform confidence bins $[0.0, 0.1], \dots, [0.9, 1.0]$.
- **Risk-Coverage Curve:** Evaluated by sorting pixel confidences and computing cumulative selective risk across coverage milestones (100%, 95%, 90%, 80%, 70%, 50%).
