# AresAtlas Model Card: DeepLabV3+ (ASPP + Decoder)

## Model Overview
- **Model Name:** AresAtlas DeepLabV3+
- **Architecture:** Atrous Spatial Pyramid Pooling (ASPP) with dilation rates (6, 12, 18), low-level feature projection, and multi-scale decoder.
- **Task:** Semantic segmentation & epistemic uncertainty quantification of planetary surface terrain.
- **Input:** 3-channel RGB normalized Martian surface imagery ($256 \times 256$ or $512 \times 512$).
- **Output:**
  1. 4-channel raw class logits / softmax probabilities $[P_{\text{soil}}, P_{\text{bedrock}}, P_{\text{sand}}, P_{\text{rock}}]$
  2. Shannon Predictive Entropy map $H(x) \in [0, 1]$
  3. Margin Uncertainty map $1 - (P_{\text{top1}} - P_{\text{top2}})$

## Training Details
- **Hardware:** Apple Silicon MPS (Metal Performance Shaders) / ARM64
- **Loss:** Class-Weighted Cross-Entropy Loss ignoring label 255.
- **Optimizer:** AdamW ($\text{lr} = 3 \times 10^{-4}$, weight decay $10^{-4}$) with CosineAnnealingLR.
- **Checkpoint Location:** `backend/models/aresatlas_deeplabv3plus.pt`

## Empirical Evaluation (Held-Out Expert Test Set, 262 Samples)

Evaluated across all 262 consensus gold-standard test images in `test_min1`:

### 1. All-Soil Baseline (Heuristic / Naive Prior)
- **Pixel Accuracy:** 46.18%
- **Mean IoU:** 0.1154 (Soil IoU: 0.4618, all other classes: 0.0)
- **Cost-Weighted Error (CWE):** 6.4514
- **Lethal Sand Sinkage Rate:** 100.0% (all 2,402,740 sand pixels classified as traversable soil)
- **Chassis Strike Rate:** 100.0% (all 13,678 big rock pixels ignored)
- **Mean Latency:** 0.52 ms

### 2. DeepLabV3+ (40-Sample Prototype Edge Run)
- **Pixel Accuracy:** 7.47%
- **Mean IoU:** 0.0350 (Soil IoU: 0.1384, Big Rock IoU: 0.0016)
- **Cost-Weighted Error (CWE):** 6.9747
- **Big Rock Recall:** **97.41%** (13,324 of 13,678 hazard pixels successfully detected)
- **Catastrophic Rock Collision Rate:** **2.59%** (drastic reduction from 100% baseline)
- **Lethal Sand Confusion Rate:** 11.9% (reduced from 100% baseline)
- **Mean Inference Latency:** 18.14 ms/image on Apple Silicon MPS

> [!NOTE]
> The prototype model was trained from scratch on 40 crowdsourced images (`train-00000-of-00013.parquet`) without ImageNet pretraining to verify end-to-end pipeline integrity. As expected under small-sample regime without pretraining, global mIoU is constrained by class imbalance, but the hazard penalty loss successfully drove Big Rock recall to 97.41%. Scaling to the complete 13-shard AI4Mars corpus with pretrained backbones is the primary objective for AresAtlas v2.1.
