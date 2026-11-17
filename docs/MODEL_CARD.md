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
- **Pixel Accuracy:** 29.33%
- **Mean IoU:** 0.0733
- **Cost-Weighted Error (CWE):** **5.3883** (compared to **6.4514** for All-Soil baseline, representing a 1.0631 point risk reduction)
- **Mean Inference Latency:** 28.4 ms/image on Apple Silicon MPS
