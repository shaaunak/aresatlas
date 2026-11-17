# AI4Mars Dataset Card (Curiosity MSL Subset)

## Dataset Summary
The **AI4Mars** dataset was created by NASA Jet Propulsion Laboratory (JPL) to advance autonomous navigation for planetary rovers. It consists of grayscale and color imagery captured by the Navigation Cameras (Navcam) and Mast Cameras (Mastcam) aboard the Mars Science Laboratory (MSL) Curiosity rover operating inside Gale Crater on Mars.

- **Creators:** NASA Jet Propulsion Laboratory (Swan, Ono, et al., 2021)
- **Hugging Face Mirror:** `hassanjbara/AI4MARS`
- **License:** CC0 1.0 Universal (Public Domain)
- **Data Volume:** 322 real NASA images partitioned into:
  - `test_min1` (262 held-out images with 100% agreement expert annotations)
  - `val` (20 images)
  - `train` (40 images)

## Class Taxonomy
- `0`: **Soil** — fine-grained regolith, smooth traversal, high wheel friction coefficient.
- `1`: **Bedrock** — consolidated rock outcrop, flat surface, low-to-moderate roughness.
- `2`: **Sand** — aeolian sand dunes and wind-rippled drifts, critical wheel-slip hazard.
- `3`: **Big Rock** — boulders > 15–20 cm in diameter, severe chassis collision and high-centering hazard.
- `255`: **Unlabeled / Sky / Rover Body** — ignored during loss calculation and evaluation.

## Partition Integrity
- In AresAtlas v2.0, the expert consensus gold standard (`test_min1`) was strictly quarantined prior to model hyperparameter tuning.
- Missing labels account for 42.46% of raw pixels; our preprocessing enforces that unlabeled pixels are strictly masked out rather than imputed as safe soil.
