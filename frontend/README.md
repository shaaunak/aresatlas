# AresAtlas — Reference-oriented interactive frontend starter

This is a dependency-free, runnable HTML/CSS/JavaScript frontend assembled with original clean illustrative Mars image assets. It includes a visually richer Uncertainty Lab and Research page, cinematic hero, a route simulation engine, overlay controls and a model-evaluation synthetic chart preview.

**This is a starting implementation, NOT an approved pixel-exact reproduction of the concept art and NOT a trained ML research platform.** For the complete visual build handoff, use the sibling `prompts/01_MASTER_EXECUTE.md` in the master ZIP.

## Run

```bash
python3 server.py
# open http://localhost:4173
```

The code requires no `npm install`. Port can be changed with `PORT=4181 python3 server.py`.

## Pages

`#home`, `#terrain`, `#route`, `#regret`, `#uncertainty`, `#models`, `#dataset`, `#lab`, `#research`.

The local route planner and selective risk demo operate on **deterministic synthetic data**. Model inference, true expert annotations and real benchmark data require a compatible FastAPI backend. Metrics in Model Evaluation's standalone chart preview are explicitly marked SYNTHETIC EXAMPLE, not a trained model run. The hero and landscapes are original AI-generated illustration, not authentic NASA photography or measured Mars geometry.

Default backend URL is `http://localhost:8000`, overridable by `window.ARESATLAS_API_URL` at load time or localStorage key `aresatlas_api`.

## Git safety

Do not commit delivery archives, benchmark PDFs, reference mockup screenshots, QA screenshots, datasets, checkpoints or local secrets. Keep only runtime source/assets when integrating. Do not push/deploy before visual approval and actual tests.
