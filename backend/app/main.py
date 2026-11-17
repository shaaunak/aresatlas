from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import torch

from .config import DATA_DIR, ROOT_DIR
from .api import samples, predict, route, benchmark
from src.aresatlas.api.research import router as research_router

app = FastAPI(
    title="AresAtlas: Mars Terrain Intelligence & Decision-Aware Navigation Research",
    description="Research-grade platform evaluating the downstream cost of semantic terrain perception errors and uncertainty in simulated Martian route planning.",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(samples.router)
app.include_router(predict.router)
app.include_router(route.router)
app.include_router(benchmark.router)
app.include_router(research_router)

# Aliases for Section 10.1 Master Specification & Standalone Frontend
app.add_api_route("/predict", predict.predict_terrain, methods=["POST"])

@app.get("/health")
@app.get("/api/health")
async def health_check():
    return {
        "status": "operational",
        "platform": "AresAtlas Research Engine v2.0",
        "pytorch_version": torch.__version__,
        "acceleration": "Apple Silicon MPS" if torch.backends.mps.is_available() else "CPU",
        "dataset": "NASA JPL AI4Mars (MSL Curiosity)",
        "classes": ["Soil", "Bedrock", "Sand", "Big Rock", "Unlabeled"]
    }

# Mount static asset streams
images_dir = os.path.join(DATA_DIR, "images")
if os.path.exists(images_dir):
    app.mount("/static/images", StaticFiles(directory=images_dir), name="static_images")

labels_dir = os.path.join(DATA_DIR, "labels")
if os.path.exists(labels_dir):
    app.mount("/static/labels", StaticFiles(directory=labels_dir), name="static_labels")

# Mount frontend web application
frontend_dir = os.path.join(ROOT_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
