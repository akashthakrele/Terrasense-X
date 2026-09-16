"""
Terrasense-X — Offline satellite image semantic search & change detection.
Run: python main.py
"""
import os
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.api import router
import config as cfg

# Ensure dirs exist
for d in [cfg.DATA_DIR, cfg.BEFORE_DIR, cfg.AFTER_DIR, cfg.MODELS_DIR,
          cfg.INDEX_DIR, cfg.OUTPUTS_DIR, cfg.FRONTEND_DIR]:
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="Terrasense-X")
app.include_router(router)

# Serve static assets
app.mount("/static/demo", StaticFiles(directory=os.path.join(cfg.DATA_DIR, "semantic_demo")), name="demo")
app.mount("/static/ssl4eo", StaticFiles(directory=cfg.SSL4EO_DIR), name="ssl4eo")
app.mount("/static", StaticFiles(directory=cfg.FRONTEND_DIR), name="static")
app.mount("/data", StaticFiles(directory=cfg.DATA_DIR), name="data")
app.mount("/outputs", StaticFiles(directory=cfg.OUTPUTS_DIR), name="outputs")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(cfg.FRONTEND_DIR, "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
