"""
Terrasense-X — Offline satellite image semantic search & change detection.
Run: python main.py
"""
import os, uvicorn
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

# Serve frontend
app.mount("/static", StaticFiles(directory=cfg.FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(cfg.FRONTEND_DIR, "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
