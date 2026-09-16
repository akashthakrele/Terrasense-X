import os
import faiss
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import config as cfg
from backend.ml.search import text_search

router = APIRouter()

@router.get("/search")
def search(q: str = Query(..., description="Text query"), k: int = Query(10, le=50)):
    raw_results = text_search(q, k)
    pairs = {}
    for res in raw_results:
        fname = res["filename"]
        score = res["score"]
        if fname not in pairs:
            pairs[fname] = {
                "pair_id": fname,
                "before": {
                    "filename": fname,
                    "url": f"/static/demo/before/{fname}"
                },
                "after": {
                    "filename": fname,
                    "url": f"/static/demo/after/{fname}"
                },
                "similarity": score
            }
        else:
            if score > pairs[fname]["similarity"]:
                pairs[fname]["similarity"] = score
                
    grouped_results = list(pairs.values())
    # Sort by similarity descending
    grouped_results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return JSONResponse({"query": q, "results": grouped_results})


@router.get("/change-demo")
def change_demo():
    # Return V1 and V2 experiment metrics
    images = []
    if os.path.exists(cfg.OUTPUTS_DIR):
        import glob
        paths = glob.glob(os.path.join(cfg.OUTPUTS_DIR, "compare_*.png"))
        images = [p.replace("\\", "/") for p in paths]
        
    return JSONResponse({
        "status": "ok",
        "pairs_evaluated": 20,
        "v1": {
            "false_positives": 1099177,
            "precision": 0.0421,
            "recall": 0.1166,
            "f1": 0.0619,
            "iou": 0.0319
        },
        "v2": {
            "false_positives": 401555,
            "precision": 0.1844,
            "recall": 0.2192,
            "f1": 0.2003,
            "iou": 0.1113
        },
        "reduction_percentage": 63.47,
        "images": images
    })


@router.get("/health")
def health():
    indexed = 0
    if os.path.exists(cfg.INDEX_PATH):
        try:
            index = faiss.read_index(cfg.INDEX_PATH)
            indexed = index.ntotal
        except Exception:
            pass
    return {
        "status": "ok", 
        "indexed": indexed,
        "model": cfg.MODEL_NAME,
        "offline": True
    }
