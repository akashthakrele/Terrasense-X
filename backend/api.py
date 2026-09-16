import os
import faiss
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import config as cfg
from backend.ml.search import text_search

router = APIRouter()

@router.get("/search")
def search(q: str = Query(..., description="Text query"), k: int = Query(10, le=50), dataset: str = Query("ssl4eo")):
    raw_results = text_search(q, k, dataset)
    
    if dataset == 's2looking':
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
        grouped_results.sort(key=lambda x: x["similarity"], reverse=True)

        RELEVANCE_THRESHOLD = 0.29
        no_match = False
        if not grouped_results or grouped_results[0]["similarity"] < RELEVANCE_THRESHOLD:
            no_match = True

        return JSONResponse({
            "query": q,
            "no_match": no_match,
            "no_match_reason": (
                "No strong semantic match found in the current local demo index. "
                "The 40-image S2Looking subset may not contain this concept."
            ) if no_match else None,
            "results": grouped_results if not no_match else []
        })

    elif dataset == 'ssl4eo':
        # Group by location ID
        locs = {}
        for res in raw_results:
            loc_id = res["location_id"]
            if loc_id not in locs:
                locs[loc_id] = {
                    "location_id": loc_id,
                    "similarity": res["score"],
                    "latitude": res["latitude"],
                    "longitude": res["longitude"],
                    "crs": res["crs"],
                    "spacecraft": res["spacecraft"],
                    "observations": []
                }
            else:
                if res["score"] > locs[loc_id]["similarity"]:
                    locs[loc_id]["similarity"] = res["score"]
            
            # Add this observation if not already present
            obs_id = res["observation_id"]
            if not any(o["observation_id"] == obs_id for o in locs[loc_id]["observations"]):
                locs[loc_id]["observations"].append({
                    "observation_id": obs_id,
                    "filename": res["filename"],
                    "image_url": f"/static/ssl4eo/rgb/{loc_id}/{res['filename']}",
                    "acquisition_start": res["acquisition_start"],
                    "cloud_cover": res["cloud_cover"]
                })
        
        # Sort each location's observations by date
        for loc in locs.values():
            loc["observations"].sort(key=lambda x: x["acquisition_start"])
            
        grouped_results = list(locs.values())
        grouped_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        RELEVANCE_THRESHOLD = 0.20 # different threshold for SSL4EO
        no_match = False
        if not grouped_results or grouped_results[0]["similarity"] < RELEVANCE_THRESHOLD:
            no_match = True

        return JSONResponse({
            "query": q,
            "no_match": no_match,
            "no_match_reason": "No strong semantic match found in the SSL4EO index." if no_match else None,
            "results": grouped_results if not no_match else []
        })



from backend.ml.change_detect import detect_change

@router.get("/analyze")
def analyze(pair_id: str = Query(None, description="Pair ID for S2Looking"),
            dataset: str = Query("s2looking"),
            location_id: str = Query(None),
            obs1: str = Query(None),
            obs2: str = Query(None)):
    
    if dataset == "ssl4eo":
        if not location_id or not obs1 or not obs2:
            return JSONResponse({"error": "Missing location_id, obs1, or obs2 for SSL4EO"}, status_code=400)
        img1_path = os.path.join(cfg.SSL4EO_DIR, "rgb", location_id, obs1)
        img2_path = os.path.join(cfg.SSL4EO_DIR, "rgb", location_id, obs2)
        safe_pair_id = f"ssl4eo_{location_id}_{obs1.replace('.png', '')}_{obs2.replace('.png', '')}.png"
        active_pair_id = safe_pair_id
    else:
        if not pair_id:
            return JSONResponse({"error": "Missing pair_id for S2Looking"}, status_code=400)
        img1_path = os.path.join(cfg.BEFORE_DIR, pair_id)
        img2_path = os.path.join(cfg.AFTER_DIR, pair_id)
        active_pair_id = pair_id
        
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        return JSONResponse({"error": "Images not found at resolved paths"}, status_code=404)
        
    try:
        result = detect_change(img1_path, img2_path, active_pair_id)
        
        # Add frontend-requested fields for SSL4EO
        if dataset == "ssl4eo":
            result["location_id"] = location_id
            
        return JSONResponse({"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/s2looking-demo")
def s2looking_demo():
    demo_cases = [
        {
            "pair_id": "1009.png",
            "demo_latitude": 34.0522,
            "demo_longitude": -118.2437,
            "location_label": "North American Urban Zone",
            "geo_status": "SIMULATED",
            "dataset": "S2LOOKING",
            "before": {"url": "/static/demo/before/1009.png"},
            "after": {"url": "/static/demo/after/1009.png"},
            "ground_truth_mask": "/static/demo/label/1009.png"
        },
        {
            "pair_id": "100.png",
            "demo_latitude": 51.5074,
            "demo_longitude": -0.1278,
            "location_label": "European Peri-urban Area",
            "geo_status": "SIMULATED",
            "dataset": "S2LOOKING",
            "before": {"url": "/static/demo/before/100.png"},
            "after": {"url": "/static/demo/after/100.png"},
            "ground_truth_mask": "/static/demo/label/100.png"
        },
        {
            "pair_id": "10.png",
            "demo_latitude": -33.8688,
            "demo_longitude": 151.2093,
            "location_label": "Oceania Coastal Region",
            "geo_status": "SIMULATED",
            "dataset": "S2LOOKING",
            "before": {"url": "/static/demo/before/10.png"},
            "after": {"url": "/static/demo/after/10.png"},
            "ground_truth_mask": "/static/demo/label/10.png"
        }
    ]
    return JSONResponse({"status": "ok", "results": demo_cases})


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
