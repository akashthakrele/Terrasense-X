import os
import faiss
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import config as cfg
from backend.ml.search import text_search, text_search_large

router = APIRouter()


def _match_quality(score: float) -> str:
    """Return an honest label for a RemoteCLIP similarity score.
    SEMANTIC_MATCH  : score >= SEMANTIC_MIN_SCORE (empirical threshold)
    CLOSEST_AVAILABLE: below threshold — FAISS returned something but it
                       may not genuinely represent the queried concept.
    """
    if score >= cfg.SEMANTIC_MIN_SCORE:
        return "SEMANTIC_MATCH"
    return "CLOSEST_AVAILABLE"


@router.get("/search")
def search(
    q: str = Query(..., description="Text query"),
    k: int = Query(10, le=50),
    dataset: str = Query("s2looking")
):
    # ── s2looking_large: 500-pair train index, pair-level dedup ──────────────
    if dataset == "s2looking_large":
        pairs = text_search_large(q, k_pairs=k, dataset="s2looking_large")

        # All results below threshold are labeled CLOSEST_AVAILABLE;
        # no_match is only true when there are zero results
        no_match = len(pairs) == 0

        results = []
        for p in pairs:
            quality = _match_quality(p["score"])
            split = p.get("split", "train")
            pid   = p["pair_id"]
            results.append({
                "pair_id":       pid,
                "similarity":    round(p["score"], 4),
                "match_quality": quality,
                "split":         split,
                "before": {
                    "filename": pid,
                    "url": f"/s2looking/image/{split}/Image2/{pid}"
                },
                "after": {
                    "filename": pid,
                    "url": f"/s2looking/image/{split}/Image1/{pid}"
                },
                "label_url":  f"/s2looking/label/{split}/{pid}",
                "is_curated": False,
                "index":      "s2looking_large"
            })

        return JSONResponse({
            "query":              q,
            "dataset_label":     "S2LOOKING CHANGE BENCHMARK",
            "dataset_note":      "Building-change detection benchmark. Broad semantic coverage is limited.",
            "no_match":          no_match,
            "no_match_reason":   "No results in the S2Looking large index." if no_match else None,
            "results":           results
        })

    # ── s2looking demo (20-pair curated subset) ──────────────────────────────
    raw_results = text_search(q, k, dataset)

    if dataset == "s2looking":
        pairs = {}
        for res in raw_results:
            fname = res["filename"]
            score = res["score"]
            if fname not in pairs:
                pairs[fname] = {
                    "pair_id":  fname,
                    "before":   {"filename": fname, "url": f"/static/demo/after/{fname}"},
                    "after":    {"filename": fname, "url": f"/static/demo/before/{fname}"},
                    "label_url": f"/static/demo/label/{fname}",
                    "similarity": score,
                    "is_curated": False
                }
            else:
                if score > pairs[fname]["similarity"]:
                    pairs[fname]["similarity"] = score

        grouped = sorted(pairs.values(), key=lambda x: x["similarity"], reverse=True)

        # Add match_quality after sorting
        for item in grouped:
            item["match_quality"] = _match_quality(item["similarity"])
            item["similarity"] = round(item["similarity"], 4)

        no_match = len(grouped) == 0

        return JSONResponse({
            "query":          q,
            "dataset_label": "S2LOOKING CHANGE BENCHMARK",
            "dataset_note":  "Building-change detection benchmark. Broad semantic coverage is limited.",
            "no_match":      no_match,
            "no_match_reason": "No results in the local demonstration index." if no_match else None,
            "results":       grouped
        })

    elif dataset == "ssl4eo":
        locs = {}
        for res in raw_results:
            loc_id = res["location_id"]
            if loc_id not in locs:
                locs[loc_id] = {
                    "location_id": loc_id,
                    "similarity":  res["score"],
                    "latitude":    res["latitude"],
                    "longitude":   res["longitude"],
                    "crs":         res["crs"],
                    "spacecraft":  res["spacecraft"],
                    "observations": []
                }
            else:
                if res["score"] > locs[loc_id]["similarity"]:
                    locs[loc_id]["similarity"] = res["score"]

            obs_id = res["observation_id"]
            if not any(o["observation_id"] == obs_id for o in locs[loc_id]["observations"]):
                locs[loc_id]["observations"].append({
                    "observation_id":   obs_id,
                    "filename":         res["filename"],
                    "image_url":        f"/static/ssl4eo/rgb/{loc_id}/{res['filename']}",
                    "acquisition_start": res["acquisition_start"],
                    "cloud_cover":      res["cloud_cover"]
                })

        for loc in locs.values():
            loc["observations"].sort(key=lambda x: x["acquisition_start"])

        grouped = sorted(locs.values(), key=lambda x: x["similarity"], reverse=True)

        SSL4EO_THRESHOLD = 0.20
        no_match = not grouped or grouped[0]["similarity"] < SSL4EO_THRESHOLD

        return JSONResponse({
            "query":       q,
            "no_match":    no_match,
            "no_match_reason": "No strong semantic match in SSL4EO index." if no_match else None,
            "results":     grouped if not no_match else []
        })


# ── Safe S2Looking label endpoint ────────────────────────────────────────────
@router.get("/s2looking/label/{split}/{filename}")
def serve_s2looking_label(split: str, filename: str):
    """Serve a S2Looking ground-truth label image.
    Only allows known splits and strict basename validation.
    """
    if split not in cfg.S2LOOKING_VALID_SPLITS:
        raise HTTPException(status_code=400, detail=f"Unknown split '{split}'")

    # Strict basename check — reject any path traversal attempts
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if split == "demo":
        label_path = os.path.join(cfg.DEMO_LABEL_DIR, safe_name)
    else:
        label_path = os.path.join(cfg.S2LOOKING_FULL_DIR, split, "label", safe_name)

    if not os.path.exists(label_path):
        raise HTTPException(status_code=404, detail="Label not found")

    return FileResponse(label_path, media_type="image/png")


# ── Safe S2Looking image endpoint (for large-index results) ──────────────────
@router.get("/s2looking/image/{split}/{source}/{filename}")
def serve_s2looking_image(split: str, source: str, filename: str):
    """Serve an S2Looking Image1 or Image2 from the full dataset.
    Only allows known splits and Image1/Image2 sources.
    """
    if split not in cfg.S2LOOKING_VALID_SPLITS:
        raise HTTPException(status_code=400, detail=f"Unknown split '{split}'")
    if source not in {"Image1", "Image2"}:
        raise HTTPException(status_code=400, detail=f"Unknown source '{source}'")

    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    img_path = os.path.join(cfg.S2LOOKING_FULL_DIR, split, source, safe_name)

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(img_path, media_type="image/png")


# ── Change detection ─────────────────────────────────────────────────────────
from backend.ml.change_detect import detect_change

@router.get("/analyze")
def analyze(
    pair_id:     str = Query(None),
    dataset:     str = Query("s2looking"),
    split:       str = Query("demo"),
    location_id: str = Query(None),
    obs1:        str = Query(None),
    obs2:        str = Query(None)
):
    if dataset == "ssl4eo":
        if not location_id or not obs1 or not obs2:
            return JSONResponse({"error": "Missing location_id, obs1, or obs2"}, status_code=400)
        img1_path = os.path.join(cfg.SSL4EO_DIR, "rgb", location_id, obs1)
        img2_path = os.path.join(cfg.SSL4EO_DIR, "rgb", location_id, obs2)
        safe_pair_id = f"ssl4eo_{location_id}_{obs1.replace('.png','')}_{obs2.replace('.png','')}.png"
        active_pair_id = safe_pair_id

    elif dataset == "s2looking_large":
        # Images come from the full S2Looking dataset via safe paths
        if not pair_id:
            return JSONResponse({"error": "Missing pair_id"}, status_code=400)
        safe_split = split if split in cfg.S2LOOKING_VALID_SPLITS else "train"
        safe_pid   = os.path.basename(pair_id)
        # S2Looking convention: Image2 is temporally BEFORE, Image1 is temporally AFTER
        img1_path  = os.path.join(cfg.S2LOOKING_FULL_DIR, safe_split, "Image2", safe_pid) # BEFORE
        img2_path  = os.path.join(cfg.S2LOOKING_FULL_DIR, safe_split, "Image1", safe_pid) # AFTER
        active_pair_id = safe_pid

    else:  # s2looking demo
        if not pair_id:
            return JSONResponse({"error": "Missing pair_id"}, status_code=400)
        safe_pid   = os.path.basename(pair_id)
        # Note: demo/before folder contains Image1 (AFTER), demo/after folder contains Image2 (BEFORE)
        img1_path  = os.path.join(cfg.AFTER_DIR, safe_pid)  # BEFORE
        img2_path  = os.path.join(cfg.BEFORE_DIR,  safe_pid) # AFTER
        active_pair_id = safe_pid

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        return JSONResponse({"error": "Images not found at resolved paths"}, status_code=404)

    # Temporary debug check to make temporal ordering explicit in server logs
    print(f"\n[DEBUG] Temporal Pair Resolved: {active_pair_id}")
    print(f"        BEFORE -> {img1_path}")
    print(f"        AFTER  -> {img2_path}\n")

    try:
        result = detect_change(img1_path, img2_path, active_pair_id)
        if dataset == "ssl4eo":
            result["location_id"] = location_id
        return JSONResponse({"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Demo cases endpoint ───────────────────────────────────────────────────────
@router.get("/s2looking-demo")
def s2looking_demo():
    demo_cases = [
        {
            "pair_id": "1009.png", "demo_latitude": 34.0522,
            "demo_longitude": -118.2437, "location_label": "North American Urban Zone",
            "geo_status": "SIMULATED", "dataset": "S2LOOKING",
            "before": {"url": "/static/demo/after/1009.png"},
            "after":  {"url": "/static/demo/before/1009.png"},
            "label_url": "/s2looking/label/demo/1009.png"
        },
        {
            "pair_id": "100.png", "demo_latitude": 51.5074,
            "demo_longitude": -0.1278, "location_label": "European Peri-urban Area",
            "geo_status": "SIMULATED", "dataset": "S2LOOKING",
            "before": {"url": "/static/demo/after/100.png"},
            "after":  {"url": "/static/demo/before/100.png"},
            "label_url": "/s2looking/label/demo/100.png"
        },
        {
            "pair_id": "10.png", "demo_latitude": -33.8688,
            "demo_longitude": 151.2093, "location_label": "Oceania Coastal Region",
            "geo_status": "SIMULATED", "dataset": "S2LOOKING",
            "before": {"url": "/static/demo/after/10.png"},
            "after":  {"url": "/static/demo/before/10.png"},
            "label_url": "/s2looking/label/demo/10.png"
        }
    ]
    return JSONResponse({"status": "ok", "results": demo_cases})


# ── Change benchmark summary ──────────────────────────────────────────────────
@router.get("/change-demo")
def change_demo():
    images = []
    if os.path.exists(cfg.OUTPUTS_DIR):
        import glob
        paths = glob.glob(os.path.join(cfg.OUTPUTS_DIR, "compare_*.png"))
        images = [p.replace("\\", "/") for p in paths]

    return JSONResponse({
        "status": "ok",
        "pairs_evaluated": 20,
        "v1": {"false_positives": 1099177, "precision": 0.0421,
               "recall": 0.1166, "f1": 0.0619, "iou": 0.0319},
        "v2": {"false_positives": 401555,  "precision": 0.1844,
               "recall": 0.2192, "f1": 0.2003, "iou": 0.1113},
        "reduction_percentage": 63.47,
        "images": images
    })


# ── Health ────────────────────────────────────────────────────────────────────
@router.get("/health")
def health():
    indexed = 0
    large_indexed = 0
    if os.path.exists(cfg.INDEX_PATH):
        try:
            index = faiss.read_index(cfg.INDEX_PATH)
            indexed = index.ntotal
        except Exception:
            pass
    if os.path.exists(cfg.S2LOOKING_LARGE_FAISS):
        try:
            li = faiss.read_index(cfg.S2LOOKING_LARGE_FAISS)
            large_indexed = li.ntotal
        except Exception:
            pass
    return {
        "status":         "ok",
        "indexed":        indexed,
        "large_indexed":  large_indexed,
        "model":          cfg.MODEL_NAME,
        "offline":        True,
        "threshold":      cfg.SEMANTIC_MIN_SCORE
    }
