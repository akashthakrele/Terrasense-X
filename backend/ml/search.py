"""Text query -> top-k image results from FAISS index."""
import os
import json
import faiss
import torch
import config as cfg
from backend.ml.embed import load_remoteclip, DEVICE

_model = None
_tokenizer = None

# S2Looking small demo cache
_index_s2looking = None
_names_s2looking = None

# SSL4EO cache
_index_ssl4eo = None
_names_ssl4eo = None

# S2Looking large index cache
_index_s2looking_large = None
_names_s2looking_large = None

S2LOOKING_LARGE_FAISS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "static_index", "s2looking_large.faiss"
)
S2LOOKING_LARGE_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "static_index", "s2looking_large.json"
)


def _load_resources(dataset='ssl4eo'):
    global _model, _tokenizer
    global _index_s2looking, _names_s2looking
    global _index_ssl4eo, _names_ssl4eo
    global _index_s2looking_large, _names_s2looking_large

    if _model is None:
        _model, _, _tokenizer = load_remoteclip()

    if dataset == 's2looking':
        if _index_s2looking is None and os.path.exists(cfg.INDEX_PATH):
            _index_s2looking = faiss.read_index(cfg.INDEX_PATH)
        if _names_s2looking is None and os.path.exists(cfg.NAMES_PATH):
            with open(cfg.NAMES_PATH, "r") as f:
                _names_s2looking = json.load(f)
        return _index_s2looking, _names_s2looking

    elif dataset == 'ssl4eo':
        if _index_ssl4eo is None and os.path.exists(cfg.SSL4EO_INDEX_PATH):
            _index_ssl4eo = faiss.read_index(cfg.SSL4EO_INDEX_PATH)
        if _names_ssl4eo is None and os.path.exists(cfg.SSL4EO_NAMES_PATH):
            with open(cfg.SSL4EO_NAMES_PATH, "r") as f:
                _names_ssl4eo = json.load(f)
        return _index_ssl4eo, _names_ssl4eo

    elif dataset == 's2looking_large':
        if _index_s2looking_large is None and os.path.exists(S2LOOKING_LARGE_FAISS):
            _index_s2looking_large = faiss.read_index(S2LOOKING_LARGE_FAISS)
            print("[search] Loaded s2looking_large FAISS index:"
                  f" {_index_s2looking_large.ntotal} vectors")
        if _names_s2looking_large is None and os.path.exists(S2LOOKING_LARGE_JSON):
            with open(S2LOOKING_LARGE_JSON, "r") as f:
                _names_s2looking_large = json.load(f)
        return _index_s2looking_large, _names_s2looking_large

    return None, None


def _encode_text(query: str):
    """Encode a text query with RemoteCLIP, return (1, 512) float32 numpy."""
    text_tokens = _tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        text_features = _model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype("float32")


def text_search(query, k=5, dataset='ssl4eo'):
    """Search the requested index, return raw per-vector results (no pair dedup)."""
    idx, nms = _load_resources(dataset)
    if idx is None or nms is None or idx.ntotal == 0:
        return []

    text_features_np = _encode_text(query)
    D, I = idx.search(text_features_np, k)

    results = []
    for dist, idx_val in zip(D[0], I[0]):
        if idx_val < 0 or idx_val >= len(nms):
            continue
        entry = dict(nms[idx_val])
        entry["score"] = float(dist)
        results.append(entry)
    return results


def text_search_large(query, k_pairs=10, dataset='s2looking_large'):
    """
    Search the large S2Looking index and return up to k_pairs UNIQUE pairs.

    Before + after of the same pair_id are merged into one result entry.
    The best (highest) score from either image is used for ranking.
    """
    # Fetch more raw vectors than needed because before+after collapse to 1 pair
    raw_k = k_pairs * 4
    idx, nms = _load_resources(dataset)
    if idx is None or nms is None or idx.ntotal == 0:
        return []

    text_features_np = _encode_text(query)
    D, I = idx.search(text_features_np, min(raw_k, idx.ntotal))

    # Deduplicate by pair_id — keep best score
    seen_pairs = {}
    for dist, idx_val in zip(D[0], I[0]):
        if idx_val < 0 or idx_val >= len(nms):
            continue
        entry   = nms[idx_val]
        pair_id = entry["pair_id"]
        score   = float(dist)

        if pair_id not in seen_pairs:
            seen_pairs[pair_id] = {
                "score":       score,
                "pair_id":     pair_id,
                "split":       entry.get("split", ""),
                "filename":    pair_id,
                "before_path": entry.get("before_path", ""),
                "after_path":  entry.get("after_path", ""),
                "label_path":  entry.get("label_path", ""),
            }
        else:
            # keep highest score across before/after
            if score > seen_pairs[pair_id]["score"]:
                seen_pairs[pair_id]["score"] = score

    # Sort by score descending
    ranked = sorted(seen_pairs.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:k_pairs]
