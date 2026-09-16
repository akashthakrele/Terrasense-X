"""Text query -> top-k image results from FAISS index."""
import os
import json
import faiss
import torch
import config as cfg
from backend.ml.embed import load_remoteclip, DEVICE

_model = None
_tokenizer = None
_index = None
_names = None

def _load_resources():
    global _model, _tokenizer, _index, _names
    if _model is None:
        _model, _, _tokenizer = load_remoteclip()
    if _index is None and os.path.exists(cfg.INDEX_PATH):
        _index = faiss.read_index(cfg.INDEX_PATH)
    if _names is None and os.path.exists(cfg.NAMES_PATH):
        with open(cfg.NAMES_PATH, "r") as f:
            _names = json.load(f)

def text_search(query, k=5):
    _load_resources()
    if _index is None or _names is None or _index.ntotal == 0:
        return []
    
    text_tokens = _tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        text_features = _model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
    text_features_np = text_features.cpu().numpy().astype("float32")
    
    D, I = _index.search(text_features_np, k)
    
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(_names):
            continue
        entry = _names[idx]
        results.append({
            "filename": entry["filename"],
            "source": entry["source"],
            "path": entry["path"],
            "score": float(dist)
        })
    return results
