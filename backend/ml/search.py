"""Text query -> top-k image results from FAISS index."""
import os
import json
import faiss
import torch
import config as cfg
from backend.ml.embed import load_remoteclip, DEVICE

_model = None
_tokenizer = None

# S2Looking cache
_index_s2looking = None
_names_s2looking = None

# SSL4EO cache
_index_ssl4eo = None
_names_ssl4eo = None

def _load_resources(dataset='ssl4eo'):
    global _model, _tokenizer
    global _index_s2looking, _names_s2looking
    global _index_ssl4eo, _names_ssl4eo
    
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
        
    return None, None

def text_search(query, k=5, dataset='ssl4eo'):
    idx, nms = _load_resources(dataset)
    if idx is None or nms is None or idx.ntotal == 0:
        return []
    
    text_tokens = _tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        text_features = _model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
    text_features_np = text_features.cpu().numpy().astype("float32")
    
    D, I = idx.search(text_features_np, k)
    
    results = []
    for dist, idx_val in zip(D[0], I[0]):
        if idx_val < 0 or idx_val >= len(nms):
            continue
        entry = nms[idx_val]
        entry_data = {
            "score": float(dist)
        }
        entry_data.update(entry)
        results.append(entry_data)
    return results
