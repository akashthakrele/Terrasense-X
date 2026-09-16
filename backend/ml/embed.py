"""
Build FAISS index from satellite images using RemoteCLIP (ViT-B-32) embeddings.

RemoteCLIP: CLIP fine-tuned on remote sensing imagery.
  - Repo: huggingface.co/chendelong/RemoteCLIP
  - Architecture: open_clip ViT-B-32 + fine-tuned checkpoint
  - Loading: open_clip.create_model_and_transforms() + torch.load(checkpoint)

Usage:
    python backend/ml/embed.py
"""
import os
import sys
import glob
import json
import numpy as np
import faiss
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
import open_clip

# Add project root to path so config imports work when run standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as cfg

# ── Device ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_remoteclip():
    """Download RemoteCLIP checkpoint from HF and load into open_clip architecture."""
    print(f"[model] Downloading RemoteCLIP checkpoint: {cfg.REMOTECLIP_REPO}/{cfg.REMOTECLIP_CKPT}")
    ckpt_path = hf_hub_download(
        repo_id=cfg.REMOTECLIP_REPO,
        filename=cfg.REMOTECLIP_CKPT,
        cache_dir=cfg.MODELS_DIR,
    )
    print(f"[model] Checkpoint at: {ckpt_path}")

    model, _, preprocess = open_clip.create_model_and_transforms(cfg.MODEL_NAME)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    # RemoteCLIP checkpoints may store state_dict directly or under a key
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(DEVICE)
    model.eval()

    tokenizer = open_clip.get_tokenizer(cfg.MODEL_NAME)
    print(f"[model] RemoteCLIP {cfg.MODEL_NAME} loaded on {DEVICE}")
    return model, preprocess, tokenizer


def gather_images():
    """Scan data/before/ and data/after/ for image files."""
    entries = []
    for source_name, source_dir in [("before", cfg.BEFORE_DIR), ("after", cfg.AFTER_DIR)]:
        if not os.path.isdir(source_dir):
            continue
        for ext in cfg.IMAGE_EXTS:
            for path in glob.glob(os.path.join(source_dir, "**", ext), recursive=True):
                entries.append({
                    "filename": os.path.basename(path),
                    "source": source_name,
                    "path": path.replace("\\", "/"),
                })
    # Sort for deterministic ordering
    entries.sort(key=lambda e: e["path"])
    for i, e in enumerate(entries):
        e["id"] = i
    return entries


def embed_images(model, preprocess, entries, batch_size=32):
    """Generate normalized embeddings for all image entries."""
    all_embs = []
    valid_entries = []
    for i in range(0, len(entries), batch_size):
        batch_entries = entries[i : i + batch_size]
        imgs = []
        batch_valid = []
        for entry in batch_entries:
            try:
                img = Image.open(entry["path"]).convert("RGB")
                imgs.append(preprocess(img))
                batch_valid.append(entry)
            except Exception as e:
                print(f"[WARN] Corrupt/unreadable image skipped: {entry['path']} — {e}")
        if not imgs:
            continue
        batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            embs = model.encode_image(batch)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().numpy())
        valid_entries.extend(batch_valid)
    if not all_embs:
        return np.array([]), valid_entries
    return np.vstack(all_embs).astype("float32"), valid_entries


def build_index(model=None, preprocess=None):
    """Full pipeline: scan images → embed → build FAISS index → save."""
    # Gather
    entries = gather_images()
    print(f"[index] Found {len(entries)} images in {cfg.BEFORE_DIR}/ and {cfg.AFTER_DIR}/")
    if not entries:
        print("[index] No images found. Add images to data/before/ and data/after/ and re-run.")
        return None, []

    # Load model if not passed
    if model is None or preprocess is None:
        model, preprocess, _ = load_remoteclip()

    # Embed
    embs, valid_entries = embed_images(model, preprocess, entries)
    if embs.size == 0:
        print("[index] All images were corrupt. No index built.")
        return None, []

    # Re-assign IDs after filtering
    for i, e in enumerate(valid_entries):
        e["id"] = i

    # Build FAISS index (cosine sim via inner product on L2-normed vectors)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # Save
    os.makedirs(cfg.INDEX_DIR, exist_ok=True)
    faiss.write_index(index, cfg.INDEX_PATH)
    with open(cfg.NAMES_PATH, "w") as f:
        json.dump(valid_entries, f, indent=2)

    print(f"[index] Embedding dimension: {dim}")
    print(f"[index] FAISS index.ntotal: {index.ntotal}")
    print(f"[index] Saved: {cfg.INDEX_PATH}")
    print(f"[index] Saved: {cfg.NAMES_PATH}")
    return index, valid_entries


if __name__ == "__main__":
    build_index()
