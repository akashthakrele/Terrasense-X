"""
build_s2looking_index.py
------------------------
Builds a RemoteCLIP-based FAISS index from the full S2Looking dataset.

Usage:
    python scripts/build_s2looking_index.py --split train --pairs 500
    python scripts/build_s2looking_index.py --split train --pairs 3500
    python scripts/build_s2looking_index.py --split all --pairs 1000

Outputs (never overwrites the existing demo index):
    static_index/s2looking_large.faiss
    static_index/s2looking_large.json
"""

import argparse
import json
import os
import sys
import time

import faiss
import numpy as np

# project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as cfg
from backend.ml.embed import load_remoteclip, DEVICE

# S2Looking dataset root
S2LOOKING_ROOT = r"C:\Users\asus\Downloads\archive\S2Looking"

SPLIT_DIRS = {
    "train": os.path.join(S2LOOKING_ROOT, "train"),
    "val":   os.path.join(S2LOOKING_ROOT, "val"),
    "test":  os.path.join(S2LOOKING_ROOT, "test"),
}

# output paths
OUT_FAISS = os.path.join(ROOT, "static_index", "s2looking_large.faiss")
OUT_JSON  = os.path.join(ROOT, "static_index", "s2looking_large.json")


def collect_pairs(split, max_pairs):
    if split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [split]

    pairs = []
    for sp in splits:
        sp_dir    = SPLIT_DIRS[sp]
        img1_dir  = os.path.join(sp_dir, "Image1")
        img2_dir  = os.path.join(sp_dir, "Image2")
        label_dir = os.path.join(sp_dir, "label")

        if not os.path.isdir(img1_dir):
            print("  [WARN] " + img1_dir + " not found, skipping split '" + sp + "'")
            continue

        filenames = sorted(
            f for f in os.listdir(img1_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        )
        print("  [" + sp + "] found " + str(len(filenames)) + " Image1 files")

        for fname in filenames:
            if len(pairs) >= max_pairs:
                break
            img1_path  = os.path.join(img1_dir, fname)
            img2_path  = os.path.join(img2_dir, fname)
            label_path = os.path.join(label_dir, fname)

            if not os.path.exists(img1_path):
                continue
            if not os.path.exists(img2_path):
                print("    [WARN] missing Image2 for " + fname)
                continue

            pairs.append({
                "pair_id":    fname,
                "split":      sp,
                "img1_path":  img1_path,
                "img2_path":  img2_path,
                "label_path": label_path if os.path.exists(label_path) else None,
            })

    return pairs[:max_pairs]


def embed_image_batch(model, preprocess, paths, device):
    import torch
    from PIL import Image

    tensors = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(preprocess(img))
        except Exception as e:
            print("    [WARN] could not load " + p + ": " + str(e))
            tensors.append(torch.zeros(3, 224, 224))

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


def main():
    parser = argparse.ArgumentParser(description="Build S2Looking FAISS index")
    parser.add_argument("--split",     default="train",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--pairs",     type=int, default=500)
    parser.add_argument("--batch",     type=int, default=8)
    parser.add_argument("--out_faiss", default=OUT_FAISS)
    parser.add_argument("--out_json",  default=OUT_JSON)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Building S2Looking large index")
    print("  split=" + args.split + "  pairs=" + str(args.pairs) + "  batch=" + str(args.batch))
    print("  device=" + str(DEVICE))
    print("="*60 + "\n")

    # 1. load model
    print("[1/4] Loading RemoteCLIP...")
    model, preprocess, _ = load_remoteclip()
    model.eval()
    print("  RemoteCLIP loaded on " + str(DEVICE))

    # 2. collect pairs
    print("\n[2/4] Collecting pairs from '" + args.split + "' split...")
    pairs = collect_pairs(args.split, args.pairs)
    print("  Collected " + str(len(pairs)) + " pairs")

    if len(pairs) == 0:
        print("ERROR: no pairs found")
        sys.exit(1)

    # 3. build flat lists: Image1 then Image2 for every pair (interleaved)
    print("\n[3/4] Embedding " + str(len(pairs)*2) + " images in batches of " + str(args.batch) + "...")
    t0 = time.time()

    all_paths = []
    all_meta  = []

    for pair in pairs:
        all_paths.append(pair["img1_path"])
        all_meta.append({
            "source":      "before",
            "pair_id":     pair["pair_id"],
            "split":       pair["split"],
            "filename":    pair["pair_id"],
            "path":        pair["img1_path"],
            "before_path": pair["img1_path"],
            "after_path":  pair["img2_path"],
            "label_path":  pair["label_path"],
        })
        all_paths.append(pair["img2_path"])
        all_meta.append({
            "source":      "after",
            "pair_id":     pair["pair_id"],
            "split":       pair["split"],
            "filename":    pair["pair_id"],
            "path":        pair["img2_path"],
            "before_path": pair["img1_path"],
            "after_path":  pair["img2_path"],
            "label_path":  pair["label_path"],
        })

    total_imgs = len(all_paths)
    all_embeddings = []

    for batch_start in range(0, total_imgs, args.batch):
        batch_end   = min(batch_start + args.batch, total_imgs)
        batch_paths = all_paths[batch_start:batch_end]
        pair_start  = batch_start // 2 + 1
        pair_end    = batch_end   // 2
        sys.stdout.write("  Processing pairs " + str(pair_start) + "-" + str(pair_end) + " / " + str(len(pairs)) + "\r")
        sys.stdout.flush()
        emb = embed_image_batch(model, preprocess, batch_paths, DEVICE)
        all_embeddings.append(emb)

    print()
    embeddings = np.vstack(all_embeddings)
    elapsed = time.time() - t0
    print("  Embedded " + str(total_imgs) + " images in " + str(round(elapsed,1)) + "s")

    # 4. build FAISS and save
    print("\n[4/4] Building FAISS IndexFlatIP...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(args.out_faiss), exist_ok=True)
    faiss.write_index(index, args.out_faiss)
    print("  Saved FAISS index: " + args.out_faiss)
    print("  Vectors in index: " + str(index.ntotal))

    for i, m in enumerate(all_meta):
        m["vector_id"] = i

    with open(args.out_json, "w") as f:
        json.dump(all_meta, f, indent=2)
    print("  Saved metadata:   " + args.out_json)

    print("\n" + "="*60)
    print("  DONE")
    print("  Pairs indexed   : " + str(len(pairs)))
    print("  Vectors created : " + str(index.ntotal))
    print("  Dimension       : " + str(dim))
    print("  FAISS file      : " + args.out_faiss)
    print("  Metadata file   : " + args.out_json)
    print("  Build time      : " + str(round(elapsed,1)) + "s")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
