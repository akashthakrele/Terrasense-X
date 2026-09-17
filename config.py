import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BEFORE_DIR = os.path.join(DATA_DIR, "semantic_demo", "before")
AFTER_DIR  = os.path.join(DATA_DIR, "semantic_demo", "after")
DEMO_LABEL_DIR = os.path.join(DATA_DIR, "semantic_demo", "label")

MODEL_NAME = "ViT-B-32"
REMOTECLIP_REPO = "chendelong/RemoteCLIP"
REMOTECLIP_CKPT = "RemoteCLIP-ViT-B-32.pt"
TOP_K = 5
INDEX_DIR  = "static_index"
INDEX_PATH = os.path.join(BASE_DIR, "static_index", "index.faiss")
NAMES_PATH = os.path.join(BASE_DIR, "static_index", "names.json")

# ── S2Looking large index (500-pair train subset) ────────────────────────────
S2LOOKING_LARGE_FAISS = os.path.join(BASE_DIR, "static_index", "s2looking_large.faiss")
S2LOOKING_LARGE_JSON  = os.path.join(BASE_DIR, "static_index", "s2looking_large.json")

# Full S2Looking dataset on local disk (offline, never downloaded at runtime)
S2LOOKING_FULL_DIR = r"C:\Users\asus\Downloads\archive\S2Looking"
S2LOOKING_VALID_SPLITS = {"train", "val", "test", "demo"}

# ── Semantic relevance threshold ─────────────────────────────────────────────
# Observed score distribution from RemoteCLIP on S2Looking images:
#   Meaningful hits cluster at 0.26–0.32 (cosine similarity on L2-normed vectors).
#   Random / noise level is roughly 0.20–0.24.
# A threshold of 0.26 separates meaningful FAISS retrievals from low-signal hits.
# Results below this threshold are returned as "CLOSEST_AVAILABLE" (honest label),
# not as confirmed semantic matches.
# NOTE: This value was chosen empirically from the observed distribution and
# has NOT been validated on a held-out semantic annotation dataset.
SEMANTIC_MIN_SCORE = 0.26

# ── SSL4EO configuration (hidden from judge-facing UI, retained for future) ──
SSL4EO_DIR        = r"C:\Users\asus\Downloads\ssl4eo-s12_100patches"
SSL4EO_INDEX_PATH = os.path.join(BASE_DIR, "static_index", "ssl4eo", "index.faiss")
SSL4EO_NAMES_PATH = os.path.join(BASE_DIR, "static_index", "ssl4eo", "names.json")

# ── General ──────────────────────────────────────────────────────────────────
MODELS_DIR   = "models"
OUTPUTS_DIR  = "outputs"
FRONTEND_DIR = "frontend"
IMAGE_EXTS   = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp", "*.webp")
