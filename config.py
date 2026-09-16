import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BEFORE_DIR = os.path.join(DATA_DIR, "semantic_demo", "before")
AFTER_DIR = os.path.join(DATA_DIR, "semantic_demo", "after")

MODEL_NAME = "ViT-B-32"
REMOTECLIP_REPO = "chendelong/RemoteCLIP"
REMOTECLIP_CKPT = "RemoteCLIP-ViT-B-32.pt"
TOP_K = 5
INDEX_DIR = "static_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
NAMES_PATH = os.path.join(INDEX_DIR, "names.json")
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
FRONTEND_DIR = "frontend"
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp", "*.webp")
