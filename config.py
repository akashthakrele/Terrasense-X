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
INDEX_PATH = os.path.join(BASE_DIR, "static_index", "index.faiss")
NAMES_PATH = os.path.join(BASE_DIR, "static_index", "names.json")

# SSL4EO Search Configuration
SSL4EO_DIR = r"C:\Users\asus\Downloads\ssl4eo-s12_100patches"
SSL4EO_INDEX_PATH = os.path.join(BASE_DIR, "static_index", "ssl4eo", "index.faiss")
SSL4EO_NAMES_PATH = os.path.join(BASE_DIR, "static_index", "ssl4eo", "names.json")

# V1 Evaluation Configuration
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
FRONTEND_DIR = "frontend"
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp", "*.webp")
