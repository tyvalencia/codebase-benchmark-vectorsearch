import os
from pathlib import Path

# --- Data paths ---
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "dat"
TRAIN_TSV = DATA_DIR / "train.tsv"
DEV_TSV = DATA_DIR / "dev.tsv"
TEST_TSV = DATA_DIR / "test.tsv"

# --- Model and retrieval settings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2") 
TOP_K = int(os.getenv("TOP_K", 5))
SEED = int(os.getenv("GLOBAL_SEED", 42))
INDEX_PATH = BASE_DIR / "dat/index.faiss"
RECALL_KS = [1, 2, 5, 10, 15, 20] 