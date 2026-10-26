"""Central configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- paths -----------------------------------------------------------------
DATA_CSV = Path(os.environ.get("CROP_CSV", ROOT / "data" / "Crop_recommendation.csv"))
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", ROOT / "data" / "knowledge"))
MODEL_PATH = ROOT / "models" / "crop_model.joblib"
CHROMA_DIR = ROOT / "models" / "chroma_db"
OUTPUT_DIR = ROOT / "outputs"

# ---- dataset / model -------------------------------------------------------
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LABEL = "label"
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Physically plausible limits (anything outside is rejected as invalid input).
PLAUSIBLE_RANGES = {
    "N": (0, 300),          # kg/ha
    "P": (0, 300),          # kg/ha
    "K": (0, 400),          # kg/ha
    "temperature": (-10, 60),   # deg C
    "humidity": (0, 100),       # %
    "ph": (0, 14),
    "rainfall": (0, 1000),      # mm
}

# ---- Ollama ----------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ---- agent behaviour -------------------------------------------------------
MIN_CONFIDENCE = 0.50      # below this the report carries a low-confidence warning
CLOSE_CALL_MARGIN = 0.15   # top-1 minus top-2 probability below this = "close call"
RAG_TOP_K = 3
MIN_RAG_SCORE_CHROMA = 0.35   # cosine similarity floor for embedding search
MIN_RAG_SCORE_TFIDF = 0.02    # cosine similarity floor for TF-IDF fallback
MAX_REVISIONS = 1             # at most ONE review-triggered revision
