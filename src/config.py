from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

NVIDIA_RAW_DIR = RAW_DATA_DIR / "nvidia"
NVIDIA_PROCESSED_DIR = PROCESSED_DATA_DIR / "nvidia"

# NVIDIA
NVIDIA_CIK = "0001045810"
NVIDIA_TICKER = "NVDA"
NVIDIA_COMPANY = "NVIDIA Corporation"

# SEC
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "Jordan Phillips jordan@example.com"
)

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

# Retrieval
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 5

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.8-flash"