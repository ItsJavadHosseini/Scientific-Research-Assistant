import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-base",
)

# Qdrant Configuration
QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    "storage/qdrant",
)


# Validate required LLM settings
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

if not OPENAI_BASE_URL:
    raise ValueError("OPENAI_BASE_URL is missing from .env")

if not LLM_MODEL:
    raise ValueError("LLM_MODEL is missing from .env")