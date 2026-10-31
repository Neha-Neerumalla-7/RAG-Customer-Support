"""
Step 7: Centralized Configuration Module
Loads environment variables and sets defaults for all RAG hyperparameters.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Application configuration and hyperparameters."""
    # Data and Persistence Paths
    DATA_PATH: str = os.getenv("DATA_PATH", os.path.join("data", "support_kb.txt"))
    INDEX_PATH: str = os.getenv("INDEX_PATH", "faiss_index")

    # Embedding Hyperparameters
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Retrieval Hyperparameters
    TOP_K: int = int(os.getenv("TOP_K", "3"))

    # Optional API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")


config = AppConfig()
