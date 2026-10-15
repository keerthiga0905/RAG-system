"""
Application Configuration for CIT Knowledge AI Backend
Reads environment variables, sets directory paths, and default model parameters.
"""

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "CIT Knowledge AI Backend"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # Directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    PROCESSED_DIR: str = os.path.join(DATA_DIR, "processed")
    RAW_DIR: str = os.path.join(DATA_DIR, "raw")
    WEB_CACHE_DIR: str = os.path.join(DATA_DIR, "web_cache")
    VECTOR_STORE_DIR: str = os.path.join(BASE_DIR, "vector_store")
    CHROMA_DIR: str = os.path.join(VECTOR_STORE_DIR, "chroma")
    REPORTS_DIR: str = os.path.join(BASE_DIR, "reports")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")

    # Models
    BASELINE_MODEL_PATH: str = os.path.join(MODELS_DIR, "baseline", "all-MiniLM-L6-v2")
    FINETUNED_MODEL_PATH: str = os.path.join(MODELS_DIR, "finetuned", "cit-minilm-l6-v2")
    DEFAULT_RETRIEVAL_STRATEGY: str = "hybrid"  # "baseline", "finetuned", "hybrid"

    # Retrieval parameters
    DEFAULT_TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.35  # Threshold below which evidence is deemed insufficient

    # Multi-Source Pipeline Configuration
    ENABLE_LEVEL2_CIT_WEBSITE: bool = True
    ENABLE_LEVEL3_PUBLIC_WEB: bool = True
    WEB_TIMEOUT_SECONDS: float = 8.0
    WEB_CACHE_TTL_HOURS: int = 24
    OFFICIAL_CIT_BASE_URL: str = "https://cit.edu.in"

    # Optional External Search & LLM Keys (Read from .env)
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    SERPER_API_KEY: str = ""
    TAVILY_API_KEY: str = ""

    # CORS
    CORS_ORIGINS: list = ["*"]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
