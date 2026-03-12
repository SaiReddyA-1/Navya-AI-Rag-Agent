from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    """
    Centralized configuration room for the Enterprise RAG Pipeline.
    Every module must pull its magic numbers and keys from this single source of truth.
    """

    # --- System & Telemetry ---
    APP_NAME: str = "Enterprise Document Intelligence Platform"
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"

    # --- Ingestion Configuration ---
    SUPPORTED_EXTENSIONS: List[str] = [
        ".pdf", ".docx", ".txt", ".json", ".xml",
        ".jpg", ".jpeg", ".png",
        ".csv", ".xlsx", ".xls"
    ]

    # --- Processing & Triage ---
    MIN_TEXT_THRESHOLD: int = 50
    TRIAGE_QUALITY_THRESHOLD: float = 0.60

    # --- Chunking Engine ---
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 150

    # --- Embedding Engine & Search ---
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSIONS: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32

    # --- RAG Models ---
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    TOP_K_RETRIEVAL: int = 15
    TOP_K_RERANK: int = 5

    # --- Groq LLM ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"

    # --- OpenSearch ---
    OPENSEARCH_HOST: str = "localhost"
    OPENSEARCH_PORT: int = 9200
    VECTOR_INDEX_NAME: str = "enterprise_docs"

    # --- Data ---
    DATA_DIR: str = "Data/Rag"
    MANIFEST_FILE: str = "Data/file_manifest.json"

    class Config:
        env_file = ".env"

# Instantiate centrally so it's initialized only once.
settings = Settings()
