"""Configuration settings for Agent5."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Agent5 configuration settings."""
    
    # Ollama settings
    ollama_base_url: str = os.getenv("AGENT5_OLLAMA_BASE_URL") or os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    ollama_chat_model: str = os.getenv("AGENT5_OLLAMA_CHAT_MODEL") or os.getenv(
        "OLLAMA_CHAT_MODEL", "qwen3:8b"
    )
    ollama_embed_model: str = os.getenv("AGENT5_OLLAMA_EMBED_MODEL") or os.getenv(
        "OLLAMA_EMBED_MODEL", "jina/jina-embeddings-v2-base-en"
    )
    
    # ChromaDB settings
    chroma_dir: str = os.getenv("AGENT5_CHROMA_DIR") or os.getenv("CHROMA_DIR", ".chroma")
    
    # AST chunking settings
    min_chunk_lines: int = int(os.getenv("AGENT5_MIN_CHUNK_LINES", "10"))
    max_chunk_lines: int = int(os.getenv("AGENT5_MAX_CHUNK_LINES", "500"))
    chunk_overlap_lines: int = int(os.getenv("AGENT5_CHUNK_OVERLAP_LINES", "20"))


SETTINGS = Settings()

