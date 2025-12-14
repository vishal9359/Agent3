from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5")
    ollama_embed_model: str = os.getenv(
        "OLLAMA_EMBED_MODEL", "jina/jina-embeddings-v2-base-en"
    )
    chroma_dir: str = os.getenv("CHROMA_DIR", ".chroma")


SETTINGS = Settings()


