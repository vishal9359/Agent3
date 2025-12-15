from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Prefer AGENT3_* env vars (local override), fall back to standard OLLAMA_*.
    ollama_base_url: str = os.getenv("AGENT3_OLLAMA_BASE_URL") or os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    # NOTE: Default changed from qwen2.5 -> qwen3 because many Ollama installs
    # don't have a bare 'qwen2.5' tag, but do have 'qwen3' / 'qwen3:8b'.
    ollama_chat_model: str = os.getenv("AGENT3_OLLAMA_CHAT_MODEL") or os.getenv(
        "OLLAMA_CHAT_MODEL", "qwen3"
    )
    ollama_embed_model: str = os.getenv("AGENT3_OLLAMA_EMBED_MODEL") or os.getenv(
        "OLLAMA_EMBED_MODEL", "jina/jina-embeddings-v2-base-en"
    )
    chroma_dir: str = os.getenv("AGENT3_CHROMA_DIR") or os.getenv("CHROMA_DIR", ".chroma")


SETTINGS = Settings()


