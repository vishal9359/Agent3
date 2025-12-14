from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import Chroma

from agent3.config import SETTINGS
from agent3.ollama_compat import get_ollama_embeddings


def get_embeddings():
    return get_ollama_embeddings(model=SETTINGS.ollama_embed_model, base_url=SETTINGS.ollama_base_url)


def get_vectorstore(collection: str) -> Chroma:
    persist_dir = Path(SETTINGS.chroma_dir).resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )


