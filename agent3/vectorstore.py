from __future__ import annotations

from pathlib import Path

try:
    # langchain-community has deprecated Chroma; prefer the dedicated package.
    from langchain_chroma import Chroma  # type: ignore
except Exception:  # pragma: no cover
    from langchain_community.vectorstores import Chroma  # type: ignore

from agent3.config import SETTINGS
from agent3.ollama_compat import get_ollama_embeddings


def get_embeddings(*, embed_model: str | None = None, ollama_base_url: str | None = None):
    return get_ollama_embeddings(
        model=embed_model or SETTINGS.ollama_embed_model,
        base_url=ollama_base_url or SETTINGS.ollama_base_url,
    )


def get_vectorstore(
    collection: str,
    *,
    embed_model: str | None = None,
    ollama_base_url: str | None = None,
    chroma_dir: str | None = None,
) -> Chroma:
    persist_dir = Path(chroma_dir or SETTINGS.chroma_dir).resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(
            embed_model=embed_model,
            ollama_base_url=ollama_base_url,
        ),
        persist_directory=str(persist_dir),
    )


