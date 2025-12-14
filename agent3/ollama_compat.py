from __future__ import annotations

from typing import Any


def get_chat_ollama(*, model: str, base_url: str) -> Any:
    """
    Prefer the dedicated langchain-ollama package; fall back to langchain-community.
    Returns an object with an .invoke(messages) method.
    """
    try:
        from langchain_ollama import ChatOllama  # type: ignore

        return ChatOllama(model=model, base_url=base_url)
    except Exception:
        from langchain_community.chat_models import ChatOllama  # type: ignore

        return ChatOllama(model=model, base_url=base_url)


def get_ollama_embeddings(*, model: str, base_url: str) -> Any:
    """
    Prefer the dedicated langchain-ollama package; fall back to langchain-community.
    Returns an embeddings object compatible with Chroma.
    """
    try:
        from langchain_ollama import OllamaEmbeddings  # type: ignore

        return OllamaEmbeddings(model=model, base_url=base_url)
    except Exception:
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

        return OllamaEmbeddings(model=model, base_url=base_url)


