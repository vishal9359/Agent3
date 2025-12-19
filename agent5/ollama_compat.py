"""Ollama compatibility layer for LangChain."""
from __future__ import annotations

from agent5.config import SETTINGS


def get_chat_ollama(model: str | None = None, base_url: str | None = None):
    """
    Get an Ollama chat model instance.
    
    Args:
        model: Model name (defaults to config)
        base_url: Ollama base URL (defaults to config)
        
    Returns:
        LangChain ChatOllama instance
    """
    from langchain_ollama import ChatOllama
    
    return ChatOllama(
        model=model or SETTINGS.ollama_chat_model,
        base_url=base_url or SETTINGS.ollama_base_url,
        temperature=0.0,
    )


def get_ollama_llm(model: str | None = None, base_url: str | None = None):
    """
    Get an Ollama LLM instance (alias for get_chat_ollama for backward compatibility).
    
    Args:
        model: Model name (defaults to config)
        base_url: Ollama base URL (defaults to config)
        
    Returns:
        LangChain ChatOllama instance
    """
    return get_chat_ollama(model=model, base_url=base_url)


def get_ollama_embeddings(model: str | None = None, base_url: str | None = None):
    """
    Get an Ollama embeddings instance.
    
    Args:
        model: Embedding model name (defaults to config)
        base_url: Ollama base URL (defaults to config)
        
    Returns:
        LangChain OllamaEmbeddings instance
    """
    from langchain_ollama import OllamaEmbeddings
    
    return OllamaEmbeddings(
        model=model or SETTINGS.ollama_embed_model,
        base_url=base_url or SETTINGS.ollama_base_url,
    )

