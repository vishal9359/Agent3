"""Vector store management for RAG."""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma

from agent5.config import SETTINGS
from agent5.ollama_compat import get_ollama_embeddings


def get_vectorstore(
    collection_name: str,
    *,
    embed_model: str | None = None,
    ollama_base_url: str | None = None,
    persist_directory: str | None = None,
) -> Chroma:
    """
    Get or create a Chroma vector store.
    
    Args:
        collection_name: Name of the collection
        embed_model: Embedding model name
        ollama_base_url: Ollama base URL
        persist_directory: Directory to persist the database
        
    Returns:
        Chroma vector store instance
    """
    embeddings = get_ollama_embeddings(model=embed_model, base_url=ollama_base_url)
    persist_dir = persist_directory or SETTINGS.chroma_dir
    
    # Ensure directory exists
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def clear_collection(collection_name: str, persist_directory: str | None = None) -> None:
    """
    Clear a Chroma collection.
    
    Args:
        collection_name: Name of the collection to clear
        persist_directory: Directory where the database is persisted
    """
    persist_dir = persist_directory or SETTINGS.chroma_dir
    
    # Get embeddings (required to access collection)
    embeddings = get_ollama_embeddings()
    
    try:
        vs = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        
        # Delete the collection
        vs.delete_collection()
    except Exception:
        # Collection might not exist, that's fine
        pass

