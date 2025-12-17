"""
RAG (Retrieval-Augmented Generation) system for C++ code understanding.

This module provides question-answering capabilities using RAG over indexed C++ projects.
It uses AST-aware chunking for better semantic understanding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agent5.config import SETTINGS
from agent5.ollama_compat import get_chat_ollama
from agent5.vectorstore import get_vectorstore


class RAGState(TypedDict, total=False):
    """State for the RAG graph."""
    
    question: str
    retrieved_docs: list[Document]
    answer: str


SYSTEM_PROMPT = """You are a senior C++ codebase analyst with deep expertise in software architecture.

Answer ONLY from the provided context (code snippets from the repository). Do not guess or hallucinate.
If the context is insufficient, explicitly state what files/functions/information is missing.

Rules:
- Prefer precise, verifiable statements grounded in the provided code
- When stating a fact, cite at least one file path: (source: <path>)
- For flow/architecture questions, provide step-by-step explanation with citations
- If multiple interpretations exist, list them and state what would disambiguate
- Use the semantic metadata (function names, qualified names, dependencies) to understand relationships
- Pay attention to chunk_type metadata: 'function', 'class', 'namespace', 'header'
"""


def _format_context(docs: list[Document], max_chars: int = 30000) -> str:
    """
    Format retrieved documents as context for the LLM.
    
    Args:
        docs: Retrieved documents
        max_chars: Maximum characters to include
        
    Returns:
        Formatted context string
    """
    parts: list[str] = []
    total = 0
    
    for d in docs:
        # Extract metadata
        src = d.metadata.get("relpath") or d.metadata.get("source") or "unknown"
        chunk_type = d.metadata.get("chunk_type", "code")
        name = d.metadata.get("name", "")
        qualified_name = d.metadata.get("qualified_name", "")
        
        # Build header
        header = f"\n--- FILE: {src}"
        if chunk_type != "file" and name:
            header += f" | {chunk_type.upper()}: {qualified_name or name}"
        header += " ---\n"
        
        # Add content
        chunk = d.page_content
        block = header + chunk + "\n"
        
        if total + len(block) > max_chars:
            break
        
        parts.append(block)
        total += len(block)
    
    return "".join(parts).strip()


def _dedupe_docs(docs: list[Document]) -> list[Document]:
    """
    Remove duplicate documents.
    
    Args:
        docs: List of documents
        
    Returns:
        Deduplicated list
    """
    seen: set[tuple[str, str]] = set()
    result: list[Document] = []
    
    for d in docs:
        src = d.metadata.get("relpath") or d.metadata.get("source") or "unknown"
        qname = d.metadata.get("qualified_name", "")
        key = (src, qname)
        
        if key not in seen:
            seen.add(key)
            result.append(d)
    
    return result


def _retrieve_docs(vs, query: str, k: int) -> list[Document]:
    """
    Retrieve relevant documents from vector store.
    
    Args:
        vs: Vector store
        query: Search query
        k: Number of documents to retrieve
        
    Returns:
        List of retrieved documents
    """
    docs: list[Document] = []
    
    # Similarity search
    try:
        docs.extend(vs.similarity_search(query, k=k))
    except Exception:
        pass
    
    # MMR (Maximal Marginal Relevance) for diversity
    try:
        mmr = getattr(vs, "max_marginal_relevance_search", None)
        if callable(mmr):
            docs.extend(mmr(query, k=k, fetch_k=max(32, k * 4)))
    except Exception:
        pass
    
    return _dedupe_docs(docs)


def build_rag_app(
    collection: str,
    *,
    k: int = 10,
    chat_model: str | None = None,
    embed_model: str | None = None,
    ollama_base_url: str | None = None,
    focus_file: Path | None = None,
    project_path: Path | None = None,
):
    """
    Build a RAG application for answering questions about a C++ project.
    
    Args:
        collection: Name of the Chroma collection
        k: Number of documents to retrieve
        chat_model: Chat model name
        embed_model: Embedding model name
        ollama_base_url: Ollama base URL
        focus_file: Optional file to include in full
        project_path: Project root path
        
    Returns:
        Compiled LangGraph application
    """
    base_url = ollama_base_url or SETTINGS.ollama_base_url
    vs = get_vectorstore(collection, embed_model=embed_model, ollama_base_url=base_url)
    llm = get_chat_ollama(model=chat_model or SETTINGS.ollama_chat_model, base_url=base_url)
    
    # Load focus file if provided
    focus_docs: list[Document] = []
    if focus_file and focus_file.exists():
        try:
            text = focus_file.read_text(encoding="utf-8", errors="ignore")
            if text:
                rel = str(focus_file)
                if project_path and project_path in focus_file.parents:
                    try:
                        rel = str(focus_file.relative_to(project_path))
                    except ValueError:
                        pass
                
                focus_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(focus_file),
                            "relpath": rel,
                            "chunk_type": "focus",
                        },
                    )
                )
        except Exception:
            pass
    
    def retrieve(state: RAGState) -> RAGState:
        """Retrieve relevant documents."""
        q = state["question"]
        docs = _retrieve_docs(vs, q, k)
        
        # Add focus docs
        if focus_docs:
            docs = focus_docs + docs
        
        return {"retrieved_docs": _dedupe_docs(docs)}
    
    def generate(state: RAGState) -> RAGState:
        """Generate answer using LLM."""
        q = state["question"]
        docs = state.get("retrieved_docs", [])
        context = _format_context(docs)
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question:\n{q}\n\nContext:\n{context}\n"),
        ]
        
        resp = llm.invoke(messages)
        answer = getattr(resp, "content", str(resp))
        
        return {"answer": answer}
    
    # Build graph
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    return graph.compile()


def ask_question(
    collection: str,
    question: str,
    *,
    k: int = 10,
    chat_model: str | None = None,
    embed_model: str | None = None,
    ollama_base_url: str | None = None,
    focus_file: Path | None = None,
    project_path: Path | None = None,
) -> str:
    """
    Ask a question about a C++ project using RAG.
    
    Args:
        collection: Name of the indexed collection
        question: Question to ask
        k: Number of documents to retrieve
        chat_model: Chat model name
        embed_model: Embedding model name
        ollama_base_url: Ollama base URL
        focus_file: Optional file to focus on
        project_path: Project root path
        
    Returns:
        Answer string
        
    Raises:
        RuntimeError: If model not found or other error
    """
    app = build_rag_app(
        collection,
        k=k,
        chat_model=chat_model,
        embed_model=embed_model,
        ollama_base_url=ollama_base_url,
        focus_file=focus_file,
        project_path=project_path,
    )
    
    try:
        result: dict[str, Any] = app.invoke({"question": question})
    except Exception as e:
        msg = str(e)
        if "model" in msg and "not found" in msg:
            raise RuntimeError(
                "Ollama chat model not found. Please set:\n"
                "- CLI: --chat_model qwen3:8b\n"
                "- env: OLLAMA_CHAT_MODEL=qwen3:8b\n"
                f"Error: {msg}"
            ) from e
        raise
    
    return str(result.get("answer", "")).strip()

