from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agent3.config import SETTINGS
from agent3.ollama_compat import get_chat_ollama
from agent3.vectorstore import get_vectorstore


class RAGState(TypedDict, total=False):
    question: str
    retrieved_docs: list[Document]
    answer: str


SYSTEM_PROMPT = """You are a senior C++ codebase analyst.
Answer ONLY from the provided context (snippets from the repository). Do not guess.
If the context is insufficient, explicitly say what files/functions are missing.

Rules:
- Prefer precise, verifiable statements grounded in the context.
- When you state a fact, cite at least one file path using: (source: <path>).
- If asked for a flow/architecture, provide a step-by-step flow and cite sources per step.
- If multiple plausible interpretations exist, list them and state what evidence would disambiguate.
"""


def _format_context(docs: list[Document], max_chars: int = 25_000) -> str:
    parts: list[str] = []
    total = 0
    for d in docs:
        src = d.metadata.get("relpath") or d.metadata.get("source") or "unknown"
        chunk = d.page_content
        block = f"\n---\nFILE: {src}\n{chunk}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts).strip()


def _dedupe_docs(docs: list[Document]) -> list[Document]:
    """
    Keep first occurrence of (relpath/source, chunk_index, content_hash).
    """
    out: list[Document] = []
    seen: set[tuple[str, int | None, int]] = set()
    for d in docs:
        src = str(d.metadata.get("relpath") or d.metadata.get("source") or "unknown")
        chunk_index = d.metadata.get("chunk_index")
        h = hash(d.page_content)
        key = (src, chunk_index if isinstance(chunk_index, int) else None, h)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _retrieve_docs(vs, q: str, k: int) -> list[Document]:
    """
    Combine similarity search + MMR (if available) for better coverage.
    """
    docs: list[Document] = []
    try:
        docs.extend(vs.similarity_search(q, k=k))
    except Exception:
        pass

    # Chroma supports MMR; fall back silently if not available.
    try:
        mmr = getattr(vs, "max_marginal_relevance_search", None)
        if callable(mmr):
            docs.extend(mmr(q, k=k, fetch_k=max(32, k * 4)))
    except Exception:
        pass

    return _dedupe_docs(docs)


def build_rag_app(
    *,
    collection: str,
    k: int = 8,
    chat_model: str | None = None,
    embed_model: str | None = None,
    ollama_base_url: str | None = None,
    project_path: Path | None = None,
    focus: Path | None = None,
):
    base_url = ollama_base_url or SETTINGS.ollama_base_url
    vs = get_vectorstore(collection, embed_model=embed_model, ollama_base_url=base_url)
    llm = get_chat_ollama(model=chat_model or SETTINGS.ollama_chat_model, base_url=base_url)

    focus_docs: list[Document] = []
    if focus is not None:
        fp = (
            (project_path.resolve() / focus).resolve()
            if project_path is not None and not focus.is_absolute()
            else focus.resolve()
        )
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        rel = (
            str(fp.relative_to(project_path.resolve())).replace("\\", "/")
            if project_path is not None and project_path.resolve() in fp.parents
            else str(fp)
        )
        if text:
            focus_docs.append(Document(page_content=text, metadata={"relpath": rel, "source": str(fp)}))

    def retrieve(state: RAGState) -> RAGState:
        q = state["question"]
        docs = _dedupe_docs(focus_docs + _retrieve_docs(vs, q, k))
        return {"retrieved_docs": docs}

    def generate(state: RAGState) -> RAGState:
        q = state["question"]
        docs = state.get("retrieved_docs", [])
        context = _format_context(docs)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question:\n{q}\n\nContext:\n{context}\n"),
        ]
        resp = llm.invoke(messages)
        return {"answer": getattr(resp, "content", str(resp))}

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


def ask(
    *,
    collection: str,
    question: str,
    k: int = 8,
    chat_model: str | None = None,
    embed_model: str | None = None,
    ollama_base_url: str | None = None,
    project_path: Path | None = None,
    focus: Path | None = None,
) -> str:
    app = build_rag_app(
        collection=collection,
        k=k,
        chat_model=chat_model,
        embed_model=embed_model,
        ollama_base_url=ollama_base_url,
        project_path=project_path,
        focus=focus,
    )
    try:
        out: dict[str, Any] = app.invoke({"question": question})
    except Exception as e:
        msg = str(e)
        # Ollama error message typically looks like:
        # "ollama._types.ResponseError: model 'qwen2.5' not found (status code: 404)"
        if "model" in msg and "not found" in msg:
            raise RuntimeError(
                "Ollama chat model not found. Set one of:\n"
                "- CLI: --model qwen3  (or --model qwen3:8b / qwen2.5-coder)\n"
                "- env: OLLAMA_CHAT_MODEL=qwen3\n"
                f"Underlying error: {msg}"
            ) from e
        raise
    return str(out.get("answer", "")).strip()


