from __future__ import annotations

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
Use ONLY the provided context to answer. If the context is insufficient, say what you need.

Rules:
- Prefer precise, verifiable statements grounded in the context.
- Cite file paths from metadata using the format: (source: <path>).
- If asked for flow/architecture, summarize control flow and key modules.
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


def build_rag_app(*, collection: str, k: int = 8):
    vs = get_vectorstore(collection)
    llm = get_chat_ollama(model=SETTINGS.ollama_chat_model, base_url=SETTINGS.ollama_base_url)

    def retrieve(state: RAGState) -> RAGState:
        q = state["question"]
        docs = vs.similarity_search(q, k=k)
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


def ask(*, collection: str, question: str, k: int = 8) -> str:
    app = build_rag_app(collection=collection, k=k)
    out: dict[str, Any] = app.invoke({"question": question})
    return str(out.get("answer", "")).strip()


