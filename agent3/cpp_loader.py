from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent3.fs_utils import CPP_EXTS, RAG_EXTRA_EXTS, iter_files, safe_read_text


def build_cpp_documents(
    project_path: Path,
    *,
    scope: Path | None = None,
    exclude_dir_names: set[str] | None = None,
) -> list[Document]:
    project_path = project_path.resolve()
    scope_path = (scope or project_path).resolve()

    if not scope_path.exists():
        raise FileNotFoundError(f"Scope path not found: {scope_path}")
    if not project_path.exists():
        raise FileNotFoundError(f"Project path not found: {project_path}")

    docs: list[Document] = []
    for rec in iter_files(
        scope_path, include_exts=(CPP_EXTS | RAG_EXTRA_EXTS), exclude_dir_names=exclude_dir_names
    ):
        text = safe_read_text(rec.path)
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(rec.path),
                    "relpath": rec.relpath,
                    "ext": rec.path.suffix.lower(),
                },
            )
        )
    return docs


def chunk_documents(
    docs: list[Document],
    *,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: list[Document] = []
    for d in docs:
        parts = splitter.split_text(d.page_content)
        for i, part in enumerate(parts):
            md = dict(d.metadata)
            md["chunk_index"] = i
            chunks.append(Document(page_content=part, metadata=md))
    return chunks


