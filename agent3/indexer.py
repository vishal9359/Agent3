from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from agent3.cpp_loader import build_cpp_documents, chunk_documents
from agent3.logging_utils import console
from agent3.vectorstore import get_vectorstore


def index_project(
    *,
    project_path: Path,
    collection: str,
    scope: Path | None = None,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    clear_collection: bool = False,
    ollama_base_url: str | None = None,
    embed_model: str | None = None,
) -> None:
    vs = get_vectorstore(collection, embed_model=embed_model, ollama_base_url=ollama_base_url)
    if clear_collection:
        console.print(f"[yellow]Clearing collection:[/yellow] {collection}")
        vs.delete_collection()
        vs = get_vectorstore(collection, embed_model=embed_model, ollama_base_url=ollama_base_url)

    console.print("[cyan]Scanning C++ files...[/cyan]")
    docs = build_cpp_documents(project_path=project_path, scope=scope)
    console.print(f"[green]Loaded files:[/green] {len(docs)}")

    console.print("[cyan]Chunking...[/cyan]")
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    console.print(f"[green]Chunks:[/green] {len(chunks)}")

    console.print("[cyan]Embedding + indexing into Chroma...[/cyan]")
    # Add in batches to avoid huge requests.
    batch = 128
    for i in tqdm(range(0, len(chunks), batch)):
        vs.add_documents(chunks[i : i + batch])

    # Newer Chroma persists automatically, but persist() keeps compatibility.
    try:
        vs.persist()
    except Exception:
        pass

    console.print(f"[green]Done.[/green] Collection='{collection}' dir='{vs._persist_directory}'")


