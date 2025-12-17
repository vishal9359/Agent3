"""Project indexing functionality."""
from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from agent5.cpp_loader import build_cpp_documents
from agent5.logging_utils import console
from agent5.vectorstore import clear_collection, get_vectorstore


def index_project(
    *,
    project_path: Path,
    collection: str,
    scope: Path | None = None,
    clear_collection_first: bool = False,
    ollama_base_url: str | None = None,
    embed_model: str | None = None,
    batch_size: int = 100,
) -> int:
    """
    Index a C++ project into a vector store using AST-aware chunking.
    
    Args:
        project_path: Root path of the project
        collection: Name of the collection to create/update
        scope: Optional scope path to limit indexing
        clear_collection_first: Clear existing collection first
        ollama_base_url: Ollama base URL
        embed_model: Embedding model name
        batch_size: Number of documents to add at once
        
    Returns:
        Number of documents indexed
    """
    project_path = project_path.resolve()
    
    console.print(f"[bold cyan]Indexing project:[/bold cyan] {project_path}")
    console.print(f"[bold cyan]Collection:[/bold cyan] {collection}")
    
    if clear_collection_first:
        console.print("[yellow]Clearing existing collection...[/yellow]")
        clear_collection(collection)
    
    console.print("[cyan]Building AST-aware document chunks...[/cyan]")
    docs = build_cpp_documents(
        project_path,
        scope=scope,
        use_ast_chunking=True,
    )
    
    if not docs:
        console.print("[red]No documents found to index![/red]")
        return 0
    
    console.print(f"[green]Found {len(docs)} semantic chunks[/green]")
    
    # Get vector store
    vs = get_vectorstore(
        collection,
        embed_model=embed_model,
        ollama_base_url=ollama_base_url,
    )
    
    # Add documents in batches
    console.print("[cyan]Adding documents to vector store...[/cyan]")
    
    for i in tqdm(range(0, len(docs), batch_size), desc="Indexing"):
        batch = docs[i : i + batch_size]
        vs.add_documents(batch)
    
    console.print(f"[bold green]✓ Indexed {len(docs)} documents[/bold green]")
    return len(docs)

