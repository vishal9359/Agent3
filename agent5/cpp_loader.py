"""
C++ project loader with AST-aware chunking.

This module loads C++ projects and creates semantic chunks using AST analysis.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from agent5.ast_chunker import chunk_cpp_file
from agent5.fs_utils import CPP_EXTS, RAG_EXTRA_EXTS, iter_files, safe_read_text


def build_cpp_documents(
    project_path: Path,
    *,
    scope: Path | None = None,
    exclude_dir_names: set[str] | None = None,
    use_ast_chunking: bool = True,
) -> list[Document]:
    """
    Build LangChain documents from a C++ project.
    
    Args:
        project_path: Root path of the project
        scope: Optional scope path to limit indexing
        exclude_dir_names: Directory names to exclude
        use_ast_chunking: Use AST-aware chunking (recommended)
        
    Returns:
        List of LangChain Document objects
    """
    project_path = project_path.resolve()
    scope_path = (scope or project_path).resolve()
    
    if not scope_path.exists():
        raise FileNotFoundError(f"Scope path not found: {scope_path}")
    if not project_path.exists():
        raise FileNotFoundError(f"Project path not found: {project_path}")
    
    docs: list[Document] = []
    
    # Process C++ files with AST chunking
    for rec in iter_files(scope_path, include_exts=CPP_EXTS, exclude_dir_names=exclude_dir_names):
        if use_ast_chunking:
            # Use AST-aware chunking
            chunks = chunk_cpp_file(rec.path, rec.relpath)
            
            for chunk in chunks:
                docs.append(
                    Document(
                        page_content=chunk.content,
                        metadata={
                            "source": str(chunk.file_path),
                            "relpath": chunk.rel_path,
                            "chunk_type": chunk.chunk_type,
                            "name": chunk.name,
                            "qualified_name": chunk.qualified_name,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "dependencies": chunk.dependencies,
                            **chunk.metadata,
                        },
                    )
                )
        else:
            # Fallback to simple text loading
            text = safe_read_text(rec.path)
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(rec.path),
                            "relpath": rec.relpath,
                            "ext": rec.path.suffix.lower(),
                            "chunk_type": "file",
                        },
                    )
                )
    
    # Process additional files (README, CMakeLists.txt, etc.) without AST chunking
    for rec in iter_files(
        scope_path, include_exts=RAG_EXTRA_EXTS, exclude_dir_names=exclude_dir_names
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
                    "chunk_type": "documentation",
                },
            )
        )
    
    return docs


def chunk_documents_simple(
    docs: list[Document],
    *,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Simple text-based chunking for documents (fallback/legacy).
    
    This is kept for compatibility but AST chunking is preferred for C++ code.
    
    Args:
        docs: List of documents to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked documents
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
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

