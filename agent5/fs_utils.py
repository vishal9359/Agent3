"""Filesystem utilities for working with C++ projects."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# C++ file extensions
CPP_EXTS = {".cpp", ".cc", ".cxx", ".c++", ".hpp", ".h", ".hh", ".hxx", ".h++", ".inl", ".ipp"}

# Additional extensions for RAG context
RAG_EXTRA_EXTS = {".txt", ".md", ".cmake", ".mk", ".makefile"}

# Directories to exclude by default
DEFAULT_EXCLUDE_DIRS = {
    "build",
    "builds",
    "dist",
    "out",
    "target",
    "node_modules",
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "venv",
    ".venv",
    "env",
    ".env",
}


@dataclass(frozen=True)
class FileRecord:
    """Record of a file in the project."""
    path: Path
    relpath: str


def iter_files(
    root: Path,
    *,
    include_exts: set[str] | None = None,
    exclude_dir_names: set[str] | None = None,
) -> Iterable[FileRecord]:
    """
    Recursively iterate over files in a directory tree.
    
    Args:
        root: Root directory to search
        include_exts: Set of file extensions to include (e.g., {'.cpp', '.h'})
        exclude_dir_names: Set of directory names to skip
        
    Yields:
        FileRecord objects for matching files
    """
    root = root.resolve()
    exclude_dirs = exclude_dir_names or DEFAULT_EXCLUDE_DIRS
    
    if not root.exists():
        return
    
    if root.is_file():
        if include_exts is None or root.suffix.lower() in include_exts:
            yield FileRecord(path=root, relpath=root.name)
        return
    
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        
        # Skip excluded directories
        if any(parent.name in exclude_dirs for parent in path.parents):
            continue
        
        # Check extension
        if include_exts is not None and path.suffix.lower() not in include_exts:
            continue
        
        try:
            relpath = str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            relpath = path.name
        
        yield FileRecord(path=path, relpath=relpath)


def safe_read_text(path: Path, *, encoding: str = "utf-8") -> str:
    """
    Safely read text from a file, returning empty string on error.
    
    Args:
        path: Path to file
        encoding: Text encoding
        
    Returns:
        File contents or empty string
    """
    try:
        return path.read_text(encoding=encoding, errors="ignore")
    except Exception:
        return ""

